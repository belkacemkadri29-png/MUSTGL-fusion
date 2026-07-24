import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
import time
from typing import Tuple, Dict, List

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"[DEVICE] Total memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)


def set_seeds(seed=215):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DEFAULT_SEEDS = [215, 1024, 702, 31, 75]

set_seeds(DEFAULT_SEEDS[0])


class EchoStateNetwork:
    def __init__(self,
                 input_size: int,
                 reservoir_size: int = 100,
                 output_size: int = None,
                 spectral_radius: float = 0.9,
                 sparsity: float = 0.9,
                 input_scaling: float = 1.0,
                 leak_rate: float = 0.3,
                 device: torch.device = DEVICE):

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size if output_size else reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.device = device

        W_in = np.random.uniform(
            -input_scaling,
            input_scaling,
            (reservoir_size, input_size)
        )

        W_reservoir = np.random.randn(reservoir_size, reservoir_size)
        mask = np.random.rand(reservoir_size, reservoir_size) > sparsity
        W_reservoir[~mask] = 0

        eigenvalues = np.linalg.eigvals(W_reservoir)
        current_spectral_radius = np.max(np.abs(eigenvalues))

        if current_spectral_radius > 0:
            W_reservoir = W_reservoir * (spectral_radius / current_spectral_radius)

        bias = np.random.uniform(-0.1, 0.1, reservoir_size)

        self.W_in = torch.tensor(W_in, dtype=torch.float32, device=device)
        self.W_reservoir = torch.tensor(W_reservoir, dtype=torch.float32, device=device)
        self.bias = torch.tensor(bias, dtype=torch.float32, device=device)

        self.state = torch.zeros(reservoir_size, dtype=torch.float32, device=device)

    def reset_state(self):
        self.state = torch.zeros(self.reservoir_size, dtype=torch.float32, device=self.device)

    def forward_step(self, input_t: torch.Tensor) -> torch.Tensor:
        if not isinstance(input_t, torch.Tensor):
            input_t = torch.tensor(input_t, dtype=torch.float32)
        input_t = input_t.to(self.device)

        pre_activation = (
            torch.matmul(self.W_in, input_t) +
            torch.matmul(self.W_reservoir, self.state) +
            self.bias
        )

        activation = torch.tanh(pre_activation)
        self.state = (1 - self.leak_rate) * self.state + self.leak_rate * activation

        return self.state

    def forward_sequence(self, X: torch.Tensor, return_states: bool = True) -> torch.Tensor:
        X = X.to(self.device)
        sequence_length = X.shape[0]
        self.reset_state()

        if return_states:
            states = []
            for t in range(sequence_length):
                state_t = self.forward_step(X[t])
                states.append(state_t)
            return torch.stack(states)
        else:
            state_t = self.state
            for t in range(sequence_length):
                state_t = self.forward_step(X[t])
            return state_t


class ESNTemporalRepresentation:
    def __init__(self,
                 input_size: int,
                 hidden_dim: int,
                 config: dict = None,
                 device: torch.device = DEVICE):

        default_config = {
            'reservoir_size': 200,
            'spectral_radius': 0.9,
            'sparsity': 0.9,
            'input_scaling': 1.0,
            'leak_rate': 0.3,
            'ridge_alpha': 1.0
        }

        if config:
            default_config.update(config)

        self.config = default_config
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.device = device

        self.esn = EchoStateNetwork(
            input_size=input_size,
            reservoir_size=default_config['reservoir_size'],
            output_size=hidden_dim,
            spectral_radius=default_config['spectral_radius'],
            sparsity=default_config['sparsity'],
            input_scaling=default_config['input_scaling'],
            leak_rate=default_config['leak_rate'],
            device=device
        )

        self.readout = Ridge(alpha=default_config['ridge_alpha'])
        self.is_trained = False

    def train_readout(self, X: torch.Tensor, target: torch.Tensor = None):
        print("  [Training ESN Readout...]")

        reservoir_states = self.esn.forward_sequence(X, return_states=True)
        reservoir_states_np = reservoir_states.detach().cpu().numpy()

        if target is None:
            target = X

        target_np = target.detach().cpu().numpy()

        self.readout.fit(reservoir_states_np, target_np)
        self.is_trained = True

        print("  [ESN Readout Trained]")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        reservoir_states = self.esn.forward_sequence(X, return_states=True)

        if self.is_trained:
            reservoir_states_np = reservoir_states.detach().cpu().numpy()
            H_temporal_np = self.readout.predict(reservoir_states_np)
            H_temporal = torch.tensor(H_temporal_np, dtype=torch.float32, device=self.device)
        else:
            if self.config['reservoir_size'] != self.hidden_dim:
                H_temporal = reservoir_states[:, :self.hidden_dim]
            else:
                H_temporal = reservoir_states

        return H_temporal


INJ_IDX   = list(range(0,14))
FLOW_IDX  = list(range(14,20))
THETA_IDX = list(range(20,48))

def physical_constraints_loss(A, lambda1=0.1, lambda2=0.1):
    inj   = torch.tensor(INJ_IDX,   dtype=torch.long, device=A.device)
    flow  = torch.tensor(FLOW_IDX,  dtype=torch.long, device=A.device)
    theta = torch.tensor(THETA_IDX, dtype=torch.long, device=A.device)

    A_inj_theta = A[inj][:, theta]
    A_theta_inj = A[theta][:, inj]
    symmetry_loss = torch.norm(A_inj_theta - A_theta_inj.T, p='fro') ** 2

    A_flow_inj  = A[flow][:, inj]
    causal_loss = torch.norm(A_flow_inj, p='fro') ** 2

    return lambda1 * symmetry_loss + lambda2 * causal_loss


class AnomalyDetectionPipeline:
    def __init__(self, config: Dict, device: torch.device = DEVICE):
        self.config = config
        self.device = device
        self.scaler = MinMaxScaler()
        self.adjacency_matrix = None
        self.fusion_model = None
        self.threshold = None
        self.W_inv = None
        self.training_time = 0
        self.testing_time = 0
        self.W1 = None
        self.W_Q = None
        self.W_K = None
        self.W_list = None
        self.alpha = None
        self.W_res = None
        self.decoder = None
        self.batch_norm = None
        self.dropout = None
        self.esn_model = None
        self.seed = None

    def learn_adjacency_matrix(self, X: torch.Tensor) -> torch.Tensor:
        print("\n[Adjacency Training]")
        X_transposed = X.T
        num_features, N = X_transposed.shape

        embedding_dim = self.config['embedding_dim']
        attention_dim = self.config['attention_dim']

        self.W1 = nn.Parameter(torch.randn(N, embedding_dim, device=self.device) * 0.01)
        self.W_Q = nn.Parameter(torch.randn(embedding_dim, attention_dim, device=self.device) * 0.01)
        self.W_K = nn.Parameter(torch.randn(embedding_dim, attention_dim, device=self.device) * 0.01)

        optimizer = torch.optim.Adam([self.W1, self.W_Q, self.W_K],
                                   lr=self.config['adj_learning_rate'])

        for epoch in range(self.config['adj_epochs']):
            optimizer.zero_grad()

            E = torch.tanh(torch.matmul(X_transposed, self.W1))
            Q = torch.matmul(E, self.W_Q)
            K = torch.matmul(E, self.W_K)
            attention_scores = torch.matmul(Q, K.T) / (attention_dim ** 0.5)
            A = F.softmax(attention_scores, dim=1)
            X_reconstructed = torch.matmul(X, A)
            lambda1 = self.config.get('lambda_symmetry', 0.1)
            lambda2 = self.config.get('lambda_causal',   0.1)
            reconstruction_loss = torch.mean((X_reconstructed - X) ** 2)
            phy_loss = physical_constraints_loss(A, lambda1, lambda2)
            loss = reconstruction_loss + phy_loss

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.config["adj_epochs"]}], '
                f'Recon: {reconstruction_loss.item():.6f}, '
                f'Phy: {phy_loss.item():.6f}')

        with torch.no_grad():
            E = torch.tanh(torch.matmul(X_transposed, self.W1))
            Q = torch.matmul(E, self.W_Q)
            K = torch.matmul(E, self.W_K)
            attention_scores = torch.matmul(Q, K.T) / (attention_dim ** 0.5)
            self.adjacency_matrix = F.softmax(attention_scores, dim=1)

        return self.adjacency_matrix

    def train_spatial_representation(self, X: torch.Tensor) -> torch.Tensor:
        print("\n[Spatial Training]")
        N, num_features = X.shape
        A = self.adjacency_matrix
        K = self.config['gcn_hops']
        hidden_dim = self.config['hidden_dim']

        self.W_list = nn.ParameterList(
            [nn.Parameter(torch.randn(num_features, hidden_dim, device=self.device)) for _ in range(K)]
        )
        self.alpha = nn.Parameter(torch.randn(K, device=self.device))
        self.W_res = nn.Parameter(torch.randn(num_features, hidden_dim, device=self.device))
        self.decoder = nn.Parameter(torch.randn(hidden_dim, num_features, device=self.device))
        self.batch_norm = nn.BatchNorm1d(hidden_dim).to(self.device)
        self.dropout = nn.Dropout(self.config['dropout_rate'])

        params = list(self.W_list) + [self.alpha, self.W_res, self.decoder] + list(self.batch_norm.parameters())
        optimizer = optim.Adam(params, lr=self.config['gcn_learning_rate'])

        A_k_list = [torch.matrix_power(A, k+1) for k in range(K)]

        for epoch in range(self.config['gcn_epochs']):
            optimizer.zero_grad()

            H = torch.zeros(N, hidden_dim, device=self.device)
            for k in range(K):
                support = X @ A_k_list[k] @ self.W_list[k]
                H += self.alpha[k] * support
            H += X @ self.W_res
            H = self.batch_norm(H)
            H = self.dropout(H)
            H = torch.sigmoid(H)
            output = H @ self.decoder

            reconstruction_loss = F.mse_loss(output, X)
            variance = H.var(dim=0).mean()
            diversity_loss = 1.0 / (variance + 1e-6)
            loss = reconstruction_loss + self.config['lambda_reg'] * diversity_loss

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{self.config["gcn_epochs"]}], Loss: {loss.item():.6f}')

        return H.detach()

    def train_temporal_representation(self, X: torch.Tensor) -> torch.Tensor:
        print("\n[ESN Training]")
        N, num_features = X.shape
        hidden_dim = self.config['hidden_dim']

        esn_config = {
            'reservoir_size': self.config.get('esn_reservoir_size', 200),
            'spectral_radius': self.config.get('esn_spectral_radius', 0.9),
            'sparsity': self.config.get('esn_sparsity', 0.9),
            'input_scaling': self.config.get('esn_input_scaling', 1.0),
            'leak_rate': self.config.get('esn_leak_rate', 0.3),
            'ridge_alpha': self.config.get('esn_ridge_alpha', 1.0)
        }

        self.esn_model = ESNTemporalRepresentation(
            input_size=num_features,
            hidden_dim=hidden_dim,
            config=esn_config,
            device=self.device
        )

        self.esn_model.train_readout(X)

        with torch.no_grad():
            H_temporal = self.esn_model.forward(X)

        return H_temporal

    def train_fusion_network(self, H_spatial: torch.Tensor, H_temporal: torch.Tensor,
                           X: torch.Tensor) -> torch.Tensor:
        print("\n[Fusion]")
        class AdaptiveFusionNN(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim*2, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc_out = nn.Linear(hidden_dim, input_dim)

            def forward(self, H_spatial, H_temporal):
                H_concat = torch.cat([H_spatial, H_temporal], dim=1)
                h1 = F.relu(self.fc1(H_concat))
                h2 = F.relu(self.fc2(h1))
                y_out = self.fc_out(h2)
                return y_out

        N, m = H_spatial.shape
        fusion_hidden_dim = self.config['fusion_hidden_dim']

        self.fusion_model = AdaptiveFusionNN(m, fusion_hidden_dim).to(self.device)
        optimizer = optim.Adam(self.fusion_model.parameters(),
                             lr=self.config['fusion_learning_rate'])

        for epoch in range(self.config['fusion_epochs']):
            optimizer.zero_grad()
            H_fused = self.fusion_model(H_spatial, H_temporal)
            loss = F.mse_loss(H_fused, X)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{self.config["fusion_epochs"]}], Loss: {loss.item():.6f}')

        with torch.no_grad():
            H_fused = self.fusion_model(H_spatial, H_temporal)

        return H_fused

    def setup_anomaly_detector_unsupervised(self, X_train: torch.Tensor, H_fused_train: torch.Tensor):
        residuals = (X_train - H_fused_train).detach()

        self.residuals_std = residuals.std(dim=0, unbiased=True)
        self.residuals_std[self.residuals_std == 0] = 1e-6
        residuals_normalized = residuals / self.residuals_std

        residuals_np = residuals_normalized.cpu().numpy()

        print(f"\n  DIAGNOSTIC RÉSIDUS TRAIN :")
        print(f"  Mean abs résidus : {np.abs(residuals_np).mean():.4f}")
        print(f"  Std résidus      : {residuals_np.std():.4f}")
        print(f"  Max abs résidus  : {np.abs(residuals_np).max():.4f}")

        self.isolation_forest = IsolationForest(
            contamination='auto',
            n_estimators=100,
            max_samples='auto',
            random_state=self.seed
        )

        self.isolation_forest.fit(residuals_np)

        return residuals_np

    def detect_anomalies(self, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        H_spatial = self._get_spatial_representation(X)
        H_temporal = self._get_temporal_representation(X)

        with torch.no_grad():
            H_fused = self.fusion_model(H_spatial, H_temporal)

        residuals = (X - H_fused).detach()
        residuals_normalized = residuals / self.residuals_std
        residuals_np = residuals_normalized.cpu().numpy()

        print(f"\n  DIAGNOSTIC RÉSIDUS TEST :")
        print(f"  Mean abs résidus : {np.abs(residuals_np).mean():.4f}")
        print(f"  Std résidus      : {residuals_np.std():.4f}")
        print(f"  Max abs résidus  : {np.abs(residuals_np).max():.4f}")

        predictions = self.isolation_forest.predict(residuals_np)
        anomalies = (predictions == -1)
        IF_scores = -self.isolation_forest.decision_function(residuals_np)

        return anomalies, IF_scores

    def _get_spatial_representation(self, X: torch.Tensor) -> torch.Tensor:
        N, num_features = X.shape
        A = self.adjacency_matrix
        K = self.config['gcn_hops']
        hidden_dim = self.config['hidden_dim']

        A_k_list = [torch.matrix_power(A, k+1) for k in range(K)]

        with torch.no_grad():
            H = torch.zeros(N, hidden_dim, device=self.device)
            for k in range(K):
                support = X @ A_k_list[k] @ self.W_list[k]
                H += self.alpha[k] * support
            H += X @ self.W_res
            H = self.batch_norm(H)
            H = torch.sigmoid(H)

        return H

    def _get_temporal_representation(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            H_temporal = self.esn_model.forward(X)
        return H_temporal

    def fit(self, X_train: np.ndarray, seed: int = 215):
        self.seed = seed
        set_seeds(seed)

        start_time = time.time()

        X_normalized = self.scaler.fit_transform(X_train)

        X = torch.tensor(X_normalized, dtype=torch.float32, device=self.device)

        self.learn_adjacency_matrix(X)
        H_spatial = self.train_spatial_representation(X)
        H_temporal = self.train_temporal_representation(X)
        H_fused = self.train_fusion_network(H_spatial, H_temporal, X)
        train_scores = self.setup_anomaly_detector_unsupervised(X, H_fused)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        self.training_time = time.time() - start_time
        return train_scores

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        X_normalized = self.scaler.transform(X_test)

        print(f"\n  DIAGNOSTIC SCALER TEST :")
        print(f"  Min après transform : {X_normalized.min():.4f}")
        print(f"  Max après transform : {X_normalized.max():.4f}")

        n_below_0 = (X_normalized < 0).sum()
        n_above_1 = (X_normalized > 1).sum()
        total = X_normalized.size
        print(f"  Valeurs < 0 : {n_below_0} ({100*n_below_0/total:.2f}%)")
        print(f"  Valeurs > 1 : {n_above_1} ({100*n_above_1/total:.2f}%)")
        if n_below_0 + n_above_1 > 0:
           out_of_range_per_col = ((X_normalized < 0) | (X_normalized > 1)).sum(axis=0)
           worst_cols = np.argsort(out_of_range_per_col)[::-1][:10]
           print(f"  Colonnes les plus impactées (index) : {worst_cols}")
           print(f"  Valeurs extrêmes (index) : {worst_cols} -> max={X_normalized[:, worst_cols].max(axis=0)}")

        X = torch.tensor(X_normalized, dtype=torch.float32, device=self.device)
        anomalies, scores = self.detect_anomalies(X)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        self.testing_time = time.time() - start_time
        return anomalies, scores

    def evaluate_performance(self, y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, scores)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'specificity': specificity,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'training_time': self.training_time,
            'testing_time': self.testing_time,
            'confusion_matrix': cm
        }

        return metrics


def run_complete_evaluation(train_path: str, test_path: str, label_column: str,
                             config: Dict, seed: int = 215, device: torch.device = DEVICE):
    df_train_raw = pd.read_excel(train_path)
    X_train = df_train_raw.fillna(0).values.astype(float)

    df_test_raw = pd.read_excel(test_path)

    if label_column not in df_test_raw.columns:
        possible_labels = ['label', 'Label', 'LABEL', 'attack_label', 'class', 'Class', 'target', 'y']
        for possible in possible_labels:
            if possible in df_test_raw.columns:
                label_column = possible
                break

    y_test = df_test_raw[label_column].values
    df_test_features = df_test_raw.drop(columns=[label_column])
    X_test = df_test_features.fillna(0).values.astype(float)

    if X_train.shape[1] != X_test.shape[1]:
        min_features = min(X_train.shape[1], X_test.shape[1])
        X_train = X_train[:, :min_features]
        X_test = X_test[:, :min_features]

    pipeline = AnomalyDetectionPipeline(config, device=device)

    train_scores = pipeline.fit(X_train, seed=seed)
    y_pred, test_scores = pipeline.predict(X_test)

    metrics = pipeline.evaluate_performance(y_test, y_pred.astype(int), test_scores)

    print("\n" + "="*60)
    print(f"RÉSULTATS — SEED {seed}")
    print("="*60)
    print(f"Training Time: {metrics['training_time']:.2f}s")
    print(f"Testing Time:  {metrics['testing_time']:.2f}s")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"F1-Score:      {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:       {metrics['roc_auc']:.4f}")
    print("="*60)

    results_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred.astype(int),
        'anomaly_score': test_scores
    })

    return pipeline, metrics, results_df


def run_multi_seed_evaluation(train_path: str, test_path: str, label_column: str,
                               config: Dict, seeds: List[int] = None,
                               device: torch.device = DEVICE):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    df_train_raw = pd.read_excel(train_path)
    X_train_full = df_train_raw.fillna(0).values.astype(float)

    df_test_raw = pd.read_excel(test_path)

    if label_column not in df_test_raw.columns:
        possible_labels = ['label', 'Label', 'LABEL', 'attack_label', 'class', 'Class', 'target', 'y']
        for possible in possible_labels:
            if possible in df_test_raw.columns:
                label_column = possible
                break

    y_test = df_test_raw[label_column].values
    df_test_features = df_test_raw.drop(columns=[label_column])
    X_test_full = df_test_features.fillna(0).values.astype(float)

    if X_train_full.shape[1] != X_test_full.shape[1]:
        min_features = min(X_train_full.shape[1], X_test_full.shape[1])
        X_train_full = X_train_full[:, :min_features]
        X_test_full = X_test_full[:, :min_features]

    all_metrics = []
    all_results = {}
    pipelines = {}

    for seed in seeds:
        print("\n" + "#"*70)
        print(f"#  DÉMARRAGE RUN — SEED = {seed}")
        print("#"*70)

        pipeline = AnomalyDetectionPipeline(config, device=device)
        pipeline.fit(X_train_full, seed=seed)
        y_pred, test_scores = pipeline.predict(X_test_full)

        metrics = pipeline.evaluate_performance(y_test, y_pred.astype(int), test_scores)
        metrics['seed'] = seed
        all_metrics.append(metrics)

        results_df = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': y_pred.astype(int),
            'anomaly_score': test_scores
        })
        all_results[seed] = results_df
        pipelines[seed] = pipeline

        print("\n" + "-"*60)
        print(f"RÉSULTATS — SEED {seed}")
        print("-"*60)
        print(f"Training Time: {metrics['training_time']:.2f}s")
        print(f"Testing Time:  {metrics['testing_time']:.2f}s")
        print(f"Precision:     {metrics['precision']:.4f}")
        print(f"Recall:        {metrics['recall']:.4f}")
        print(f"F1-Score:      {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:       {metrics['roc_auc']:.4f}")
        print(f"Accuracy:      {metrics['accuracy']:.4f}")
        print("-"*60)

    metric_cols = ['precision', 'recall', 'f1_score', 'roc_auc', 'accuracy',
                    'specificity', 'training_time', 'testing_time']

    per_seed_df = pd.DataFrame(all_metrics)[['seed'] + metric_cols]

    summary_rows = []
    for col in metric_cols:
        vals = per_seed_df[col].values.astype(float)
        summary_rows.append({
            'metric': col,
            'mean': vals.mean(),
            'std': vals.std(ddof=1) if len(vals) > 1 else 0.0,
            'min': vals.min(),
            'max': vals.max()
        })
    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "="*70)
    print("RÉSULTATS PAR SEED")
    print("="*70)
    print(per_seed_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "="*70)
    print(f"RÉSUMÉ AGRÉGÉ SUR {len(seeds)} SEEDS {seeds}")
    print("="*70)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("="*70)

    return pipelines, per_seed_df, summary_df


if __name__ == "__main__":
    train_path = "595FINAL_TRAINING_14_BUS_SYSTEM.xlsx"
    test_path = "595test_clipped_p2.5_p95.xlsx"

    config = {
        "embedding_dim": 512,
        "attention_dim": 64,
        "adj_epochs": 10,
        "adj_learning_rate": 0.01,
        "gcn_hops": 3,
        "hidden_dim": 48,
        "gcn_epochs": 50,
        "gcn_learning_rate": 0.01,
        "dropout_rate": 0.5,
        "lambda_reg": 0.1,
        "fusion_hidden_dim": 64,
        "fusion_epochs": 50,
        "fusion_learning_rate": 0.01,
        "esn_reservoir_size": 200,
        "esn_spectral_radius": 0.96,
        "esn_sparsity": 0.96,
        "esn_input_scaling": 1,
        "esn_leak_rate": 0.7,
        "esn_ridge_alpha": 1,
        "lambda_symmetry": 0.1,
        "lambda_causal": 0.1
    }

    SEEDS = [215, 604, 948, 382, 123]

    pipelines, per_seed_df, summary_df = run_multi_seed_evaluation(
        train_path=train_path,
        test_path=test_path,
        label_column='attack_label',
        config=config,
        seeds=SEEDS,
        device=DEVICE
    )
