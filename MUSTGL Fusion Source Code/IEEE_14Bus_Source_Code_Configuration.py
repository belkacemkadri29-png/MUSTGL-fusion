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

# ============================================================================
# DEVICE — DÉTECTION AUTOMATIQUE GPU (A100 / T4 / CPU fallback)
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[DEVICE] Utilisation de : {DEVICE}")
if torch.cuda.is_available():
    print(f"[DEVICE] GPU détecté : {torch.cuda.get_device_name(0)}")
    print(f"[DEVICE] Mémoire totale : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # ------------------------------------------------------------------
    # Déterminisme strict sur GPU : garantit que deux runs GPU avec la
    # même seed donnent EXACTEMENT les mêmes résultats (utile pour ton
    # évaluation multi-seed). N'assure PAS l'identité bit-à-bit avec le
    # CPU (l'ordre de calcul GPU/CPU diffère structurellement), mais
    # élimine la variance résiduelle entre deux exécutions GPU.
    # Doit être fait AVANT toute allocation/opération CUDA.
    # ------------------------------------------------------------------
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # warn_only=True : certaines opérations (ex. certains backwards de BatchNorm/
    # matrix_power sur GPU) n'ont pas encore d'implémentation déterministe dans
    # PyTorch et lèveraient une RuntimeError bloquante avec warn_only=False.
    # warn_only=True applique le déterminisme partout où c'est possible et se
    # contente d'un avertissement (au lieu de planter) pour le reste — c'est le
    # réglage recommandé pour un pipeline complexe comme celui-ci.
    torch.use_deterministic_algorithms(True, warn_only=True)
    print("[DEVICE] Mode déterministe GPU activé (CUBLAS_WORKSPACE_CONFIG + use_deterministic_algorithms, warn_only=True)")

# ============================================================================
# SEED — FIXE L'INITIALISATION DES PARAMÈTRES (GPU T4/A100 Colab)
# ============================================================================

def set_seeds(seed=215):
    """Fixe toutes les sources d'aléatoire pour reproductibilité sur GPU"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Liste des seeds à évaluer (peut être surchargée à l'appel de run_multi_seed_evaluation)
DEFAULT_SEEDS = [215, 1024, 702, 31, 75]

# Appel global au démarrage (sera re-fixée à chaque fit() de toute façon)
set_seeds(DEFAULT_SEEDS[0])

# ============================================================================
# ECHO STATE NETWORK IMPLEMENTATION — VERSION 100% TORCH / GPU
# ============================================================================
#
# CHANGEMENT CLÉ vs version CPU :
#   L'ancienne implémentation faisait, à CHAQUE pas de temps :
#       input_np = input_t.detach().cpu().numpy()   <-- va-et-vient GPU->CPU
#       np.dot(...)                                  <-- calcul en NumPy (CPU)
#       torch.tensor(...)                             <-- retour CPU->GPU
#   C'était donc systématiquement bloqué sur CPU, quel que soit le device
#   demandé, et en plus très lent (sync GPU<->CPU par pas de temps).
#
#   Ici, W_in / W_reservoir / bias / state sont des tenseurs torch qui
#   restent sur `device` du début à la fin. Le calcul (matmul + tanh)
#   tourne entièrement sur GPU si device='cuda'. On ne repasse en NumPy
#   qu'une seule fois à la fin (pour Ridge / IsolationForest, qui sont
#   des algos sklearn, donc CPU par nature).
# ============================================================================

class EchoStateNetwork:
    """Echo State Network (ESN) pour l'extraction de représentations temporelles — torch/GPU"""

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

        # Matrice de poids d'entrée — initialisée avec seed fixée (NumPy pour rester
        # 100% identique bit-à-bit à la version CPU d'origine, puis transférée sur device)
        W_in = np.random.uniform(
            -input_scaling,
            input_scaling,
            (reservoir_size, input_size)
        )

        # Matrice de poids récurrente — initialisée avec seed fixée
        W_reservoir = np.random.randn(reservoir_size, reservoir_size)
        mask = np.random.rand(reservoir_size, reservoir_size) > sparsity
        W_reservoir[~mask] = 0

        eigenvalues = np.linalg.eigvals(W_reservoir)
        current_spectral_radius = np.max(np.abs(eigenvalues))

        if current_spectral_radius > 0:
            W_reservoir = W_reservoir * (spectral_radius / current_spectral_radius)

        bias = np.random.uniform(-0.1, 0.1, reservoir_size)

        # --- Tout le calcul se fait ensuite en torch, sur `device` ---
        self.W_in = torch.tensor(W_in, dtype=torch.float32, device=device)
        self.W_reservoir = torch.tensor(W_reservoir, dtype=torch.float32, device=device)
        self.bias = torch.tensor(bias, dtype=torch.float32, device=device)

        self.state = torch.zeros(reservoir_size, dtype=torch.float32, device=device)

    def reset_state(self):
        """Réinitialise l'état interne du réservoir"""
        self.state = torch.zeros(self.reservoir_size, dtype=torch.float32, device=self.device)

    def forward_step(self, input_t: torch.Tensor) -> torch.Tensor:
        """Un pas de temps du réservoir — reste entièrement sur `device`, aucun aller-retour CPU"""
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
        """Traite une séquence complète, séquentiellement (dépendance temporelle),
        mais sans jamais quitter le GPU entre les pas de temps."""
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
    """Wrapper pour utiliser l'ESN dans le pipeline de détection d'anomalies"""

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

        # Readout layer — entraîné avec régression Ridge (sklearn = CPU par nature)
        self.readout = Ridge(alpha=default_config['ridge_alpha'])
        self.is_trained = False

    def train_readout(self, X: torch.Tensor, target: torch.Tensor = None):
        """Entraîne la couche readout avec régression linéaire (Ridge).
        Le passage dans le réservoir tourne sur GPU ; seul le fit final de
        Ridge (sklearn) nécessite un passage en NumPy/CPU, une seule fois."""
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
        """Interface compatible avec le code existant"""
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


# ============================================================================
# FEATURE INDICES (57 bus system)
# ============================================================================

INJ_IDX   = list(range(0,118))
FLOW_IDX  = list(range(118,186))
THETA_IDX = list(range(186,422))

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


# ============================================================================
# ANOMALY DETECTION PIPELINE WITH ESN — GPU AWARE
# ============================================================================

class AnomalyDetectionPipeline:
    """Pipeline intégré pour la détection d'anomalies spatiotemporelle avec ESN"""

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
        self.seed = None  # dernière seed utilisée pour fit()

    def learn_adjacency_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """Étape 1: Apprentissage de la matrice d'adjacence"""
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
        """Étape 2: Apprentissage de la représentation spatiale (GCN)"""
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
        """Étape 3: Apprentissage de la représentation temporelle (ESN)"""
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
        """Étape 4: Entraînement du réseau de fusion"""
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
        """Configuration du détecteur d'anomalies (Unsupervised).
        IsolationForest est un algo sklearn : il ne tourne que sur CPU/NumPy,
        donc on convertit ici, une seule fois, en toute fin de pipeline GPU."""
        residuals = (X_train - H_fused_train).detach()

        self.residuals_std = residuals.std(dim=0, unbiased=True)
        self.residuals_std[self.residuals_std == 0] = 1e-6
        residuals_normalized = residuals / self.residuals_std

        residuals_np = residuals_normalized.cpu().numpy()

        print(f"\n  DIAGNOSTIC RÉSIDUS TRAIN :")
        print(f"  Mean abs résidus : {np.abs(residuals_np).mean():.4f}")
        print(f"  Std résidus      : {residuals_np.std():.4f}")
        print(f"  Max abs résidus  : {np.abs(residuals_np).max():.4f}")

        # ✅ random_state = seed courante (celle passée à fit()) — reproductible par seed
        self.isolation_forest = IsolationForest(
            contamination='auto',
            n_estimators=100,
            max_samples='auto',
            random_state=self.seed
        )

        self.isolation_forest.fit(residuals_np)

        return residuals_np

    def detect_anomalies(self, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Détection d'anomalies sur nouvelles données"""
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
        """Calcul de la représentation spatiale pour nouvelles données"""
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
        """Calcul de la représentation temporelle pour nouvelles données"""
        with torch.no_grad():
            H_temporal = self.esn_model.forward(X)
        return H_temporal

    def fit(self, X_train: np.ndarray, seed: int = 215):
        """Entraînement complet du pipeline pour une seed donnée"""

        self.seed = seed
        # ✅ SEED FIXÉE ICI — avant toute initialisation de paramètres
        set_seeds(seed)

        start_time = time.time()
        #self.clip_low  = np.percentile(X_train, 3, axis=0)
        #self.clip_high = np.percentile(X_train, 97, axis=0)
        #X_train_clipped = np.clip(X_train, self.clip_low, self.clip_high)

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
        #X_test_clipped = np.clip(X_test, self.clip_low, self.clip_high)

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
        """Calcul des métriques de performance"""
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


# ============================================================================
# ÉVALUATION SUR UNE SEULE SEED (inchangé, réutilisable indépendamment)
# ============================================================================

def run_complete_evaluation(train_path: str, test_path: str, label_column: str,
                             config: Dict, seed: int = 215, device: torch.device = DEVICE):
    """Évaluation complète du pipeline pour une seed unique"""

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


# ============================================================================
# ÉVALUATION MULTI-SEED
# ============================================================================

def run_multi_seed_evaluation(train_path: str, test_path: str, label_column: str,
                               config: Dict, seeds: List[int] = None,
                               output_path: str = 'anomaly_detection_results_multiseed.xlsx',
                               device: torch.device = DEVICE):
    """
    Lance le pipeline complet (fit + predict + evaluate) une fois par seed,
    puis agrège les métriques (moyenne, écart-type, min, max) sur l'ensemble
    des seeds.

    Les données sont chargées une seule fois. Seule l'initialisation aléatoire
    du pipeline (poids, ESN, Isolation Forest) change à chaque seed.
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    # Chargement des données une seule fois
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

    # ---- Agrégation des métriques sur toutes les seeds ----
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

    # ---- Sauvegarde Excel : un onglet par seed + résumé ----
    with pd.ExcelWriter(output_path) as writer:
        per_seed_df.to_excel(writer, sheet_name='per_seed_metrics', index=False)
        summary_df.to_excel(writer, sheet_name='summary', index=False)
        for seed, df in all_results.items():
            sheet_name = f'predictions_seed_{seed}'[:31]  # limite Excel = 31 car.
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nRésultats sauvegardés dans : {output_path}")

    return pipelines, per_seed_df, summary_df




if __name__ == "__main__":
    train_path = "NEW_118bussystem30000sampl118.xlsx"
    test_path = "NEW50%118_INJECTIONKADRIBALANCEDstate_50_test_with_attacks_case57.xlsx"

    config = {
        "embedding_dim": 512,
        "attention_dim": 64,
        "adj_epochs": 10,
        "adj_learning_rate": 0.01,
        "gcn_hops": 4,
        "hidden_dim": 422,
        "gcn_epochs": 100,
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
        output_path='anomaly_detection_results_multiseed.xlsx',
        device=DEVICE
    )
