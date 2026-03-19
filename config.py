"""
=== SPEC [config] ===
Entrées  : Hyperparamètres du projet PC-JEPA
Sorties  : ModelConfig (NamedTuple immuable)
Invariants:
  - Tous les hyperparamètres vivent ici (R7 — zéro magic number)
  - Immuable via NamedTuple (R4)
  - Aucun code métier ici
=== FIN SPEC ===
"""
from typing import NamedTuple


class ModelConfig(NamedTuple):
    # Espace latent
    d_z: int = 256            # dimension des représentations latentes

    # Encoder CNN
    enc_channels: tuple = (32, 64, 128)
    enc_kernel:   int   = 4
    enc_stride:   int   = 2

    # Predictive Coding
    pc_n_layers:  int   = 3       # profondeur de la hiérarchie PC
    pc_alpha:     float = 0.1     # lr de la boucle d'inférence
    pc_tol:       float = 1e-3    # critère d'arrêt de convergence
    pc_max_iter:  int   = 20      # nb max d'itérations d'inférence

    # Predictor (GRU + MLP)
    pred_hidden:  int   = 256     # dim hidden du GRU
    pred_k_embed: int   = 16      # dim embedding de l'horizon k
    pred_mlp_dim: int   = 512     # dim couches cachées MLP head
    pred_K:       int   = 5       # horizon max de prédiction

    # Pertes
    lambda_pc:  float = 0.1       # poids L_PC
    lambda_var: float = 0.01      # poids L_var
    gamma_var:  float = 1.0       # variance cible (anti-collapse)

    # Transformer baseline (parité paramètres avec PC-JEPA predictor ~930K)
    trans_n_layers: int = 1       # 1 couche → ~930K params (parité avec GRU+MLP)
    trans_n_heads:  int = 4       # têtes d'attention (d_z doit être divisible)
    trans_ffn_dim:  int = 256     # dim FFN interne (réduit pour parité)

    # Optimisation (Boucle 2)
    learning_rate: float = 3e-4
    n_epochs:      int   = 100
    warmup_steps:  int   = 500

    # Reproductibilité (R5)
    seed: int = 42
