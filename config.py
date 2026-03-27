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
    pc_alpha:     float = 0.1     # lr de la boucle d'inférence (stable : α < 2/||H||_2 ≈ 0.22)
    pc_tol:       float = 0.1     # critère MSE d'arrêt atteignable avec l'échelle d'erreur actuelle
    pc_n_inference_steps: int = 50 # nb fixe de pas d'inférence (lax.scan, coût constant)
    pc_init_mode: str = "zeros"   # "zeros" (robuste, Rao & Ballard) | "feedforward" (pseudo-inverse)

    # Predictor (Transformer + MLP)
    pred_k_embed: int   = 16      # dim embedding de l'horizon k
    pred_mlp_dim: int   = 512     # dim couches cachées MLP head
    pred_K:       int   = 5       # horizon max de prédiction

    # Pertes
    lambda_pc:    float = 0.01    # poids L_PC
    lambda_pc_l2: float = 1e-3    # L2 sur W_pred : filet de sécurité contre ||W||_2 > sqrt(2/α-1)
    lambda_var:   float = 0.1     # poids L_var (0.01 → 0.1 : empêche collapse différentiel selon n)
    gamma_var:    float = 1.0     # variance cible (anti-collapse)
    prec_alpha:   float = 0.0     # curriculum divisif : 0=standard, 1=divisif pur

    # Anti-collapse — mode d'ablation (tableau comparatif contre LeWM)
    #   "var"     : L_var = max(0, γ - Var(z))        [défaut, ce papier]
    #   "sigreg"  : SIGReg Gaussien isotrope          [baseline LeWM]
    #   "pc_only" : aucun terme anti-collapse explicite
    # Le poids est toujours lambda_var, quelle que soit la valeur de ce flag.
    anti_collapse_mode: str = "var"
    sigreg_n_proj:      int = 64   # projections aléatoires pour anti_collapse_mode="sigreg"

    # EMA — optionnel (désactivé par défaut, activable pour ablation)
    # La boucle d'inférence PC (Boucle 1) est le mécanisme anti-collapse principal.
    # L'EMA peut être activé pour ablation : comparer PC vs PC+EMA.
    use_ema:   bool  = False     # activer l'EMA sur l'encodeur cible
    ema_tau:   float = 0.996     # momentum EMA (anneal vers 1.0 via cosine schedule)

    # Transformer (predictor PC-JEPA et baseline partagent ces hyperparamètres)
    trans_n_layers: int = 1       # 1 couche → ~935K params (parité ≤25%)
    trans_n_heads:  int = 4       # têtes d'attention (d_z doit être divisible)
    trans_ffn_dim:  int = 256     # dim FFN interne

    # Optimisation (Boucle 2)
    learning_rate: float = 1e-3
    n_epochs:      int   = 100
    warmup_steps:  int   = 500

    # Reproductibilité (R5)
    seed: int = 42

    # Phase 1 — PC-JEPA v2
    use_pc_errors_in_predictor: bool = False
