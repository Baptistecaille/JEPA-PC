"""
=== SPEC [losses] ===
Entrées  : prédictions z_pred (B,K,d_z), cibles z_target (B,K,d_z) stop_grad,
           état PC convergé, config
Sorties  : scalaires de perte + dict détaillé
Invariants:
  - z_target DOIT être stop_gradient avant d'être passé à loss_jepa (R3)
    Cette fonction ne fait PAS le stop_gradient — responsabilité de trainer.py
  - Aucun magic number (R7)
  - Toutes les pertes sont différentiables
=== FIN SPEC ===
"""
import jax
import jax.numpy as jnp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.pc_nodes import PCHierarchyState
    from config import ModelConfig


# ---------------------------------------------------------------------------
# L_JEPA — perte principale de prédiction latente
# ---------------------------------------------------------------------------

def loss_jepa(z_pred: jnp.ndarray, z_target: jnp.ndarray) -> jnp.ndarray:
    """
    L_JEPA = (1/K) Σ_k ||ẑ_{t+k} - sg(z_{t+k})||²

    CRITIQUE (R3) : z_target DOIT être stop_gradient avant d'être passé ici.
    Cette fonction ne fait PAS le stop_gradient — c'est la responsabilité
    de l'appelant (trainer.py). Design explicite pour forcer la conscience.

    z_pred   : (B, K, d_z)
    z_target : (B, K, d_z)  — DOIT être stop_gradient
    return   : scalaire float32
    """
    return jnp.mean((z_pred - z_target) ** 2)


# ---------------------------------------------------------------------------
# L_PC — énergie libre résiduelle (régularisation PC)
# ---------------------------------------------------------------------------

def loss_pc(hierarchy_state: 'PCHierarchyState') -> jnp.ndarray:
    """
    L_PC = Σ_l Σ_i ||ε_i^l||² / σ_l²

    Régularise la convergence PC via l'énergie libre résiduelle après inférence.
    On utilise precision comme 1/σ² (précision diagonale).

    hierarchy_state.errors     : (L, B, d_z)
    hierarchy_state.precisions : (L, B, d_z)
    return : scalaire float32
    """
    # Énergie pondérée : Σ_l,b,d  precision * error²
    weighted_sq = hierarchy_state.precisions * hierarchy_state.errors ** 2
    return jnp.mean(weighted_sq)


# ---------------------------------------------------------------------------
# L_var — anti-collapse
# ---------------------------------------------------------------------------

def loss_variance(z: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """
    L_var = max(0, γ - Var(z))
    Anti-collapse : pénalise si la variance des représentations < γ.

    z     : (B, d_z) ou (N, d_z) après reshape
    gamma : float — variance cible (config.gamma_var)
    return : scalaire float32
    """
    var = jnp.var(z, axis=0).mean()          # variance moyenne sur d_z
    return jax.nn.relu(gamma - var)


# ---------------------------------------------------------------------------
# Perte totale
# ---------------------------------------------------------------------------

def total_loss(
    weights: dict,
    pc_state_converged: 'PCHierarchyState',
    z_pred: jnp.ndarray,
    z_target_sg: jnp.ndarray,
    config: 'ModelConfig',
) -> tuple:
    """
    L_total = L_JEPA + λ_PC · L_PC + λ_var · L_var

    z_target_sg : stop_gradient déjà appliqué par le trainer (R3).

    Returns:
        (loss_total, dict_losses_détaillées)
    """
    l_jepa = loss_jepa(z_pred, z_target_sg)
    l_pc   = loss_pc(pc_state_converged)
    l_var  = loss_variance(
        z_pred.reshape(-1, z_pred.shape[-1]),   # (B*K, d_z)
        config.gamma_var,
    )

    total = l_jepa + config.lambda_pc * l_pc + config.lambda_var * l_var

    return total, {
        'loss_total': total,
        'loss_jepa':  l_jepa,
        'loss_pc':    l_pc,
        'loss_var':   l_var,
    }
