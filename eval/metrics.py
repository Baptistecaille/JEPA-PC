"""
=== SPEC [metrics] ===
Entrées  : z_pred (B,K,d_z), z_target (B,K,d_z), pc_state, T_conv
Sorties  : dict de métriques scalaires
Métriques:
  - NMSE : erreur normalisée par la norme de la cible
  - collapse_score : score d'effondrement de l'espace latent ∈ [0,1]
  - T_conv_mean / T_conv_max : convergence de la boucle PC
Invariants:
  - Toutes les métriques sont des scalaires JAX
  - Aucun magic number (R7)
=== FIN SPEC ===
"""
import jax
import jax.numpy as jnp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.pc_nodes import PCHierarchyState


# ---------------------------------------------------------------------------
# NMSE — Normalized Mean Squared Error
# ---------------------------------------------------------------------------

def nmse(z_pred: jnp.ndarray, z_target: jnp.ndarray) -> jnp.ndarray:
    """
    NMSE = E[||ẑ - z||² / ||z||²]
    Normalisée par la norme de la cible pour comparaison multi-horizon.

    z_pred   : (B, K, d_z)  ou  (B, d_z)
    z_target : même shape
    return   : scalaire float32
    """
    num = jnp.sum((z_pred - z_target) ** 2, axis=-1)        # (B, K) ou (B,)
    den = jnp.sum(z_target ** 2, axis=-1) + 1e-8            # stabilité numérique
    return jnp.mean(num / den)


# ---------------------------------------------------------------------------
# Collapse score — rang effectif de l'espace latent
# ---------------------------------------------------------------------------

def collapse_score(z: jnp.ndarray) -> jnp.ndarray:
    """
    Score d'effondrement ∈ [0, 1].
    0 = espace latent riche, 1 = effondrement total.

    Basé sur l'entropie spectrale normalisée de la matrice de covariance.
    Un espace latent pleinement utilisé a une distribution uniforme des valeurs propres
    → entropie maximale → collapse_score proche de 0.

    z : (N, d_z) — N exemples
    return : scalaire float32 ∈ [0, 1]
    """
    z_centered = z - z.mean(axis=0)
    cov = z_centered.T @ z_centered / z.shape[0]             # (d_z, d_z)
    eigenvalues = jnp.linalg.eigvalsh(cov)                   # (d_z,)
    eigenvalues = jnp.clip(eigenvalues, 1e-8, None)

    # Distribution de probabilité sur les valeurs propres
    p = eigenvalues / eigenvalues.sum()
    entropy = -jnp.sum(p * jnp.log(p))
    max_entropy = jnp.log(jnp.array(z.shape[-1], dtype=jnp.float32))

    return 1.0 - entropy / max_entropy


# ---------------------------------------------------------------------------
# Agrégation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    z_pred: jnp.ndarray,
    z_target: jnp.ndarray,
    pc_state: 'PCHierarchyState',
    T_conv: jnp.ndarray,
) -> dict:
    """
    Agrège toutes les métriques en un dict pour le logging.

    z_pred    : (B, K, d_z)
    z_target  : (B, K, d_z)
    pc_state  : PCHierarchyState
    T_conv    : scalaire ou (B,)
    """
    z_flat = z_pred.reshape(-1, z_pred.shape[-1])   # (B*K, d_z)

    return {
        'nmse':           nmse(z_pred, z_target),
        'collapse_score': collapse_score(z_flat),
        'T_conv_mean':    jnp.mean(jnp.array(T_conv, dtype=jnp.float32)),
        'T_conv_max':     jnp.max(jnp.array(T_conv, dtype=jnp.float32)),
    }


def compute_per_horizon_nmse(
    z_pred: jnp.ndarray,
    z_target: jnp.ndarray,
) -> jnp.ndarray:
    """
    NMSE par horizon k=1..K (pour les courbes d'analyse).

    z_pred   : (B, K, d_z)
    z_target : (B, K, d_z)
    return   : (K,) NMSE par horizon
    """
    num = jnp.sum((z_pred - z_target) ** 2, axis=-1)   # (B, K)
    den = jnp.sum(z_target ** 2, axis=-1) + 1e-8        # (B, K)
    return jnp.mean(num / den, axis=0)                   # (K,)
