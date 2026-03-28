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
    final_err: jnp.ndarray | None = None,
) -> dict:
    """
    Agrège toutes les métriques en un dict pour le logging.

    z_pred    : (B, K, d_z)
    z_target  : (B, K, d_z)
    pc_state  : PCHierarchyState
    T_conv    : scalaire ou (B,)
    final_err : scalaire optionnel — erreur PC résiduelle après inférence

    Deux collapse scores distincts :
      collapse_pred : structuration de l'espace des prédictions (predictor)
      collapse_ctx  : richesse de l'espace encodé (encodeur, niveau 0 PC) ← principal
    Mesurer le collapse sur z_pred au lieu de z_ctx était sémantiquement inversé :
    un z_pred bruité (petit n) donne un spectre plat → score ≈ 0 même si l'encodeur
    est en collapse, masquant le vrai problème.
    """
    z_pred_flat = z_pred.reshape(-1, z_pred.shape[-1])   # (B*K, d_z)
    z_ctx_flat  = pc_state.representations[0]             # (B, d_z) — niveau 0 = encodeur

    metrics = {
        'nmse':          nmse(z_pred, z_target),
        'collapse_pred': collapse_score(z_pred_flat),    # structuration des prédictions
        'collapse_ctx':  collapse_score(z_ctx_flat),     # richesse espace encodé ← principal
        'T_conv_mean':   jnp.mean(jnp.array(T_conv, dtype=jnp.float32)),
        'T_conv_max':    jnp.max(jnp.array(T_conv, dtype=jnp.float32)),
    }
    if final_err is not None:
        metrics['pc_error_mean'] = jnp.mean(jnp.array(final_err, dtype=jnp.float32))
    return metrics


# ---------------------------------------------------------------------------
# Violation-of-Expectation (VoE) — signal de surprise PC
# ---------------------------------------------------------------------------

def surprise_score(final_err: jnp.ndarray) -> jnp.ndarray:
    """
    Score de surprise = énergie libre résiduelle après convergence PC.

    C'est la quantité retournée par run_inference_loop comme `final_err`
    (compute_max_error au point fixe x*). Elle mesure à quel point le modèle
    PC n'a pas réussi à expliquer l'observation via sa structure hiérarchique :
      - séquence normale → F faible (le modèle la "comprend")
      - séquence impossible → F élevé (le modèle est "surpris")

    Plus élevé = plus surprenant.

    final_err : scalaire float32 — MSE résiduelle à convergence
    return    : scalaire float32 (identité, pour la sémantique)
    """
    return final_err


def violation_of_expectation_score(
    normal_err: jnp.ndarray,
    impossible_err: jnp.ndarray,
) -> jnp.ndarray:
    """
    Ratio de surprise : séquences impossibles vs normales.

    VoE > 1 : le modèle est correctement plus surpris par les violations.
    VoE ≈ 1 : le modèle ne discrimine pas.
    VoE < 1 : le modèle préfère les séquences impossibles (pathologie).

    Protocole Moving MNIST :
      - séquences normales    : trajectoires continues d'un chiffre
      - séquences impossibles : saut abrupt de position (discontinuité spatiale)
                                ou changement d'identité mid-séquence

    normal_err    : float32 — surprise_score sur batch de séquences normales
    impossible_err: float32 — surprise_score sur batch de séquences impossibles
    return        : scalaire float32 — ratio impossible/normal
    """
    return impossible_err / (normal_err + 1e-8)


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
