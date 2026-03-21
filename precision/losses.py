"""
=== SPEC [precision.losses] ===
Entrées  : errors (batch, n_units), PrecisionParams
Sorties  : scalaire de perte float32
Invariants:
  - z_target DOIT être stop_gradient avant d'arriver ici (R3 — responsabilité trainer)
  - Toutes les pertes sont différentiables (gradient via log_eps et W_inhib)
  - pc_loss_standard(errors) == loss_jepa quand alpha=0 dans pc_loss_hybrid
=== FIN SPEC ===
"""
import jax.numpy as jnp
from typing import Optional
from jax import Array

from precision.module import PrecisionParams, apply_precision_batch


# ---------------------------------------------------------------------------
# Perte PC standard — baseline additif
# ---------------------------------------------------------------------------

def pc_loss_standard(
    errors: Array,                          # shape (batch, n_units)
    precision: Optional[Array] = None,      # shape (n_units,) ou None
) -> Array:
    """
    Perte PC standard avec weighting additif.
    L = mean(π_i * e_i²)

    Baseline à battre pour l'argument de sample efficiency.
    Si precision=None, équivalent à MSE (π_i = 1 partout).

    Args:
        errors    : erreurs de prédiction brutes, shape (batch, n_units)
        precision : poids diagonaux optionnels, shape (n_units,)

    Returns:
        Scalaire float32, perte moyenne sur le batch.
    """
    squared = errors ** 2
    if precision is not None:
        squared = precision * squared
    return jnp.mean(squared)


# ---------------------------------------------------------------------------
# Perte PC divisive — normalisation PV
# ---------------------------------------------------------------------------

def pc_loss_divisive(
    errors: Array,           # shape (batch, n_units)
    params: PrecisionParams,
) -> Array:
    """
    Perte PC avec normalisation divisive (inhibition PV latérale).
    Remplace pc_loss_standard dans train_step.

    Formule :
        loss = mean( (e / (ε + W_inhib @ |e|))² )

    Propriétés :
        - Sparse : les grandes erreurs suppriment les petites (compétition)
        - Scale-invariant : invariant au scaling uniforme des erreurs
        - Differentiable partout (gradient bien défini via log_eps et W_inhib)

    Args:
        errors : erreurs PC brutes, float32, shape (batch, n_units)
        params : PrecisionParams avec W_inhib et log_eps apprenables

    Returns:
        Scalaire float32, perte moyenne sur le batch.
    """
    norm_errors = apply_precision_batch(errors, params)   # (batch, n_units)
    return jnp.mean(norm_errors ** 2)


# ---------------------------------------------------------------------------
# Perte hybride — curriculum standard → divisive
# ---------------------------------------------------------------------------

def pc_loss_hybrid(
    errors: Array,           # shape (batch, n_units)
    params: PrecisionParams,
    alpha: float = 0.5,      # 0.0 = standard pur, 1.0 = divisif pur
) -> Array:
    """
    Interpolation linéaire entre perte standard et divisive.
    Utile pour le curriculum : commencer additif, basculer vers divisif.

    Recommandation : alpha=0.0 pendant les 5 premières epochs,
    puis incrémenter de 0.1 par epoch jusqu'à alpha=1.0.

    Args:
        errors : erreurs de prédiction, shape (batch, n_units)
        params : PrecisionParams
        alpha  : coefficient de mélange ∈ [0, 1]

    Returns:
        Scalaire float32.
    """
    loss_std = pc_loss_standard(errors)
    loss_div = pc_loss_divisive(errors, params)
    return (1.0 - alpha) * loss_std + alpha * loss_div
