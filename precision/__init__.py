"""
Module de précision divisive pour PC-JEPA.

Implémente la normalisation divisive biologiquement fondée sur l'inhibition
latérale des interneurones parvalbumine (PV) du cortex (König & Negrello, 2026).

Exports publics :
    PrecisionParams      — paramètres apprenables (NamedTuple / pytree JAX)
    PrecisionState       — état interne (pour monitoring futur)
    divisive_normalize   — normalisation divisive unité par unité
    init_precision_params — initialisation avec mode d'inhibition configurable
    apply_precision_batch — normalisation sur batch complet (vmap)
    enforce_pv_constraints — contraintes biologiques post-update
    pc_loss_standard     — perte MSE standard (baseline additif)
    pc_loss_divisive     — perte divisive (inhibition PV)
    pc_loss_hybrid       — interpolation standard ↔ divisive (curriculum)
"""
from precision.module import (
    PrecisionParams,
    PrecisionState,
    divisive_normalize,
    init_precision_params,
    apply_precision_batch,
    enforce_pv_constraints,
    anti_hebb_update,
    grouped_divisive_normalize,
)
from precision.losses import (
    pc_loss_standard,
    pc_loss_divisive,
    pc_loss_hybrid,
)

__all__ = [
    # Structures
    "PrecisionParams",
    "PrecisionState",
    # Core
    "divisive_normalize",
    "init_precision_params",
    "apply_precision_batch",
    "enforce_pv_constraints",
    # Variantes
    "anti_hebb_update",
    "grouped_divisive_normalize",
    # Pertes
    "pc_loss_standard",
    "pc_loss_divisive",
    "pc_loss_hybrid",
]
