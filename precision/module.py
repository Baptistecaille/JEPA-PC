"""
=== SPEC [precision.module] ===
Entrées  : errors (n_units,) ou (batch, n_units), PCWeights inhibiteurs
Sorties  : erreurs normalisées divisivment + métriques
Invariants:
  - Compatible jax.jit, jax.grad, jax.vmap
  - Dénominateur toujours > 0 (exp(log_eps) > 0)
  - W_inhib ≥ 0, diag = 0, rows ≈ 1 (inhibition pure, pas d'auto-inhibition)
  - Règle biologique : interneurones PV L2/3, inhibition latérale divisive
=== FIN SPEC ===
"""
import jax
import jax.numpy as jnp
from typing import NamedTuple
from jax import Array


# ---------------------------------------------------------------------------
# Structures de données (R4 — NamedTuple, pytree natif JAX)
# ---------------------------------------------------------------------------

class PrecisionParams(NamedTuple):
    """
    Paramètres apprenables du module de précision divisive.
    Compatible JAX pytree (NamedTuple).
    """
    W_inhib: Array   # shape (n_units, n_units) — connectivité PV latérale
    log_eps: Array   # shape ()                 — log(ε), appris, init log(1e-4)


class PrecisionState(NamedTuple):
    """État interne du module (pour usage futur — monitoring)."""
    running_mean: Array   # shape (n_units,) — moyenne glissante de |e|
    step: Array           # shape ()          — compteur de pas


# ---------------------------------------------------------------------------
# Normalisation divisive core
# ---------------------------------------------------------------------------

def divisive_normalize(
    errors: Array,        # shape (n_units,)
    W_inhib: Array,       # shape (n_units, n_units)
    log_eps: Array,       # shape () — log(ε) appris
) -> Array:
    """
    Normalisation divisive des unités d'erreur PC via inhibition PV latérale.

    Pour chaque unité i :
        e_norm[i] = e[i] / (exp(log_eps) + Σ_j W_inhib[i,j] * |e[j]|)

    Équivalence biologique :
        - Numérateur    : signal d'erreur brut (activité excitatrice)
        - Dénominateur  : inhibition divisive par pool PV pondérée par W_inhib
        - exp(log_eps)  : conductance de fuite basale (évite division par zéro)

    Args:
        errors  : signal d'erreur PC, float32, shape (n_units,)
        W_inhib : matrice de connectivité inhibitrice PV, valeurs ≥ 0, diag = 0
        log_eps : log du terme de stabilisation, shape ()

    Returns:
        Array de même shape que errors, normalisé.

    Notes:
        - Compatible jax.vmap sur le batch (appliquer sur axis 0)
        - Compatible jax.jit et jax.grad
        - Gradient bien défini partout (dénominateur > 0 garanti par exp(log_eps))
    """
    eps = jnp.exp(log_eps)
    abs_errors = jnp.abs(errors)

    # Inhibition latérale : pour chaque unité i, somme pondérée des |e[j]|
    lateral_inhibition = W_inhib @ abs_errors     # shape (n_units,)
    denominator = eps + lateral_inhibition        # toujours > 0

    return errors / denominator


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_precision_params(
    n_units: int,
    key: Array,
    inhibition_mode: str = "uniform",   # "uniform" | "random" | "identity"
    eps_init: float = 1e-4,
) -> PrecisionParams:
    """
    Initialise les paramètres du module de précision.

    Args:
        n_units         : nombre d'unités d'erreur PC (= d_z)
        key             : clé JAX random
        inhibition_mode :
            "uniform"  → compétition symétrique uniforme (recommandé pour PoC)
            "random"   → inhibition aléatoire (pour ablation)
            "identity" → aucune interaction latérale (baseline)
        eps_init        : valeur initiale de ε (stabilisation du dénominateur)

    Returns:
        PrecisionParams avec W_inhib normalisé et diagonale nulle.
    """
    if inhibition_mode == "uniform":
        # Toutes les connexions inter-unités égales : compétition symétrique
        W = jnp.ones((n_units, n_units)) / (n_units - 1)

    elif inhibition_mode == "random":
        k1, _ = jax.random.split(key)
        W = jax.nn.softplus(jax.random.normal(k1, (n_units, n_units)))
        row_sums = W.sum(axis=1, keepdims=True)
        W = W / (row_sums + 1e-8)

    elif inhibition_mode == "identity":
        # Baseline : aucune interaction latérale (dénominateur = eps seul)
        W = jnp.zeros((n_units, n_units))

    else:
        raise ValueError(f"inhibition_mode inconnu : {inhibition_mode!r}")

    # Diagonale nulle (pas d'auto-inhibition — contrainte PV biologique)
    W = W * (1 - jnp.eye(n_units))

    return PrecisionParams(
        W_inhib = W,
        log_eps = jnp.log(jnp.array(eps_init, dtype=jnp.float32)),
    )


# ---------------------------------------------------------------------------
# Application batch via vmap
# ---------------------------------------------------------------------------

def apply_precision_batch(
    errors: Array,           # shape (batch, n_units)
    params: PrecisionParams,
) -> Array:
    """
    Applique divisive_normalize sur un batch complet via jax.vmap.

    Args:
        errors : shape (batch, n_units)
        params : PrecisionParams

    Returns:
        shape (batch, n_units), normalisé divisivment.
    """
    normalize_fn = lambda e: divisive_normalize(e, params.W_inhib, params.log_eps)
    return jax.vmap(normalize_fn)(errors)


# ---------------------------------------------------------------------------
# Contraintes biologiques post-update (R2 — appliqué après optax)
# ---------------------------------------------------------------------------

def enforce_pv_constraints(params: PrecisionParams) -> PrecisionParams:
    """
    Applique les contraintes biologiques PV après chaque update gradient :
        1. Positivité  : W_inhib ≥ 0 (inhibition pure, pas d'excitation latérale)
        2. Diagonale 0 : pas d'auto-inhibition
        3. Normalisation par ligne : conservation du flux inhibiteur

    CRITIQUE : appeler après optax.apply_updates, pas avant.

    Compatible jax.jit (toutes les opérations sont JAX pures).
    """
    W = params.W_inhib

    # 1. Positivité (ReLU — coupe les poids négatifs)
    W = jax.nn.relu(W)

    # 2. Diagonale nulle (pas d'auto-inhibition)
    W = W * (1 - jnp.eye(W.shape[0]))

    # 3. Normalisation par ligne (stabilise l'apprentissage)
    row_sums = W.sum(axis=1, keepdims=True)
    W = W / (row_sums + 1e-8)

    return PrecisionParams(W_inhib=W, log_eps=params.log_eps)


# ---------------------------------------------------------------------------
# Variantes expérimentales (optionnel — ablation)
# ---------------------------------------------------------------------------

def anti_hebb_update(
    W_inhib: Array,    # shape (n_units, n_units)
    errors: Array,     # shape (n_units,)
    lr: float = 1e-3,
) -> Array:
    """
    Mise à jour anti-Hebbienne de W_inhib.
    ΔW[i,j] ∝ -|e[i]| * |e[j]|  (co-activation → inhibition renforcée)

    Alternative biologique à l'apprentissage par gradient de W_inhib.
    Utile pour ablation : comparer gradient vs anti-Hebb sur W_inhib.
    """
    abs_e = jnp.abs(errors)
    dW = -jnp.outer(abs_e, abs_e)          # co-activation → inhibition
    dW = dW * (1 - jnp.eye(len(errors)))   # pas d'auto-inhibition
    return W_inhib + lr * dW


def grouped_divisive_normalize(
    errors: Array,      # shape (n_units,)
    W_inhib: Array,     # shape (n_units, n_units) — ignoré ici (inhibition locale)
    log_eps: Array,     # shape ()
    group_size: int = 4,
) -> Array:
    """
    Normalisation divisive de groupe (inhibition intra-hypercolonne).
    Compétition uniquement entre unités du même groupe.

    group_size : nombre d'unités par groupe compétitif.
    """
    n = errors.shape[0]
    n_groups = n // group_size
    eps = jnp.exp(log_eps)

    def normalize_group(g_errors: Array) -> Array:
        abs_e = jnp.abs(g_errors)
        # Chaque unité voit la somme des AUTRES dans le groupe
        denom = eps + abs_e.sum() - abs_e
        return g_errors / denom

    groups = errors.reshape(n_groups, group_size)
    normalized = jax.vmap(normalize_group)(groups)
    return normalized.reshape(n)
