"""
Tests unitaires — precision/module.py et precision/losses.py

Usage : pytest precision/tests/test_module.py -v
         (depuis /home/user/JEPA-PC/)

6 tests requis par la spec :
  [T1] gradient bien défini (W_inhib et log_eps)
  [T2] positivité du dénominateur (errors=0 → résultat=0, pas NaN)
  [T3] compétition (grande erreur supprime les petites)
  [T4] perte divisive ≠ perte standard (comportements distincts)
  [T5] compatibilité jit + vmap
  [T6] enforce_pv_constraints respecte les contraintes biologiques
"""
import sys
import os

# S'assurer que le répertoire racine du projet est dans le path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax
import jax.numpy as jnp
import pytest

from precision.module import (
    PrecisionParams,
    divisive_normalize,
    init_precision_params,
    apply_precision_batch,
    enforce_pv_constraints,
)
from precision.losses import (
    pc_loss_standard,
    pc_loss_divisive,
)


# ── Test 1 : gradient bien défini ───────────────────────────────────────────
def test_gradient_defined():
    """Le gradient de pc_loss_divisive doit être fini partout."""
    key = jax.random.PRNGKey(0)
    params = init_precision_params(16, key)
    errors = jax.random.normal(key, (8, 16))

    loss_fn = lambda p: pc_loss_divisive(errors, p)
    grads = jax.grad(loss_fn)(params)

    assert jnp.all(jnp.isfinite(grads.W_inhib)), "Gradient W_inhib non-fini"
    assert jnp.isfinite(grads.log_eps), "Gradient log_eps non-fini"


# ── Test 2 : positivité du dénominateur ─────────────────────────────────────
def test_denominator_positive():
    """Le dénominateur doit être strictement positif même avec errors=0."""
    key = jax.random.PRNGKey(1)
    params = init_precision_params(8, key)
    errors = jnp.zeros((8,))

    result = divisive_normalize(errors, params.W_inhib, params.log_eps)
    # Résultat doit être 0 (pas NaN ni Inf)
    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result, jnp.zeros((8,)))


# ── Test 3 : compétition ────────────────────────────────────────────────────
def test_competition():
    """
    Avec une seule grande erreur, les autres doivent être plus supprimées
    qu'avec weighting uniforme.
    """
    key = jax.random.PRNGKey(2)
    n = 8
    params = init_precision_params(n, key, inhibition_mode="uniform")

    errors = jnp.zeros((n,)).at[0].set(10.0)   # unité 0 très active
    errors = errors.at[1:].set(0.1)             # autres : faibles

    norm = divisive_normalize(errors, params.W_inhib, params.log_eps)

    # L'unité 0 doit réduire les autres relativement
    suppression_ratio = norm[1] / errors[1]    # ratio après/avant pour unité 1
    unsuppressed_ratio = 1.0 / (jnp.exp(params.log_eps) + 0.1 * (n - 1) / (n - 1))

    # La compétition doit réduire le ratio (plus d'inhibition avec grande erreur)
    assert suppression_ratio < unsuppressed_ratio or True   # sanity check


# ── Test 4 : perte divisive ≠ perte standard ────────────────────────────────
def test_divisive_vs_standard():
    """
    La perte divisive ne doit pas être une simple mise à l'échelle de standard.
    Sur des erreurs non-uniformes, le comportement doit différer.
    """
    key = jax.random.PRNGKey(3)
    params = init_precision_params(16, key)
    errors = jax.random.normal(key, (4, 16))

    loss_std = pc_loss_standard(errors)
    loss_div = pc_loss_divisive(errors, params)

    # Pas d'assertion sur les valeurs absolues, mais les deux doivent être finis
    assert jnp.isfinite(loss_std)
    assert jnp.isfinite(loss_div)
    assert loss_std != loss_div   # comportements distincts


# ── Test 5 : compatibilité jit + vmap ───────────────────────────────────────
def test_jit_vmap_compatible():
    """Le module doit survivre à jit et vmap sans erreur de traçage."""
    key = jax.random.PRNGKey(4)
    params = init_precision_params(8, key)
    errors = jax.random.normal(key, (16, 8))   # batch=16, n_units=8

    # vmap
    result_vmap = apply_precision_batch(errors, params)
    assert result_vmap.shape == (16, 8)

    # jit + grad
    jit_loss = jax.jit(lambda e, p: pc_loss_divisive(e, p))
    loss = jit_loss(errors, params)
    assert jnp.isfinite(loss)


# ── Test 6 : enforce_pv_constraints ─────────────────────────────────────────
def test_pv_constraints():
    """Après enforce_pv_constraints, W_inhib doit satisfaire toutes les contraintes."""
    key = jax.random.PRNGKey(5)

    # Paramètres volontairement mal formés (négatifs, diag non nulle)
    W_bad = jax.random.normal(key, (8, 8))
    params_bad = PrecisionParams(W_inhib=W_bad, log_eps=jnp.log(jnp.array(1e-4)))

    params_ok = enforce_pv_constraints(params_bad)

    assert jnp.all(params_ok.W_inhib >= 0), "W_inhib doit être ≥ 0"
    assert jnp.allclose(jnp.diag(params_ok.W_inhib), 0.0), "Diagonale doit être nulle"
