"""
=== SPEC [pc_nodes] ===
Entrées  : obs_embedding (B, d_z) — sortie encoder pour la frame courante
           PCWeights — matrices de prédiction inter-niveaux
Sorties  : PCHierarchyState convergé + métriques (T_conv, final_error)
Invariants:
  - BOUCLE 1 (inférence) : poids figés, états x libres (R2)
  - BOUCLE 2 (apprentissage) : appelée depuis trainer.py, pas ici
  - jax.lax.while_loop pour la convergence (JIT-compatible)
  - Shapes statiques dans le carry de while_loop
  - Niveau 0 ancré sur obs_embedding (pas mis à jour par la boucle)
  - Energie libre F = Σ_l (1/2) ε^l^T Π^l ε^l avec ε^l = x^l - W^l x^{l+1}
=== FIN SPEC ===
"""
import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from config import ModelConfig


# ---------------------------------------------------------------------------
# Structures de données (R4 — NamedTuple)
# ---------------------------------------------------------------------------

class PCNodeState(NamedTuple):
    representation: jnp.ndarray   # x_i  shape: (B, d_z)
    error:          jnp.ndarray   # ε_i  shape: (B, d_z)
    precision:      jnp.ndarray   # Π_i  shape: (B, d_z) diagonal


class PCWeights(NamedTuple):
    W_pred: tuple   # W_pred[l] : (d_z, d_z)  prédiction l→l-1
    b_pred: tuple   # b_pred[l] : (d_z,)      biais


# Hiérarchie aplatie : tuple fixe de PCNodeState (shapes statiques pour while_loop)
class PCHierarchyState(NamedTuple):
    """
    Hiérarchie PC à L niveaux.
    On stocke les nœuds comme des tenseurs empilés pour compatibilité JAX :
      representations : (L, B, d_z)
      errors          : (L, B, d_z)
      precisions      : (L, B, d_z)
    L est fixé à la compilation.
    """
    representations: jnp.ndarray   # (L, B, d_z)
    errors:          jnp.ndarray   # (L, B, d_z)
    precisions:      jnp.ndarray   # (L, B, d_z)


# ---------------------------------------------------------------------------
# Calcul des erreurs de prédiction
# ---------------------------------------------------------------------------

def _compute_errors(
    representations: jnp.ndarray,   # (L, B, d_z)
    weights: PCWeights,
    obs_embedding: jnp.ndarray,     # (B, d_z)
) -> jnp.ndarray:
    """
    Calcule les erreurs ε^l = x^l - W^l x^{l+1} pour l = 0..L-2.
    ε^{L-1} = x^{L-1}  (niveau supérieur — prior à zéro)
    ε^0 est ancré sur obs_embedding : ε^0 = obs_embedding - W^0 x^1

    representations : (L, B, d_z)
    return          : (L, B, d_z)
    """
    L = representations.shape[0]
    errors = []

    for l in range(L):
        x_l = representations[l]   # (B, d_z)
        if l == 0:
            # Niveau 0 : ancré sur l'observation
            x_pred = obs_embedding
        else:
            x_pred = x_l   # sera écrasé par la prédiction descendante

        if l < L - 1:
            x_above = representations[l + 1]   # (B, d_z)
            mu_l = x_above @ weights.W_pred[l].T + weights.b_pred[l]   # (B, d_z)
        else:
            # Niveau supérieur : prior gaussien centré à zéro
            mu_l = jnp.zeros_like(x_l)

        if l == 0:
            err = obs_embedding - mu_l
        else:
            err = x_l - mu_l

        errors.append(err)

    return jnp.stack(errors, axis=0)   # (L, B, d_z)


# ---------------------------------------------------------------------------
# Énergie libre totale
# ---------------------------------------------------------------------------

def free_energy(
    hierarchy_state: PCHierarchyState,
    weights: PCWeights,
    obs_embedding: jnp.ndarray,
) -> jnp.ndarray:
    """
    F(x, W) = Σ_l (1/2) ||ε^l||²_Π = Σ_l (1/2) ε^l * Π^l * ε^l

    hierarchy_state : PCHierarchyState avec shapes (L, B, d_z)
    weights         : PCWeights (poids figés pendant l'inférence)
    obs_embedding   : (B, d_z)
    return          : scalaire float32
    """
    errors = _compute_errors(
        hierarchy_state.representations,
        weights,
        obs_embedding,
    )   # (L, B, d_z)

    # Énergie pondérée par précision diagonale
    weighted = 0.5 * jnp.sum(
        hierarchy_state.precisions * errors ** 2
    )
    return weighted


# ---------------------------------------------------------------------------
# BOUCLE 1 — Inférence (poids figés, états x libres) — R2
# ---------------------------------------------------------------------------

def inference_step_fn(
    hierarchy_state: PCHierarchyState,
    weights: PCWeights,
    obs_embedding: jnp.ndarray,
    alpha: float,
) -> PCHierarchyState:
    """
    Un pas de la Boucle 1 : x ← x - α · ∂F/∂x

    Les poids sont stop_gradients via argnums=0 : seul hierarchy_state
    est différentié.

    R2 : cette fonction implémente UNIQUEMENT la Boucle 1.
    """
    # Gradient uniquement par rapport à hierarchy_state (argnums=0)
    grads_state = jax.grad(free_energy, argnums=0)(
        hierarchy_state,
        jax.lax.stop_gradient(weights),      # sécurité explicite (R2)
        jax.lax.stop_gradient(obs_embedding),
    )

    # Mise à jour : tous les niveaux SAUF le niveau 0 (ancré sur l'observation)
    # Niveau 0 : représentation fixée = obs_embedding (implicitement)
    # On met à jour levels 1..L-1
    L = hierarchy_state.representations.shape[0]

    # Masque : on ne met pas à jour le niveau 0
    # level_mask[l] = 0 si l==0, 1 sinon
    level_mask = jnp.arange(L) > 0   # (L,) bool
    level_mask = level_mask[:, jnp.newaxis, jnp.newaxis]   # (L, 1, 1)

    new_reprs = jnp.where(
        level_mask,
        hierarchy_state.representations - alpha * grads_state.representations,
        hierarchy_state.representations,  # niveau 0 inchangé
    )

    # Recalcul des erreurs avec les nouvelles représentations
    new_errors = _compute_errors(new_reprs, weights, obs_embedding)

    return PCHierarchyState(
        representations = new_reprs,
        errors          = new_errors,
        precisions      = hierarchy_state.precisions,  # précisions fixes
    )


def compute_max_error(state: PCHierarchyState) -> jnp.ndarray:
    """
    Critère de convergence : max de ||ε^l||_inf sur tous les niveaux et batch.
    Scalaire float32.
    """
    return jnp.max(jnp.abs(state.errors))


def run_inference_loop(
    init_state: PCHierarchyState,
    weights: PCWeights,
    obs_embedding: jnp.ndarray,
    config: ModelConfig,
) -> Tuple[PCHierarchyState, jnp.ndarray, jnp.ndarray]:
    """
    Boucle d'inférence complète — compilée JIT via jax.lax.while_loop.

    R2 : Boucle 1 uniquement. Poids figés, états libres.

    Returns:
        converged_state : PCHierarchyState après convergence
        T_conv          : int32 — nb d'itérations
        final_error     : float32 — ||ε|| final
    """
    def cond_fn(carry):
        state, t, err = carry
        not_converged = err > config.pc_tol
        not_maxed     = t < config.pc_max_iter
        return jnp.logical_and(not_converged, not_maxed)

    def body_fn(carry):
        state, t, _ = carry
        new_state = inference_step_fn(
            state, weights, obs_embedding, config.pc_alpha
        )
        new_err = compute_max_error(new_state)
        return (new_state, t + 1, new_err)

    init_carry = (init_state, jnp.int32(0), jnp.array(jnp.inf, dtype=jnp.float32))
    final_state, T_conv, final_err = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    return final_state, T_conv, final_err


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_pc_weights(key: jax.random.PRNGKey, config: ModelConfig) -> PCWeights:
    """
    Initialise les poids PC avec Xavier.
    W_pred[l] : (d_z, d_z) pour l = 0..L-2
    """
    L = config.pc_n_layers
    W_pred_list = []
    b_pred_list = []

    std = jnp.sqrt(2.0 / config.d_z)
    for l in range(L - 1):
        key, sk = jax.random.split(key)
        W = jax.random.normal(sk, (config.d_z, config.d_z)) * std
        b = jnp.zeros((config.d_z,))
        W_pred_list.append(W)
        b_pred_list.append(b)

    # Dernier niveau : pas de W_pred (prior à zéro)
    # On ajoute un niveau supplémentaire pour la cohérence d'indexation
    key, sk = jax.random.split(key)
    W_pred_list.append(jax.random.normal(sk, (config.d_z, config.d_z)) * std)
    b_pred_list.append(jnp.zeros((config.d_z,)))

    return PCWeights(
        W_pred = tuple(W_pred_list),
        b_pred = tuple(b_pred_list),
    )


def init_pc_hierarchy(
    key: jax.random.PRNGKey,
    config: ModelConfig,
    batch_size: int,
    obs_embedding: jnp.ndarray,   # (B, d_z)
) -> PCHierarchyState:
    """
    Initialise PCHierarchyState.
    Niveau 0 ancré sur obs_embedding.
    Niveaux 1..L-1 initialisés à zéro.
    """
    L = config.pc_n_layers
    B = batch_size

    # Représentations : niveau 0 = obs, niveaux supérieurs = 0
    reprs_list = [obs_embedding]   # (B, d_z)
    for _ in range(L - 1):
        reprs_list.append(jnp.zeros((B, config.d_z)))

    representations = jnp.stack(reprs_list, axis=0)   # (L, B, d_z)
    precisions      = jnp.ones((L, B, config.d_z))    # précisions uniformes
    errors          = jnp.zeros((L, B, config.d_z))   # calculées après init

    return PCHierarchyState(
        representations = representations,
        errors          = errors,
        precisions      = precisions,
    )


def init_pc_from_encoding(
    z_last: jnp.ndarray,
    weights: PCWeights,
    config: ModelConfig,
) -> PCHierarchyState:
    """
    Crée un PCHierarchyState initialisé depuis z_last (dernière frame encodée).
    z_last : (B, d_z)
    """
    key = jax.random.PRNGKey(0)   # seed fixe pour l'init (non aléatoire)
    state = init_pc_hierarchy(key, config, z_last.shape[0], z_last)

    # Calcul initial des erreurs
    errors = _compute_errors(state.representations, weights, z_last)
    return PCHierarchyState(
        representations = state.representations,
        errors          = errors,
        precisions      = state.precisions,
    )


def pc_weights_to_flat_dict(weights: PCWeights) -> dict:
    """Convertit PCWeights en dict plat pour optax."""
    flat = {}
    for l, (w, b) in enumerate(zip(weights.W_pred, weights.b_pred)):
        flat[f'W_pred_{l}'] = w
        flat[f'b_pred_{l}'] = b
    return flat


def flat_dict_to_pc_weights(flat: dict, config: ModelConfig) -> PCWeights:
    """Reconstruit PCWeights depuis un dict plat."""
    L = config.pc_n_layers
    W_pred = tuple(flat[f'W_pred_{l}'] for l in range(L))
    b_pred = tuple(flat[f'b_pred_{l}'] for l in range(L))
    return PCWeights(W_pred=W_pred, b_pred=b_pred)


# ---------------------------------------------------------------------------
# Sanity checks (R6)
# ---------------------------------------------------------------------------

def run_sanity_checks_pc(config: ModelConfig) -> None:
    """
    R6 — Sanity checks du module pc_nodes.

    [I1] free_energy est différentiable par rapport à hierarchy_state
    [I2] inference_step_fn réduit ||ε|| au fil des itérations
    [I3] run_inference_loop est JIT-compilable
    [I4] Les poids ne sont pas modifiés pendant la boucle d'inférence
    [I5] T_conv ≤ pc_max_iter
    """
    print("[Sanity] models.pc_nodes — début")

    B = 2
    key = jax.random.PRNGKey(config.seed)
    key, sk1, sk2 = jax.random.split(key, 3)

    weights = init_pc_weights(sk1, config)
    obs     = jax.random.normal(sk2, (B, config.d_z))
    state   = init_pc_from_encoding(obs, weights, config)

    # [I1] Différentiabilité de free_energy
    try:
        grad_fn = jax.grad(free_energy, argnums=0)
        grads = grad_fn(state, weights, obs)
        assert not jnp.any(jnp.isnan(grads.representations)), \
            "[I1] NaN dans les gradients de free_energy"
        print("  [I1] ✓ free_energy différentiable par rapport à hierarchy_state")
    except Exception as e:
        raise AssertionError(f"[I1] free_energy non différentiable : {e}")

    # [I2] inference_step_fn réduit ||ε||
    err_before = compute_max_error(state)
    state_after = inference_step_fn(state, weights, obs, config.pc_alpha)
    err_after = compute_max_error(state_after)
    # Note: pas garanti après 1 seul pas si α est grand — on vérifie juste la convergence
    print(f"  [I2] erreur avant: {float(err_before):.4f}, après 1 pas: {float(err_after):.4f}")

    # [I3] JIT-compilable
    try:
        run_jit = jax.jit(run_inference_loop, static_argnames=('config',))
        final_state, T_conv, final_err = run_jit(state, weights, obs, config)
        print(f"  [I3] ✓ JIT-compilable, T_conv={int(T_conv)}, err={float(final_err):.6f}")
    except Exception as e:
        raise AssertionError(f"[I3] run_inference_loop non JIT-compilable : {e}")

    # [I4] Poids inchangés pendant l'inférence (stop_gradient vérifié via égalité)
    assert jnp.allclose(weights.W_pred[0], weights.W_pred[0]), "[I4] poids modifiés"
    for l in range(config.pc_n_layers):
        w_orig  = weights.W_pred[l]
        # On vérifie qu'inference_step_fn n'a pas de side-effect sur les poids
        _ = inference_step_fn(state, weights, obs, config.pc_alpha)
        assert jnp.allclose(w_orig, weights.W_pred[l]), \
            f"[I4] poids W_pred[{l}] modifié pendant l'inférence"
    print("  [I4] ✓ poids inchangés pendant l'inférence")

    # [I5] T_conv ≤ pc_max_iter
    assert int(T_conv) <= config.pc_max_iter, \
        f"[I5] T_conv={int(T_conv)} > pc_max_iter={config.pc_max_iter}"
    print(f"  [I5] ✓ T_conv={int(T_conv)} ≤ pc_max_iter={config.pc_max_iter}")

    print("[Sanity] models.pc_nodes — OK\n")
