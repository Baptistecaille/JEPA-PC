"""
=== SPEC [predictor] ===
Entrées  : z_context (B, T_in, d_z) représentations latentes PC stabilisées
Sorties  : (B, K, d_z) prédictions latentes pour horizons k=1..K
Architecture:
  GRU(d_z → pred_hidden) appliqué sur la séquence via jax.lax.scan
  h_T : (B, pred_hidden) — état caché final
  Pour k=1..K :
    k_embed = Embedding(k, pred_k_embed)
    z_hat_k = MLP([h_T, k_embed]) → d_z
Invariants:
  - JIT-compatible via jax.lax.scan pour le GRU
  - Pas de boucle Python sur K dans la version JIT
  - Poids dans un dict plat (R4)
  - Aucun magic number (R7)
=== FIN SPEC ===
"""
import jax
import jax.numpy as jnp
from typing import NamedTuple
from config import ModelConfig


# ---------------------------------------------------------------------------
# GRU minimal
# ---------------------------------------------------------------------------

def _gru_step(h: jnp.ndarray, x: jnp.ndarray, weights: dict) -> jnp.ndarray:
    """
    Un pas de GRU.
    h : (B, pred_hidden)
    x : (B, d_z)
    return : h_new (B, pred_hidden)
    """
    z = jax.nn.sigmoid(
        x @ weights['Wz_x'] + h @ weights['Wz_h'] + weights['bz']
    )
    r = jax.nn.sigmoid(
        x @ weights['Wr_x'] + h @ weights['Wr_h'] + weights['br']
    )
    g = jnp.tanh(
        x @ weights['Wg_x'] + (r * h) @ weights['Wg_h'] + weights['bg']
    )
    return (1 - z) * h + z * g


def _apply_gru_sequence(weights: dict, z_seq: jnp.ndarray) -> jnp.ndarray:
    """
    Applique le GRU sur toute la séquence via jax.lax.scan.
    z_seq : (B, T, d_z)
    return : h_T (B, pred_hidden)
    """
    B = z_seq.shape[0]
    hidden_size = weights['bz'].shape[0]
    h0 = jnp.zeros((B, hidden_size))

    # scan attend (time, batch, features)
    z_seq_t = z_seq.transpose(1, 0, 2)   # (T, B, d_z)

    def step(h, z_t):
        h_new = _gru_step(h, z_t, weights)
        return h_new, None   # on ne collecte pas tous les états cachés

    h_T, _ = jax.lax.scan(step, h0, z_seq_t)
    return h_T   # (B, pred_hidden)


# ---------------------------------------------------------------------------
# MLP head
# ---------------------------------------------------------------------------

def _apply_mlp_head(weights: dict, x: jnp.ndarray) -> jnp.ndarray:
    """
    MLP à 2 couches cachées : [h_T concat k_embed] → d_z.
    x : (B, pred_hidden + pred_k_embed)
    return : (B, d_z)
    """
    x = x @ weights['mlp_w1'] + weights['mlp_b1']
    x = jax.nn.gelu(x)
    x = x @ weights['mlp_w2'] + weights['mlp_b2']
    x = jax.nn.gelu(x)
    x = x @ weights['mlp_w3'] + weights['mlp_b3']
    return x   # (B, d_z)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_predictor(key: jax.random.PRNGKey, config: ModelConfig) -> dict:
    """
    Initialise tous les poids du prédicateur.
    Retourne un dict plat (R4).

    Clés GRU  : Wz_x, Wz_h, bz, Wr_x, Wr_h, br, Wg_x, Wg_h, bg
    Clés embed: k_embed_table  (K, pred_k_embed)
    Clés MLP  : mlp_w1..3, mlp_b1..3
    """
    weights = {}
    d_z    = config.d_z
    H      = config.pred_hidden
    K      = config.pred_K
    ke     = config.pred_k_embed
    mlp_d  = config.pred_mlp_dim

    std_gru = jnp.sqrt(2.0 / (d_z + H))

    # --- GRU ---
    for gate, wname_x, wname_h, bname in [
        ('z', 'Wz_x', 'Wz_h', 'bz'),
        ('r', 'Wr_x', 'Wr_h', 'br'),
        ('g', 'Wg_x', 'Wg_h', 'bg'),
    ]:
        key, sk1, sk2 = jax.random.split(key, 3)
        weights[wname_x] = jax.random.normal(sk1, (d_z, H)) * std_gru
        weights[wname_h] = jax.random.normal(sk2, (H,  H)) * std_gru
        weights[bname]   = jnp.zeros((H,))

    # --- Horizon embedding table ---
    key, sk = jax.random.split(key)
    weights['k_embed_table'] = jax.random.normal(sk, (K, ke)) * 0.02

    # --- MLP head ---
    mlp_in = H + ke
    for (in_d, out_d, wk, bk) in [
        (mlp_in, mlp_d,  'mlp_w1', 'mlp_b1'),
        (mlp_d,  mlp_d,  'mlp_w2', 'mlp_b2'),
        (mlp_d,  d_z,    'mlp_w3', 'mlp_b3'),
    ]:
        key, sk = jax.random.split(key)
        std = jnp.sqrt(2.0 / in_d)
        weights[wk] = jax.random.normal(sk, (in_d, out_d)) * std
        weights[bk] = jnp.zeros((out_d,))

    return weights


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def apply_predictor(weights: dict, z_context: jnp.ndarray) -> jnp.ndarray:
    """
    z_context : (B, T_in, d_z)  — représentations PC stabilisées
    return    : (B, K, d_z)     — prédictions horizons 1..K

    GRU sur la séquence entière, puis MLP par horizon avec embedding.
    JIT-compatible : on utilise vmap sur les K horizons.
    """
    # 1. Encoder la séquence avec le GRU
    h_T = _apply_gru_sequence(weights, z_context)   # (B, pred_hidden)

    # 2. Prédire pour chaque horizon k=0..K-1 (indice 0 = horizon 1)
    K = weights['k_embed_table'].shape[0]

    # vmap sur les horizons (scalable, JIT-compatible)
    def predict_one_horizon(k_idx: jnp.ndarray) -> jnp.ndarray:
        """k_idx : scalaire int — return : (B, d_z)"""
        k_emb = weights['k_embed_table'][k_idx]   # (pred_k_embed,)
        # Broadcast k_emb sur le batch
        k_emb_batch = jnp.broadcast_to(k_emb, (h_T.shape[0], k_emb.shape[0]))
        inp = jnp.concatenate([h_T, k_emb_batch], axis=-1)   # (B, H+ke)
        return _apply_mlp_head(weights, inp)   # (B, d_z)

    # Vectoriser sur les K horizons
    k_indices = jnp.arange(K)
    z_preds = jax.vmap(predict_one_horizon)(k_indices)   # (K, B, d_z)
    return z_preds.transpose(1, 0, 2)   # (B, K, d_z)


def count_predictor_params(weights: dict) -> int:
    """Compte le nombre total de paramètres."""
    return sum(v.size for v in jax.tree_util.tree_leaves(weights))


# ---------------------------------------------------------------------------
# Sanity checks (R6)
# ---------------------------------------------------------------------------

def run_sanity_checks_predictor(config: ModelConfig) -> None:
    """
    R6 — Sanity checks du module predictor.

    [P1] Forme de sortie : (B, K, d_z)
    [P2] Pas de NaN
    [P3] JIT-compilable
    [P4] Gradient calculable par rapport aux poids
    """
    print("[Sanity] models.predictor — début")

    B, T = 2, 10
    key = jax.random.PRNGKey(config.seed)
    key, sk1, sk2 = jax.random.split(key, 3)

    weights  = init_predictor(sk1, config)
    z_context = jax.random.normal(sk2, (B, T, config.d_z))

    # [P1] Forme
    z_pred = apply_predictor(weights, z_context)
    assert z_pred.shape == (B, config.pred_K, config.d_z), \
        f"[P1] shape attendue ({B},{config.pred_K},{config.d_z}), obtenue {z_pred.shape}"
    print(f"  [P1] ✓ shape: {z_pred.shape}")

    # [P2] Pas de NaN
    assert not jnp.any(jnp.isnan(z_pred)), "[P2] NaN dans la sortie du predictor"
    print("  [P2] ✓ pas de NaN")

    # [P3] JIT
    pred_jit = jax.jit(apply_predictor)
    z_jit = pred_jit(weights, z_context)
    assert jnp.allclose(z_pred, z_jit, atol=1e-5), "[P3] JIT donne des résultats différents"
    print("  [P3] ✓ JIT-compilable")

    # [P4] Gradient
    def loss_fn(w):
        return jnp.mean(apply_predictor(w, z_context) ** 2)

    grads = jax.grad(loss_fn)(weights)
    for k, g in grads.items():
        assert not jnp.any(jnp.isnan(g)), f"[P4] NaN dans le gradient de {k}"
    print("  [P4] ✓ gradients calculables")

    n_params = count_predictor_params(weights)
    print(f"  Paramètres predictor : {n_params:,}")

    print("[Sanity] models.predictor — OK\n")
