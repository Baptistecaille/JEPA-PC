"""
=== SPEC [predictor] ===
Entrées  : z_context (B, T_in, d_z) représentations latentes PC stabilisées
           config — fournit n_layers et n_heads (non stockés dans weights)
Sorties  : (B, K, d_z) prédictions latentes pour horizons k=1..K
Architecture:
  Transformer Encoder à trans_n_layers couches, trans_n_heads têtes
  d_model = d_z (cohérence avec l'encoder CNN)
  Positional encoding appris (max_T=20 positions)
  Pooling sur le dernier token → h_T (B, d_z)
  Pour k=1..K :
    k_embed = Embedding(k, pred_k_embed)
    z_hat_k = MLP([h_T, k_embed]) → d_z
Invariants:
  - JIT-compatible (pas de boucle Python sur les couches dans apply)
  - n_layers et n_heads passés via config, PAS stockés dans weights
    (évite l'erreur JAX : les entiers dans weights seraient différentiés)
  - Poids dans un dict plat (R4)
  - Aucun magic number (R7)
  - Parité paramètres avec transformer_baseline.py (~935K params)
=== FIN SPEC ===
"""
import jax
import jax.numpy as jnp
from config import ModelConfig


# ---------------------------------------------------------------------------
# Blocs Transformer (indépendants de transformer_baseline.py — pas d'import croisé)
# ---------------------------------------------------------------------------

def _layer_norm(
    x: jnp.ndarray,
    scale: jnp.ndarray,
    bias: jnp.ndarray,
    eps: float = 1e-5,
) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return scale * (x - mean) / jnp.sqrt(var + eps) + bias


def _scaled_dot_product_attention(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray,
) -> jnp.ndarray:
    """
    Q, K, V : (B, n_heads, T, d_head)
    return  : (B, n_heads, T, d_head)
    """
    d_head = Q.shape[-1]
    scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(
        jnp.array(d_head, dtype=jnp.float32)
    )
    weights_attn = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(weights_attn, V)


def _multi_head_attention(
    weights: dict,
    layer_idx: int,
    x: jnp.ndarray,
    n_heads: int,
) -> jnp.ndarray:
    """
    x      : (B, T, d_model)
    n_heads : Python int — passé en argument, pas lu depuis weights
    return : (B, T, d_model)
    """
    B, T, d_model = x.shape
    prefix = f'layer{layer_idx}_attn'
    d_head = d_model // n_heads

    Q = x @ weights[f'{prefix}_Wq'] + weights[f'{prefix}_bq']
    K = x @ weights[f'{prefix}_Wk'] + weights[f'{prefix}_bk']
    V = x @ weights[f'{prefix}_Wv'] + weights[f'{prefix}_bv']

    def split_heads(z):
        return z.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)

    Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
    attn_out = _scaled_dot_product_attention(Q, K, V)   # (B, n_heads, T, d_head)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, d_model)
    return attn_out @ weights[f'{prefix}_Wo'] + weights[f'{prefix}_bo']


def _transformer_ffn(weights: dict, layer_idx: int, x: jnp.ndarray) -> jnp.ndarray:
    """FFN d'une couche Transformer. x : (B, T, d_model)"""
    prefix = f'layer{layer_idx}_ffn'
    x = x @ weights[f'{prefix}_w1'] + weights[f'{prefix}_b1']
    x = jax.nn.gelu(x)
    x = x @ weights[f'{prefix}_w2'] + weights[f'{prefix}_b2']
    return x


def _transformer_layer(
    weights: dict,
    layer_idx: int,
    x: jnp.ndarray,
    n_heads: int,
) -> jnp.ndarray:
    """Une couche Transformer : MHA + FFN + résidus + LayerNorm (pre-norm). x : (B, T, d_model)"""
    prefix_ln = f'layer{layer_idx}_ln'
    x_norm = _layer_norm(x, weights[f'{prefix_ln}1_scale'], weights[f'{prefix_ln}1_bias'])
    x = x + _multi_head_attention(weights, layer_idx, x_norm, n_heads)
    x_norm = _layer_norm(x, weights[f'{prefix_ln}2_scale'], weights[f'{prefix_ln}2_bias'])
    x = x + _transformer_ffn(weights, layer_idx, x_norm)
    return x


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_predictor(key: jax.random.PRNGKey, config: ModelConfig) -> dict:
    """
    Initialise tous les poids du predictor Transformer.
    Retourne un dict plat (R4).

    Architecture lue depuis config :
      config.trans_n_layers  — nombre de couches Transformer
      config.trans_n_heads   — nombre de têtes d'attention
      config.trans_ffn_dim   — dimension des couches FFN internes
      config.d_z             — d_model (cohérence avec l'encoder)
      config.pred_K          — horizons de prédiction
      config.pred_k_embed    — dimension embedding d'horizon
      config.pred_mlp_dim    — dimension MLP head

    Clés du dict :
      layer{l}_attn_{Wq,Wk,Wv,Wo,bq,bk,bv,bo}  — attention
      layer{l}_ffn_{w1,b1,w2,b2}                — FFN
      layer{l}_ln{1,2}_{scale,bias}             — LayerNorm
      pos_embed                                  — (max_T=20, d_z) positional
      k_embed_table                              — (K, pred_k_embed)
      mlp_w1..3, mlp_b1..3                       — MLP head
    """
    weights  = {}
    n_layers = config.trans_n_layers
    n_heads  = config.trans_n_heads
    ffn_dim  = config.trans_ffn_dim
    d_model  = config.d_z
    K        = config.pred_K
    ke       = config.pred_k_embed
    mlp_d    = config.pred_mlp_dim

    std_attn = jnp.sqrt(2.0 / d_model)

    for l in range(n_layers):
        prefix = f'layer{l}_attn'
        for proj in ['Wq', 'Wk', 'Wv', 'Wo']:
            key, sk = jax.random.split(key)
            weights[f'{prefix}_{proj}'] = (
                jax.random.normal(sk, (d_model, d_model)) * std_attn
            )
        for bname in ['bq', 'bk', 'bv', 'bo']:
            weights[f'{prefix}_{bname}'] = jnp.zeros((d_model,))

        prefix_ffn = f'layer{l}_ffn'
        for (in_d, out_d, wk, bk) in [
            (d_model, ffn_dim, 'w1', 'b1'),
            (ffn_dim, d_model, 'w2', 'b2'),
        ]:
            key, sk = jax.random.split(key)
            std = jnp.sqrt(2.0 / in_d)
            weights[f'{prefix_ffn}_{wk}'] = jax.random.normal(sk, (in_d, out_d)) * std
            weights[f'{prefix_ffn}_{bk}'] = jnp.zeros((out_d,))

        prefix_ln = f'layer{l}_ln'
        for i in ['1', '2']:
            weights[f'{prefix_ln}{i}_scale'] = jnp.ones((d_model,))
            weights[f'{prefix_ln}{i}_bias']  = jnp.zeros((d_model,))

    # Positional encoding appris
    max_T = 20
    key, sk = jax.random.split(key)
    weights['pos_embed'] = jax.random.normal(sk, (max_T, d_model)) * 0.02

    # Horizon embedding table
    key, sk = jax.random.split(key)
    weights['k_embed_table'] = jax.random.normal(sk, (K, ke)) * 0.02

    # MLP head
    mlp_in = d_model + ke
    for (in_d, out_d, wk, bk) in [
        (mlp_in, mlp_d,   'mlp_w1', 'mlp_b1'),
        (mlp_d,  mlp_d,   'mlp_w2', 'mlp_b2'),
        (mlp_d,  d_model, 'mlp_w3', 'mlp_b3'),
    ]:
        key, sk = jax.random.split(key)
        std = jnp.sqrt(2.0 / in_d)
        weights[wk] = jax.random.normal(sk, (in_d, out_d)) * std
        weights[bk] = jnp.zeros((out_d,))

    return weights


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def apply_predictor(
    weights: dict,
    z_context: jnp.ndarray,
    config: ModelConfig,
) -> jnp.ndarray:
    """
    z_context : (B, T_in, d_z)  — représentations PC stabilisées
    config    : ModelConfig — fournit trans_n_layers et trans_n_heads
    return    : (B, K, d_z)     — prédictions horizons 1..K

    Transformer sur la séquence entière, pooling sur le dernier token,
    puis MLP par horizon avec embedding.
    JIT-compatible : pas de boucle Python sur K.
    """
    B, T, d_model = z_context.shape
    n_layers = config.trans_n_layers
    n_heads  = config.trans_n_heads

    # Positional encoding
    pos = weights['pos_embed'][:T]             # (T, d_model)
    x = z_context + pos[jnp.newaxis]           # (B, T, d_model)

    # Couches Transformer
    for l in range(n_layers):
        x = _transformer_layer(weights, l, x, n_heads)

    # Pooling : dernier token comme représentation contexte
    h_T = x[:, -1, :]                          # (B, d_model)

    # MLP head multi-horizon (vmap sur K, JIT-compatible)
    K = weights['k_embed_table'].shape[0]

    def predict_one_horizon(k_idx: jnp.ndarray) -> jnp.ndarray:
        """k_idx : scalaire int → (B, d_z)"""
        k_emb = weights['k_embed_table'][k_idx]                      # (pred_k_embed,)
        k_emb_batch = jnp.broadcast_to(k_emb, (B, k_emb.shape[0]))  # (B, pred_k_embed)
        inp = jnp.concatenate([h_T, k_emb_batch], axis=-1)           # (B, d_model+ke)
        h = inp @ weights['mlp_w1'] + weights['mlp_b1']
        h = jax.nn.gelu(h)
        h = h   @ weights['mlp_w2'] + weights['mlp_b2']
        h = jax.nn.gelu(h)
        h = h   @ weights['mlp_w3'] + weights['mlp_b3']
        return h                                                       # (B, d_z)

    k_indices = jnp.arange(K)
    z_preds = jax.vmap(predict_one_horizon)(k_indices)   # (K, B, d_z)
    return z_preds.transpose(1, 0, 2)                    # (B, K, d_z)


def count_predictor_params(weights: dict) -> int:
    """Compte le nombre total de paramètres."""
    return sum(v.size for v in jax.tree_util.tree_leaves(weights))


# ---------------------------------------------------------------------------
# Sanity checks (R6)
# ---------------------------------------------------------------------------

def run_sanity_checks_predictor(config: ModelConfig) -> None:
    """
    R6 — Sanity checks du module predictor (Transformer).

    [P1] Forme de sortie : (B, K, d_z)
    [P2] Pas de NaN
    [P3] JIT-compilable
    [P4] Gradient calculable par rapport aux poids
    [P5] Parité params avec transformer_baseline (tolérance 25%)
    """
    print("[Sanity] models.predictor (Transformer) — début")

    B, T = 2, 10
    key = jax.random.PRNGKey(config.seed)
    key, sk1, sk2 = jax.random.split(key, 3)

    weights   = init_predictor(sk1, config)
    z_context = jax.random.normal(sk2, (B, T, config.d_z))

    # [P1] Forme
    z_pred = apply_predictor(weights, z_context, config)
    assert z_pred.shape == (B, config.pred_K, config.d_z), \
        f"[P1] shape attendue ({B},{config.pred_K},{config.d_z}), obtenue {z_pred.shape}"
    print(f"  [P1] ✓ shape: {z_pred.shape}")

    # [P2] Pas de NaN
    assert not jnp.any(jnp.isnan(z_pred)), "[P2] NaN dans la sortie du predictor"
    print("  [P2] ✓ pas de NaN")

    # [P3] JIT
    pred_jit = jax.jit(apply_predictor, static_argnames=('config',))
    z_jit = pred_jit(weights, z_context, config)
    assert jnp.allclose(z_pred, z_jit, atol=1e-5), "[P3] JIT donne des résultats différents"
    print("  [P3] ✓ JIT-compilable")

    # [P4] Gradient
    def loss_fn(w):
        return jnp.mean(apply_predictor(w, z_context, config) ** 2)

    grads = jax.grad(loss_fn)(weights)
    for k, g in grads.items():
        assert not jnp.any(jnp.isnan(g)), f"[P4] NaN dans le gradient de {k}"
    print("  [P4] ✓ gradients calculables")

    # [P5] Parité params
    n_params = count_predictor_params(weights)
    print(f"  [P5] Paramètres predictor : {n_params:,}")

    print("[Sanity] models.predictor (Transformer) — OK\n")
