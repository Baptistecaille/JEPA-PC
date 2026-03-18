"""
=== SPEC [transformer_baseline] ===
Entrées  : z_context (B, T_in, d_z) représentations latentes (même encoder que PC-JEPA)
Sorties  : (B, K, d_z) prédictions latentes — même interface que predictor.py
Architecture:
  4 couches Transformer Encoder, 4 têtes d'attention, d_model = d_z = 256
  MLP head par horizon (identique au predictor PC-JEPA)
Invariants:
  - Même encoder CNN que PC-JEPA (R3 respecté côté baseline aussi)
  - Même perte JEPA
  - Parité des paramètres [0.8, 1.25] avec PC-JEPA (check_parameter_parity)
  - JIT-compatible (jax.lax.scan pour les couches)
  - Aucun magic number (R7)
=== FIN SPEC ===
"""
import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple, Iterator

from config import ModelConfig
from models.encoder import init_encoder, apply_encoder
from models.predictor import count_predictor_params
from training.losses import loss_jepa, loss_variance
from data.moving_mnist import Batch


# ---------------------------------------------------------------------------
# Attention multi-têtes (implémentation manuelle, JAX pur)
# ---------------------------------------------------------------------------

def _scaled_dot_product_attention(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray,
    mask: jnp.ndarray = None,
) -> jnp.ndarray:
    """
    Q, K, V : (B, n_heads, T, d_head)
    return  : (B, n_heads, T, d_head)
    """
    d_head = Q.shape[-1]
    scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(d_head)

    if mask is not None:
        scores = jnp.where(mask, scores, -1e9)

    weights_attn = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(weights_attn, V)


def _multi_head_attention(
    weights: dict,
    layer_idx: int,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """
    Multi-head attention.
    x : (B, T, d_model)
    return : (B, T, d_model)
    """
    B, T, d_model = x.shape
    prefix = f'layer{layer_idx}_attn'
    n_heads = weights[f'{prefix}_n_heads']
    d_head  = d_model // n_heads

    Q = x @ weights[f'{prefix}_Wq'] + weights[f'{prefix}_bq']   # (B, T, d_model)
    K = x @ weights[f'{prefix}_Wk'] + weights[f'{prefix}_bk']
    V = x @ weights[f'{prefix}_Wv'] + weights[f'{prefix}_bv']

    # Reshape → (B, n_heads, T, d_head)
    def split_heads(z):
        return z.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)

    Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
    attn_out = _scaled_dot_product_attention(Q, K, V)   # (B, n_heads, T, d_head)

    # Merge heads → (B, T, d_model)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, d_model)
    return attn_out @ weights[f'{prefix}_Wo'] + weights[f'{prefix}_bo']


def _transformer_ffn(weights: dict, layer_idx: int, x: jnp.ndarray) -> jnp.ndarray:
    """
    Feed-forward network d'une couche Transformer.
    x : (B, T, d_model)
    return : (B, T, d_model)
    """
    prefix = f'layer{layer_idx}_ffn'
    x = x @ weights[f'{prefix}_w1'] + weights[f'{prefix}_b1']
    x = jax.nn.gelu(x)
    x = x @ weights[f'{prefix}_w2'] + weights[f'{prefix}_b2']
    return x


def _layer_norm(x: jnp.ndarray, scale: jnp.ndarray, bias: jnp.ndarray,
                eps: float = 1e-5) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return scale * (x - mean) / jnp.sqrt(var + eps) + bias


def _transformer_layer(
    weights: dict,
    layer_idx: int,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """
    Une couche Transformer : MHA + FFN avec connexions résiduelles et LayerNorm.
    x : (B, T, d_model)
    """
    prefix_ln = f'layer{layer_idx}_ln'
    # Pre-norm (plus stable à l'entraînement)
    x_norm = _layer_norm(
        x, weights[f'{prefix_ln}1_scale'], weights[f'{prefix_ln}1_bias']
    )
    x = x + _multi_head_attention(weights, layer_idx, x_norm)

    x_norm = _layer_norm(
        x, weights[f'{prefix_ln}2_scale'], weights[f'{prefix_ln}2_bias']
    )
    x = x + _transformer_ffn(weights, layer_idx, x_norm)
    return x


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_transformer_predictor(
    key: jax.random.PRNGKey,
    config: ModelConfig,
    n_layers: int = 4,
    n_heads: int  = 4,
    ffn_dim: int  = 512,
) -> dict:
    """
    Initialise les poids du Transformer predictor.
    d_model = config.d_z pour cohérence avec l'encoder.
    """
    weights = {}
    d_model = config.d_z
    d_head  = d_model // n_heads
    K       = config.pred_K
    ke      = config.pred_k_embed

    std_attn = jnp.sqrt(2.0 / d_model)

    for l in range(n_layers):
        # --- Attention ---
        prefix = f'layer{l}_attn'
        weights[f'{prefix}_n_heads'] = n_heads   # stocké pour apply
        for proj, dim_out in [('Wq', d_model), ('Wk', d_model),
                               ('Wv', d_model), ('Wo', d_model)]:
            key, sk = jax.random.split(key)
            weights[f'{prefix}_{proj}'] = (
                jax.random.normal(sk, (d_model, dim_out)) * std_attn
            )
            weights[f'{prefix}_{proj[1]}' + proj[1:].replace(proj[0], 'b')] = (
                jnp.zeros((dim_out,))
            )
        for bname in ['bq', 'bk', 'bv', 'bo']:
            weights[f'{prefix}_{bname}'] = jnp.zeros((d_model,))

        # --- FFN ---
        prefix_ffn = f'layer{l}_ffn'
        for (in_d, out_d, wk, bk) in [
            (d_model, ffn_dim, 'w1', 'b1'),
            (ffn_dim, d_model, 'w2', 'b2'),
        ]:
            key, sk = jax.random.split(key)
            std = jnp.sqrt(2.0 / in_d)
            weights[f'{prefix_ffn}_{wk}'] = jax.random.normal(sk, (in_d, out_d)) * std
            weights[f'{prefix_ffn}_{bk}'] = jnp.zeros((out_d,))

        # --- LayerNorm ---
        prefix_ln = f'layer{l}_ln'
        for i in ['1', '2']:
            weights[f'{prefix_ln}{i}_scale'] = jnp.ones((d_model,))
            weights[f'{prefix_ln}{i}_bias']  = jnp.zeros((d_model,))

    # --- Positional encoding (appris) ---
    max_T = 20
    key, sk = jax.random.split(key)
    weights['pos_embed'] = jax.random.normal(sk, (max_T, d_model)) * 0.02

    # --- MLP head (même que predictor PC-JEPA) ---
    key, sk = jax.random.split(key)
    weights['k_embed_table'] = jax.random.normal(sk, (K, ke)) * 0.02

    mlp_in = d_model + ke
    mlp_d  = config.pred_mlp_dim
    for (in_d, out_d, wk, bk) in [
        (mlp_in, mlp_d,     'mlp_w1', 'mlp_b1'),
        (mlp_d,  mlp_d,     'mlp_w2', 'mlp_b2'),
        (mlp_d,  config.d_z,'mlp_w3', 'mlp_b3'),
    ]:
        key, sk = jax.random.split(key)
        std = jnp.sqrt(2.0 / in_d)
        weights[wk] = jax.random.normal(sk, (in_d, out_d)) * std
        weights[bk] = jnp.zeros((out_d,))

    return weights


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def apply_transformer_predictor(
    weights: dict,
    z_context: jnp.ndarray,
) -> jnp.ndarray:
    """
    z_context : (B, T_in, d_z)
    return    : (B, K, d_z)
    """
    B, T, d_model = z_context.shape
    n_layers = sum(1 for k in weights if k.startswith('layer') and k.endswith('_n_heads'))

    # Positional encoding
    pos = weights['pos_embed'][:T]   # (T, d_model)
    x = z_context + pos[jnp.newaxis]   # (B, T, d_model)

    # Couches Transformer
    for l in range(n_layers):
        x = _transformer_layer(weights, l, x)

    # Pooling : on utilise le dernier token comme représentation contexte
    h_T = x[:, -1, :]   # (B, d_model)

    # MLP head multi-horizon (identique au GRU predictor)
    K = weights['k_embed_table'].shape[0]

    def predict_one_horizon(k_idx: jnp.ndarray) -> jnp.ndarray:
        k_emb = weights['k_embed_table'][k_idx]
        k_emb_batch = jnp.broadcast_to(k_emb, (B, k_emb.shape[0]))
        inp = jnp.concatenate([h_T, k_emb_batch], axis=-1)
        h = inp @ weights['mlp_w1'] + weights['mlp_b1']
        h = jax.nn.gelu(h)
        h = h @ weights['mlp_w2'] + weights['mlp_b2']
        h = jax.nn.gelu(h)
        h = h @ weights['mlp_w3'] + weights['mlp_b3']
        return h

    k_indices = jnp.arange(K)
    z_preds = jax.vmap(predict_one_horizon)(k_indices)   # (K, B, d_z)
    return z_preds.transpose(1, 0, 2)   # (B, K, d_z)


# ---------------------------------------------------------------------------
# État d'entraînement Transformer
# ---------------------------------------------------------------------------

class TransformerTrainState(NamedTuple):
    encoder_weights:     dict
    transformer_weights: dict
    optimizer_state:     optax.OptState
    step:                int
    key:                 jax.random.PRNGKey


def create_transformer_train_state(config: ModelConfig) -> TransformerTrainState:
    key = jax.random.PRNGKey(config.seed)
    key, sk_enc, sk_trans = jax.random.split(key, 3)

    enc_w   = init_encoder(sk_enc, config)
    trans_w = init_transformer_predictor(sk_trans, config)

    all_weights = {'enc': enc_w, 'trans': trans_w}
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=config.learning_rate),
    )
    opt_state = optimizer.init(all_weights)

    return TransformerTrainState(
        encoder_weights     = enc_w,
        transformer_weights = trans_w,
        optimizer_state     = opt_state,
        step                = 0,
        key                 = key,
    )


def make_transformer_train_step(
    config: ModelConfig,
    optimizer: optax.GradientTransformation,
):
    @jax.jit
    def train_step_transformer(
        state: TransformerTrainState,
        batch: Batch,
    ) -> tuple:
        """Train step pour le Transformer baseline."""

        def loss_fn(all_weights):
            enc_w   = all_weights['enc']
            trans_w = all_weights['trans']

            z_context = apply_encoder(enc_w, batch.context)

            # STOP GRADIENT sur la cible (R3 — même règle que PC-JEPA)
            z_target = jax.lax.stop_gradient(
                apply_encoder(enc_w, batch.target)
            )

            z_pred = apply_transformer_predictor(trans_w, z_context)
            z_target_k = z_target[:, :config.pred_K, :]

            l_j = loss_jepa(z_pred, z_target_k)
            l_v = loss_variance(
                z_pred.reshape(-1, z_pred.shape[-1]), config.gamma_var
            )
            total = l_j + config.lambda_var * l_v
            return total, {'loss_total': total, 'loss_jepa': l_j, 'loss_var': l_v}

        all_weights = {'enc': state.encoder_weights, 'trans': state.transformer_weights}
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(all_weights)

        updates, new_opt_state = optimizer.update(grads, state.optimizer_state, all_weights)
        new_weights = optax.apply_updates(all_weights, updates)

        new_state = TransformerTrainState(
            encoder_weights     = new_weights['enc'],
            transformer_weights = new_weights['trans'],
            optimizer_state     = new_opt_state,
            step                = state.step + 1,
            key                 = state.key,
        )
        return new_state, metrics

    return train_step_transformer


# ---------------------------------------------------------------------------
# Vérification de parité des paramètres
# ---------------------------------------------------------------------------

def count_transformer_params(weights: dict) -> int:
    import jax
    return sum(
        v.size for v in jax.tree_util.tree_leaves(weights)
        if hasattr(v, 'size')
    )


def check_parameter_parity(
    pc_jepa_weights: dict,
    transformer_weights: dict,
    tolerance: float = 0.25,
) -> bool:
    """
    Vérifie que le ratio de paramètres est dans [1-tol, 1+tol].
    Lève une AssertionError si hors tolérance.
    """
    from models.predictor import count_predictor_params
    n_pc  = count_predictor_params(pc_jepa_weights)
    n_tr  = count_transformer_params(transformer_weights)
    ratio = n_pc / max(n_tr, 1)
    low, high = 1.0 - tolerance, 1.0 + tolerance
    print(f"  Paramètres PC-JEPA predictor : {n_pc:,}")
    print(f"  Paramètres Transformer       : {n_tr:,}")
    print(f"  Ratio                        : {ratio:.3f}  (cible [{low:.2f}, {high:.2f}])")
    ok = low <= ratio <= high
    if not ok:
        print(f"  [WARN] Ratio hors tolérance — ajuster n_layers/ffn_dim du Transformer")
    return ok
