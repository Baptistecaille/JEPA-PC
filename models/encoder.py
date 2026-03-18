"""
=== SPEC [encoder] ===
Entrées  : frames (B, T, H, W, C=1) float32 normalisées dans [0, 1]
Sorties  : représentations latentes (B, T, d_z) float32
Architecture:
  Conv2D(1→32,  k=4, s=2) → GELU → (B, 32, 32)
  Conv2D(32→64, k=4, s=2) → GELU → (B, 16, 16)
  Conv2D(64→128,k=4, s=2) → GELU → (B,  8,  8)
  Flatten → Linear(8*8*128 → d_z) → LayerNorm
Invariants:
  - Appliqué frame par frame via double vmap (B, T)
  - JIT-compatible (pas de boucle Python sur T ou B)
  - Poids stockés dans un dict plat (R4)
  - Aucun magic number (R7)
=== FIN SPEC ===
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple
from config import ModelConfig


# ---------------------------------------------------------------------------
# Calcul des dimensions intermédiaires
# ---------------------------------------------------------------------------

def _conv_out_size(in_size: int, kernel: int, stride: int) -> int:
    """Taille de sortie d'une conv avec padding=VALID."""
    return (in_size - kernel) // stride + 1


def _encoder_flat_dim(config: ModelConfig) -> int:
    """Dimension aplatie après les 3 convolutions."""
    h = config.img_size if hasattr(config, 'img_size') else 64
    for _ in config.enc_channels:
        h = _conv_out_size(h, config.enc_kernel, config.enc_stride)
    return h * h * config.enc_channels[-1]


# ---------------------------------------------------------------------------
# Initialisation des poids (Xavier / He)
# ---------------------------------------------------------------------------

def init_encoder(key: jax.random.PRNGKey, config: ModelConfig) -> dict:
    """
    Initialise les poids CNN. Retourne un dict plat (R4).

    Clés :
      conv{i}_w : (out_c, in_c, k, k)   — noyaux de convolution (format OIHW)
      conv{i}_b : (out_c,)              — biais
      linear_w  : (flat_dim, d_z)
      linear_b  : (d_z,)
      ln_scale  : (d_z,)               — LayerNorm
      ln_bias   : (d_z,)
    """
    weights = {}
    channels = (1,) + config.enc_channels   # (1, 32, 64, 128)

    for i, (in_c, out_c) in enumerate(zip(channels[:-1], channels[1:])):
        key, sk = jax.random.split(key)
        k = config.enc_kernel
        # Xavier uniform : fan_in = in_c * k * k, fan_out = out_c * k * k
        fan = in_c * k * k
        std = jnp.sqrt(2.0 / fan)
        weights[f'conv{i}_w'] = jax.random.normal(sk, (out_c, in_c, k, k)) * std
        weights[f'conv{i}_b'] = jnp.zeros((out_c,))

    # Dimension aplatie
    img_size = 64   # R7: taille standard Moving MNIST — doit correspondre à DataConfig.img_size
    h = img_size
    for _ in config.enc_channels:
        h = _conv_out_size(h, config.enc_kernel, config.enc_stride)
    flat_dim = h * h * config.enc_channels[-1]

    key, sk = jax.random.split(key)
    std_lin = jnp.sqrt(2.0 / flat_dim)
    weights['linear_w'] = jax.random.normal(sk, (flat_dim, config.d_z)) * std_lin
    weights['linear_b'] = jnp.zeros((config.d_z,))

    # LayerNorm parameters
    weights['ln_scale'] = jnp.ones((config.d_z,))
    weights['ln_bias']  = jnp.zeros((config.d_z,))

    return weights


# ---------------------------------------------------------------------------
# Forward pass d'une seule frame : (H, W, C) → d_z
# ---------------------------------------------------------------------------

def _layer_norm(x: jnp.ndarray, scale: jnp.ndarray, bias: jnp.ndarray,
                eps: float = 1e-5) -> jnp.ndarray:
    """LayerNorm sur le dernier axe."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return scale * (x - mean) / jnp.sqrt(var + eps) + bias


def _cnn_forward(weights: dict, frame: jnp.ndarray) -> jnp.ndarray:
    """
    Encode une seule frame.
    frame : (H, W, C)  — C=1
    return: (d_z,)
    """
    # JAX conv attend (N, C, H, W) en mode NCHW — on ajoute la dim batch
    # frame: (H, W, C) → (1, C, H, W)
    x = frame.transpose(2, 0, 1)[jnp.newaxis]   # (1, C, H, W)

    n_conv = len([k for k in weights if k.startswith('conv') and k.endswith('_w')])
    for i in range(n_conv):
        w = weights[f'conv{i}_w']   # (out_c, in_c, k, k)
        b = weights[f'conv{i}_b']   # (out_c,)
        k = w.shape[-1]
        x = jax.lax.conv_general_dilated(
            x, w,
            window_strides=(2, 2),
            padding='VALID',
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
        )
        x = x + b[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
        x = jax.nn.gelu(x)

    # Flatten : (1, C_out, H_out, W_out) → (flat_dim,)
    x = x.reshape(-1)

    # Linear
    x = x @ weights['linear_w'] + weights['linear_b']

    # LayerNorm
    x = _layer_norm(x, weights['ln_scale'], weights['ln_bias'])

    return x   # (d_z,)


# ---------------------------------------------------------------------------
# Encodage d'un batch de séquences : (B, T, H, W, C) → (B, T, d_z)
# ---------------------------------------------------------------------------

def apply_encoder(weights: dict, frames: jnp.ndarray) -> jnp.ndarray:
    """
    frames : (B, T, H, W, C)
    return : (B, T, d_z)

    Double vmap : d'abord sur B, ensuite sur T.
    JIT-compatible — aucune boucle Python sur B ou T.
    """
    # vmap sur T (inner), puis sur B (outer)
    encode_seq = jax.vmap(
        lambda f_seq: jax.vmap(
            lambda f: _cnn_forward(weights, f)
        )(f_seq)
    )
    return encode_seq(frames)   # (B, T, d_z)


# ---------------------------------------------------------------------------
# Sanity checks (R6)
# ---------------------------------------------------------------------------

def run_sanity_checks_encoder(config: ModelConfig) -> None:
    """
    R6 — Sanity checks du module encoder.
    Vérifie :
      [E1] La forme de sortie est (B, T, d_z)
      [E2] Pas de NaN dans les sorties
      [E3] La fonction est JIT-compilable
      [E4] Le gradient est calculable par rapport aux poids
      [E5] Deux initialisations avec des seeds différentes donnent des poids différents
    """
    print("[Sanity] models.encoder — début")

    B, T, H, W, C = 2, 4, 64, 64, 1

    key = jax.random.PRNGKey(config.seed)
    key, sk1, sk2 = jax.random.split(key, 3)

    weights = init_encoder(sk1, config)

    # [E1] Forme de sortie
    frames = jnp.ones((B, T, H, W, C))
    z = apply_encoder(weights, frames)
    assert z.shape == (B, T, config.d_z), \
        f"[E1] shape attendue ({B},{T},{config.d_z}), obtenue {z.shape}"
    print(f"  [E1] ✓ shape: {z.shape}")

    # [E2] Pas de NaN
    assert not jnp.any(jnp.isnan(z)), "[E2] NaN dans la sortie de l'encoder"
    print("  [E2] ✓ pas de NaN")

    # [E3] JIT-compilable
    apply_jit = jax.jit(apply_encoder)
    z_jit = apply_jit(weights, frames)
    assert jnp.allclose(z, z_jit, atol=1e-5), "[E3] JIT donne des résultats différents"
    print("  [E3] ✓ JIT-compilable")

    # [E4] Gradient calculable par rapport aux poids
    def loss_fn(w):
        return jnp.mean(apply_encoder(w, frames) ** 2)

    grads = jax.grad(loss_fn)(weights)
    for k, g in grads.items():
        assert not jnp.any(jnp.isnan(g)), f"[E4] NaN dans le gradient de {k}"
    print("  [E4] ✓ gradients calculables, pas de NaN")

    # [E5] Seeds différentes → poids différents
    weights2 = init_encoder(sk2, config)
    assert not jnp.allclose(weights['conv0_w'], weights2['conv0_w']), \
        "[E5] deux seeds identiques donnent les mêmes poids (?)"
    print("  [E5] ✓ poids différents selon le seed")

    print("[Sanity] models.encoder — OK\n")
