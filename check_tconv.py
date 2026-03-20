"""
Vérification rapide : T_conv est-il variable après le fix compute_max_error ?

Lance PC-JEPA sur n ∈ {100, 1000, 4000}, 1 seed chacun, ~200 steps.
Affiche T_conv moyen + distribution pour confirmer la convergence effective.

Usage : python check_tconv.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np

from config import ModelConfig
from data.moving_mnist import DataConfig, get_subset_dataloader, Batch
from models.encoder import init_encoder, apply_encoder
from models.pc_nodes import (
    init_pc_weights, init_pc_from_encoding,
    run_inference_loop, run_inference_loop_debug,
    pc_weights_to_flat_dict, flat_dict_to_pc_weights,
)
from training.trainer import create_train_state, make_train_step, _build_optimizer

CONTROL_NS = (100, 1000, 4000)
N_TRAIN_STEPS = 500          # assez pour que la MSE descende sous pc_tol=1e-2
N_EVAL_BATCHES = 10          # batches pour mesurer T_conv
SEED = 42


def measure_tconv(state, config, data_config, n):
    """Mesure T_conv sur N_EVAL_BATCHES après entraînement."""
    loader_fn = get_subset_dataloader(data_config, n_samples=max(n, 64), seed=SEED)
    enc_w = state.encoder_weights
    pc_w  = flat_dict_to_pc_weights(state.pc_weights_flat, config)

    tconv_values = []
    error_finals = []

    for i, batch in enumerate(loader_fn()):
        if i >= N_EVAL_BATCHES:
            break
        obs = apply_encoder(enc_w, jnp.array(batch.context))[:, -1, :]  # (B, d_z)
        init_state = init_pc_from_encoding(obs, pc_w, config)

        # run_inference_loop_debug : retourne l'historique complet
        _, T_conv, error_history = run_inference_loop_debug(
            init_state, pc_w, obs, config
        )
        tconv_values.append(T_conv)
        error_finals.append(error_history[-1])

    return np.array(tconv_values), np.array(error_finals)


def run_check():
    config      = ModelConfig(seed=SEED, pc_max_iter=100, pc_tol=1e-4)
    data_config = DataConfig(seed=SEED, batch_size=16)

    print("=" * 62)
    print("CHECK T_CONV — après fix compute_max_error (L∞ → MSE)")
    print(f"  pc_tol={config.pc_tol}  pc_max_iter={config.pc_max_iter}")
    print("=" * 62)

    results = {}

    for n in CONTROL_NS:
        print(f"\n── n={n} ──────────────────────────────────────────")

        # Entraînement court
        loader_fn = get_subset_dataloader(data_config, n_samples=n, seed=SEED)
        local_cfg = ModelConfig(seed=SEED, pc_max_iter=100, pc_tol=1e-4)
        state     = create_train_state(local_cfg)
        optimizer = _build_optimizer(local_cfg)
        step_fn   = make_train_step(local_cfg, optimizer)

        train_iter = loader_fn()
        for step in range(N_TRAIN_STEPS):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = loader_fn()
                batch = next(train_iter)
            batch_jnp = Batch(jnp.array(batch.context), jnp.array(batch.target))
            state, metrics = step_fn(state, batch_jnp)

            if step % 50 == 0:
                print(f"  step={step:3d}  loss={float(metrics['loss_total']):.4f}"
                      f"  T_conv(train)={int(metrics['T_conv']):3d}"
                      f"  pc_err={float(metrics['pc_error']):.6f}")

        # Mesure T_conv en évaluation
        tconv_arr, err_arr = measure_tconv(state, local_cfg, data_config, n)

        print(f"\n  T_conv : mean={tconv_arr.mean():.1f}  "
              f"std={tconv_arr.std():.1f}  "
              f"min={tconv_arr.min()}  max={tconv_arr.max()}")
        print(f"  MSE final : mean={err_arr.mean():.6f}  "
              f"(seuil pc_tol={local_cfg.pc_tol})")

        converged = (tconv_arr < local_cfg.pc_max_iter).mean() * 100
        print(f"  Convergence avant pc_max_iter : {converged:.0f}% des batches")

        results[n] = {
            'T_conv_mean': float(tconv_arr.mean()),
            'T_conv_std':  float(tconv_arr.std()),
            'T_conv_min':  int(tconv_arr.min()),
            'T_conv_max':  int(tconv_arr.max()),
            'pct_converged': float(converged),
            'mse_final_mean': float(err_arr.mean()),
        }

    # Verdict final
    print("\n" + "=" * 62)
    print("VERDICT")
    print(f"  {'n':>6}  {'T_conv_mean':>12}  {'T_conv_max':>10}  {'% conv.':>8}")
    print("  " + "-" * 44)
    all_variable = True
    for n, r in results.items():
        variable = r['T_conv_std'] > 0.5 or r['T_conv_max'] < config.pc_max_iter
        flag = "✓" if variable else "✗ SATURÉ"
        if not variable:
            all_variable = False
        print(f"  {n:>6}  {r['T_conv_mean']:>12.1f}  {r['T_conv_max']:>10}  "
              f"{r['pct_converged']:>7.0f}%  {flag}")

    print()
    if all_variable:
        print("✅ T_conv est variable — fix compute_max_error confirmé.")
        print("   Prêt pour exp2 complet et exp3.")
    else:
        print("❌ T_conv encore saturé sur certains n — vérifier pc_tol / pc_alpha.")
    print("=" * 62)

    return results


if __name__ == '__main__':
    results = run_check()
