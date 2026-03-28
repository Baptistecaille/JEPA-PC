"""
Diagnostic de convergence PC.

Lance PC-JEPA sur n ∈ {100, 1000, 4000}, puis mesure :
  - T_conv
  - erreur PC initiale
  - erreur PC finale

Le script peut aussi faire un sweep ciblé sur pc_tol et pc_n_inference_steps
pour distinguer un seuil irréaliste d'une dynamique PC réellement non convergente.

Usage : python check_tconv.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
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
N_TRAIN_STEPS = 500          # assez pour que la MSE descende sous le pc_tol de base
N_EVAL_BATCHES = 10          # batches pour mesurer T_conv
SEED = 42
BASE_PC_TOL = 0.1
BASE_PC_N_INFERENCE_STEPS = 3000
DEFAULT_TOLS = (BASE_PC_TOL, 0.5, 0.8)
DEFAULT_STEPS = (50, 100, 200)


def measure_tconv(state, config, data_config, n):
    """Mesure T_conv sur N_EVAL_BATCHES après entraînement."""
    loader_fn = get_subset_dataloader(data_config, n_samples=max(n, 64), seed=SEED)
    enc_w = state.encoder_weights
    pc_w  = flat_dict_to_pc_weights(state.pc_weights_flat, config)

    tconv_values = []
    error_inits = []
    error_finals = []

    for i, batch in enumerate(loader_fn()):
        if i >= N_EVAL_BATCHES:
            break
        obs = apply_encoder(enc_w, jnp.array(batch.context))[:, -1, :]  # (B, d_z)
        init_state = init_pc_from_encoding(obs, pc_w, config)
        error_inits.append(float(jnp.mean(init_state.errors ** 2)))

        # run_inference_loop_debug : retourne l'historique complet
        _, T_conv, error_history = run_inference_loop_debug(
            init_state, pc_w, obs, config
        )
        tconv_values.append(T_conv)
        error_finals.append(error_history[-1])

    return (
        np.array(tconv_values),
        np.array(error_inits),
        np.array(error_finals),
    )


def _train_short_run(n, config, data_config):
    """Entraîne un petit modèle puis retourne le TrainState."""
    loader_fn = get_subset_dataloader(data_config, n_samples=n, seed=SEED)
    state     = create_train_state(config)
    optimizer = _build_optimizer(config)
    step_fn   = make_train_step(config, optimizer)
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
    return state


def run_check(base_config=None, sweep_tols=None, sweep_steps=None):
    config = base_config or ModelConfig(
        seed=SEED,
        pc_tol=BASE_PC_TOL,
        pc_n_inference_steps=BASE_PC_N_INFERENCE_STEPS,
    )
    data_config = DataConfig(seed=SEED, batch_size=16)
    sweep_tols = sweep_tols or (config.pc_tol,)
    sweep_steps = sweep_steps or (config.pc_n_inference_steps,)

    print("=" * 62)
    print("CHECK T_CONV — diagnostic de convergence PC")
    print(f"  base pc_tol={config.pc_tol}  base pc_n_inference_steps={config.pc_n_inference_steps}")
    print("=" * 62)

    results = {}

    for n in CONTROL_NS:
        print(f"\n── n={n} ──────────────────────────────────────────")

        train_cfg = config._replace(seed=SEED)
        state = _train_short_run(n, train_cfg, data_config)

        for tol in sweep_tols:
            for n_steps in sweep_steps:
                local_cfg = train_cfg._replace(
                    pc_tol=tol,
                    pc_n_inference_steps=n_steps,
                )
                tconv_arr, err_init_arr, err_arr = measure_tconv(
                    state, local_cfg, data_config, n
                )
                converged = (tconv_arr < local_cfg.pc_n_inference_steps).mean() * 100

                key = (n, tol, n_steps)
                results[key] = {
                    'T_conv_mean': float(tconv_arr.mean()),
                    'T_conv_std':  float(tconv_arr.std()),
                    'T_conv_min':  int(tconv_arr.min()),
                    'T_conv_max':  int(tconv_arr.max()),
                    'pct_converged': float(converged),
                    'mse_init_mean': float(err_init_arr.mean()),
                    'mse_final_mean': float(err_arr.mean()),
                }

                print(f"\n  [tol={tol:.3f}, steps={n_steps}]")
                print(f"    T_conv : mean={tconv_arr.mean():.1f}  "
                      f"std={tconv_arr.std():.1f}  "
                      f"min={tconv_arr.min()}  max={tconv_arr.max()}")
                print(f"    MSE init  : mean={err_init_arr.mean():.6f}")
                print(f"    MSE final : mean={err_arr.mean():.6f}  "
                      f"(seuil pc_tol={local_cfg.pc_tol})")
                print(f"    Gain      : mean={(err_init_arr.mean() - err_arr.mean()):.6f}")
                print(f"    Convergence avant pc_n_inference_steps : {converged:.0f}% des batches")

    # Verdict final
    print("\n" + "=" * 62)
    print("VERDICT")
    print(f"  {'n':>6}  {'tol':>6}  {'steps':>7}  {'Tconv':>8}  {'MSEf':>10}  {'%conv':>7}")
    print("  " + "-" * 58)
    all_variable = True
    for (n, tol, n_steps), r in results.items():
        variable = r['T_conv_std'] > 0.5 or r['T_conv_max'] < n_steps
        flag = "✓" if variable else "✗ SATURÉ"
        if not variable:
            all_variable = False
        print(f"  {n:>6}  {tol:>6.2f}  {n_steps:>7}  {r['T_conv_mean']:>8.1f}  "
              f"{r['mse_final_mean']:>10.4f}  {r['pct_converged']:>6.0f}%  {flag}")

    print()
    if all_variable:
        print("✅ T_conv est variable — fix compute_max_error confirmé.")
    else:
        print("❌ T_conv encore saturé sur certaines configs — la dynamique PC reste à auditer.")
        print("   Si même pc_tol élevé et plus de steps saturent, viser inference_step_fn / free_energy.")
    print("=" * 62)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diagnostic de convergence PC.")
    parser.add_argument('--sweep', action='store_true',
                        help="Teste plusieurs valeurs de pc_tol et pc_n_inference_steps.")
    args = parser.parse_args()

    if args.sweep:
        results = run_check(
            sweep_tols=DEFAULT_TOLS,
            sweep_steps=DEFAULT_STEPS,
        )
    else:
        results = run_check()
