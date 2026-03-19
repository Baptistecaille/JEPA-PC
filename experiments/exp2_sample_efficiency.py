"""
Expérience 2 — Sample Efficiency (expérience principale de thèse)
Objectif : Montrer que PC-JEPA surpasse Transformer baseline
           en efficacité d'échantillonnage sur Moving MNIST.

Courbe principale : NMSE(test) en fonction du nombre d'exemples d'entraînement n.
"""
import json
import os
from datetime import datetime

import jax.numpy as jnp

from config import ModelConfig
from data.moving_mnist import DataConfig, get_dataloaders, get_subset_dataloader, Batch
from training.trainer import create_train_state, make_train_step, evaluate, _build_optimizer
from models.transformer_baseline import (
    create_transformer_train_state, make_transformer_train_step,
)
import optax


# ---------------------------------------------------------------------------
# Configuration de l'expérience (R7 — constantes nommées)
# ---------------------------------------------------------------------------

EFFICIENCY_NS  = (100, 250, 500, 1000, 2000, 4000, 7000, 10000)
SEEDS          = (42, 137, 2024)
N_EPOCHS_SHORT = 50    # budget réduit pour les petits n
N_EPOCHS_FULL  = 100   # budget complet pour n ≥ 2000

N_THRESHOLD_FULL = 2000   # seuil n ≥ N_THRESHOLD_FULL → N_EPOCHS_FULL


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _n_epochs_for(n: int) -> int:
    return N_EPOCHS_FULL if n >= N_THRESHOLD_FULL else N_EPOCHS_SHORT


def _steps_for(n: int, batch_size: int) -> int:
    n_epochs = _n_epochs_for(n)
    steps_per_epoch = max(n // batch_size, 1)
    return n_epochs * steps_per_epoch


def save_results(results: dict, tag: str = 'exp2') -> str:
    os.makedirs('results', exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'results/{tag}_{ts}.json'
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {m: float(val) for m, val in v.items() if not hasattr(val, '__len__')}
        else:
            serializable[k] = float(v) if hasattr(v, 'item') else v
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Résultats sauvegardés : {path}")
    return path


# ---------------------------------------------------------------------------
# Entraînement d'un modèle PC-JEPA sur n exemples
# ---------------------------------------------------------------------------

def _train_pc_jepa(
    n: int,
    seed: int,
    config: ModelConfig,
    data_config: DataConfig,
    test_fn,
) -> dict:
    loader_fn = get_subset_dataloader(data_config, n_samples=n, seed=seed)

    local_config = ModelConfig(
        **{k: v for k, v in config._asdict().items() if k != 'seed'},
        seed=seed,
    )
    state    = create_train_state(local_config)
    optimizer = _build_optimizer(local_config)
    step_fn  = make_train_step(local_config, optimizer)

    n_steps = _steps_for(n, data_config.batch_size)
    train_iter = loader_fn()

    for step in range(n_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = loader_fn()
            batch = next(train_iter)

        batch_jnp = Batch(jnp.array(batch.context), jnp.array(batch.target))
        state, _ = step_fn(state, batch_jnp)

    return evaluate(state, test_fn, local_config, n_batches=50)


# ---------------------------------------------------------------------------
# Entraînement d'un modèle Transformer sur n exemples
# ---------------------------------------------------------------------------

def _train_transformer(
    n: int,
    seed: int,
    config: ModelConfig,
    data_config: DataConfig,
    test_fn,
) -> dict:
    loader_fn = get_subset_dataloader(data_config, n_samples=n, seed=seed)

    local_config = ModelConfig(
        **{k: v for k, v in config._asdict().items() if k != 'seed'},
        seed=seed,
    )
    state = create_transformer_train_state(local_config)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=local_config.learning_rate),
    )
    step_fn = make_transformer_train_step(local_config, optimizer)

    n_steps = _steps_for(n, data_config.batch_size)
    train_iter = loader_fn()

    for step in range(n_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = loader_fn()
            batch = next(train_iter)

        batch_jnp = Batch(jnp.array(batch.context), jnp.array(batch.target))
        state, _ = step_fn(state, batch_jnp)

    # Évaluation Transformer (NMSE uniquement — pas de PC state)
    from models.encoder import apply_encoder
    from models.transformer_baseline import apply_transformer_predictor
    from eval.metrics import nmse

    enc_w   = state.encoder_weights
    trans_w = state.transformer_weights
    all_nmse = []

    for i, batch in enumerate(test_fn()):
        if i >= 50:
            break
        z_context  = apply_encoder(enc_w, jnp.array(batch.context))
        z_target   = apply_encoder(enc_w, jnp.array(batch.target))
        z_pred     = apply_transformer_predictor(trans_w, z_context, local_config)
        z_target_k = z_target[:, :local_config.pred_K, :]
        all_nmse.append(float(nmse(z_pred, z_target_k)))

    import numpy as np
    return {'nmse': float(np.mean(all_nmse))}


# ---------------------------------------------------------------------------
# Expérience unique
# ---------------------------------------------------------------------------

def run_single_experiment(
    n: int,
    model_type: str,
    seed: int,
    config: ModelConfig,
    data_config: DataConfig = None,
    test_fn=None,
) -> dict:
    """
    Entraîne un modèle sur n exemples, évalue sur le test set complet.
    model_type : 'pc_jepa' ou 'transformer'
    """
    if data_config is None:
        data_config = DataConfig(seed=seed)
    if test_fn is None:
        _, _, test_fn = get_dataloaders(data_config)

    if model_type == 'pc_jepa':
        return _train_pc_jepa(n, seed, config, data_config, test_fn)
    elif model_type == 'transformer':
        return _train_transformer(n, seed, config, data_config, test_fn)
    else:
        raise ValueError(f"model_type inconnu : {model_type}")


# ---------------------------------------------------------------------------
# Expérience complète
# ---------------------------------------------------------------------------

def run_exp2(config: ModelConfig, data_config: DataConfig = None) -> dict:
    """
    Lance toutes les combinaisons (model_type, n, seed).
    Sauvegarde les résultats en JSON.
    """
    if data_config is None:
        data_config = DataConfig(seed=config.seed)

    # Test set partagé entre tous les modèles (même données, même seed)
    _, _, test_fn = get_dataloaders(data_config)

    print("=" * 60)
    print("[Exp2] Sample Efficiency — PC-JEPA vs Transformer")
    print(f"  n ∈ {EFFICIENCY_NS}")
    print(f"  seeds = {SEEDS}")
    print("=" * 60)

    results = {}
    total = len(EFFICIENCY_NS) * len(SEEDS) * 2
    done  = 0

    for model_type in ['pc_jepa', 'transformer']:
        for n in EFFICIENCY_NS:
            for seed in SEEDS:
                done += 1
                print(f"\n[{done}/{total}] model={model_type}, n={n}, seed={seed}")
                try:
                    result = run_single_experiment(
                        n, model_type, seed, config, data_config, test_fn
                    )
                    key = f"{model_type}_n{n}_seed{seed}"
                    results[key] = result
                    print(f"  → nmse={result.get('nmse', '?'):.4f}")
                except Exception as e:
                    print(f"  [ERREUR] {e}")
                    results[f"{model_type}_n{n}_seed{seed}"] = {'nmse': float('nan')}

    path = save_results(results)
    _print_summary(results)

    return results


def _print_summary(results: dict) -> None:
    """Affiche un résumé des résultats moyennés par n."""
    import numpy as np
    print("\n" + "=" * 60)
    print("RÉSUMÉ — NMSE moyen sur les seeds")
    print(f"{'n':>8}  {'PC-JEPA':>10}  {'Transformer':>12}")
    print("-" * 35)
    for n in EFFICIENCY_NS:
        pc_vals  = [results.get(f'pc_jepa_n{n}_seed{s}',{}).get('nmse', float('nan'))
                    for s in SEEDS]
        tr_vals  = [results.get(f'transformer_n{n}_seed{s}',{}).get('nmse', float('nan'))
                    for s in SEEDS]
        pc_mean  = np.nanmean(pc_vals)
        tr_mean  = np.nanmean(tr_vals)
        print(f"{n:>8}  {pc_mean:>10.4f}  {tr_mean:>12.4f}")
    print("=" * 60)
