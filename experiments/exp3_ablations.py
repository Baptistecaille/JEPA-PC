"""
Expérience 3 — Ablations PC-JEPA
Objectif : Identifier la contribution de chaque composant.

Ablations :
  A1 : Sans PC  (predictor direct sur z_context sans boucle d'inférence)
  A2 : Sans JEPA stop-gradient (permet le gradient dans l'encoder cible)  ← invalide R3
  A3 : PC_max_iter = 1 (une seule itération d'inférence)
  A4 : Sans L_var (pas d'anti-collapse)
  A5 : PC_n_layers = 1 (hiérarchie plate)
"""
import jax.numpy as jnp
from config import ModelConfig
from data.moving_mnist import DataConfig, get_dataloaders, Batch
from training.trainer import create_train_state, make_train_step, evaluate, _build_optimizer


# ---------------------------------------------------------------------------
# Configs d'ablation (R7 — pas de magic numbers)
# ---------------------------------------------------------------------------

def _ablation_config(base: ModelConfig, **overrides) -> ModelConfig:
    return ModelConfig(**{**base._asdict(), **overrides})


ABLATIONS = {
    'full_model':      {},                              # modèle complet (référence)
    'no_pc_1iter':     {'pc_n_inference_steps': 1},      # A3 : 1 seule itération PC
    'no_pc_hierarchy': {'pc_n_layers': 1},              # A5 : hiérarchie plate
    'no_var_loss':     {'lambda_var': 0.0},             # A4 : sans L_var
    'no_pc_loss':      {'lambda_pc': 0.0},              # A4b: sans L_PC
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_ablation(
    name: str,
    config: ModelConfig,
    data_config: DataConfig,
    train_fn,
    test_fn,
    n_steps: int = 5000,
) -> dict:
    """Entraîne un modèle avec la configuration d'ablation et l'évalue."""
    overrides = ABLATIONS[name]
    abl_config = _ablation_config(config, **overrides)

    state    = create_train_state(abl_config)
    optimizer = _build_optimizer(abl_config)
    step_fn  = make_train_step(abl_config, optimizer)

    train_iter = train_fn()
    for step in range(n_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = train_fn()
            batch = next(train_iter)

        batch_jnp = Batch(jnp.array(batch.context), jnp.array(batch.target))
        state, _ = step_fn(state, batch_jnp)

    metrics = evaluate(state, test_fn, abl_config, n_batches=50)
    return metrics


def run_exp3(config: ModelConfig, data_config: DataConfig = None) -> dict:
    """Lance toutes les ablations et compare."""
    if data_config is None:
        data_config = DataConfig(seed=config.seed)

    train_fn, val_fn, test_fn = get_dataloaders(data_config)

    print("=" * 60)
    print("[Exp3] Ablations PC-JEPA")
    print("=" * 60)

    results = {}
    for name in ABLATIONS:
        print(f"\n  Ablation : {name}")
        try:
            r = run_ablation(name, config, data_config, train_fn, test_fn)
            results[name] = r
            print(f"    nmse={r.get('nmse', '?'):.4f}  "
                  f"collapse_ctx={r.get('collapse_ctx', '?'):.4f}")
        except Exception as e:
            print(f"    [ERREUR] {e}")
            results[name] = {}

    _print_ablation_summary(results)
    return results


def _print_ablation_summary(results: dict) -> None:
    print("\n" + "=" * 60)
    print("RÉSUMÉ ABLATIONS")
    print(f"{'Ablation':<25}  {'NMSE':>8}  {'Collapse':>10}")
    print("-" * 48)
    for name, r in results.items():
        nmse_v = r.get('nmse', float('nan'))
        coll_v = r.get('collapse_ctx', float('nan'))
        print(f"{name:<25}  {nmse_v:>8.4f}  {coll_v:>10.4f}")
    print("=" * 60)
