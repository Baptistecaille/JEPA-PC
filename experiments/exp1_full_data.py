"""
Expérience 1 — Qualité absolue (n=10000 exemples d'entraînement)
Objectif : Vérifier que PC-JEPA apprend correctement sur le dataset complet.
"""
import jax.numpy as jnp
from config import ModelConfig
from data.moving_mnist import DataConfig, get_dataloaders
from training.trainer import create_train_state, make_train_step, evaluate, _build_optimizer
from data.moving_mnist import Batch


def run_exp1(config: ModelConfig, data_config: DataConfig = None) -> dict:
    """
    Entraîne PC-JEPA sur le dataset complet (n=10000).
    Évalue sur le test set.
    Retourne les métriques finales.
    """
    if data_config is None:
        data_config = DataConfig(seed=config.seed)

    print("=" * 60)
    print("[Exp1] Qualité absolue — n=10000, PC-JEPA")
    print("=" * 60)

    train_fn, val_fn, test_fn = get_dataloaders(data_config)

    state    = create_train_state(config)
    optimizer = _build_optimizer(config)
    train_step = make_train_step(config, optimizer)

    steps_per_epoch = data_config.n_train // data_config.batch_size
    n_steps = config.n_epochs * steps_per_epoch

    train_iter = train_fn()
    history = []

    for step in range(n_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = train_fn()
            batch = next(train_iter)

        batch_jnp = Batch(
            context = jnp.array(batch.context),
            target  = jnp.array(batch.target),
        )
        state, metrics = train_step(state, batch_jnp)

        if step % 200 == 0:
            history.append({k: float(v) for k, v in metrics.items()})
            print(
                f"  step={step:5d}/{n_steps}  "
                f"loss={float(metrics['loss_total']):.4f}  "
                f"jepa={float(metrics['loss_jepa']):.4f}  "
                f"T_conv={int(metrics['T_conv'])}"
            )

    print("\n[Exp1] Évaluation sur le test set...")
    test_metrics = evaluate(state, test_fn, config, n_batches=100)
    print(f"  NMSE={test_metrics.get('nmse', 0):.4f}  "
          f"collapse={test_metrics.get('collapse_score', 0):.4f}")

    return {
        'final_metrics': test_metrics,
        'history':       history,
        'state':         state,
    }
