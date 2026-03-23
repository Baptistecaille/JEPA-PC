"""
Test isolé du Transformer baseline — diagnostic et validation.
Lance 200 steps d'entraînement sur n=500 exemples, seed=42.
Affiche les métriques et la NMSE finale.
"""
import traceback
import jax.numpy as jnp
import optax

from config import ModelConfig
from data.moving_mnist import DataConfig, get_subset_dataloader, Batch
from models.transformer_baseline import (
    create_transformer_train_state,
    make_transformer_train_step,
    apply_transformer_predictor,
)
from models.encoder import apply_encoder
from eval.metrics import nmse


def main():
    config      = ModelConfig()
    data_config = DataConfig()
    N           = 500
    SEED        = 42
    N_STEPS     = 200

    print(f"[test_transformer] n={N}, seed={SEED}, steps={N_STEPS}")

    # --- Optimizer (même pattern que PC-JEPA) ---
    n_steps  = N_STEPS
    warmup_t = min(config.warmup_steps, n_steps // 10)
    print(f"  n_steps={n_steps}, warmup_t={warmup_t}, lr={config.learning_rate}")

    schedule = optax.warmup_cosine_decay_schedule(
        init_value   = 0.0,
        peak_value   = config.learning_rate,
        warmup_steps = warmup_t,
        decay_steps  = n_steps,
        end_value    = config.learning_rate * 0.01,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule),
    )

    # --- State ---
    state = create_transformer_train_state(config, optimizer)
    print("  State créé OK")

    # --- Step function ---
    step_fn = make_transformer_train_step(config, optimizer)
    print("  Step function créée OK")

    # --- Data ---
    loader_fn  = get_subset_dataloader(data_config, n_samples=N, seed=SEED)
    train_iter = loader_fn()
    print("  Dataloader OK")

    # --- Training loop ---
    for step in range(N_STEPS):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = loader_fn()
            batch = next(train_iter)

        batch_jnp = Batch(jnp.array(batch.context), jnp.array(batch.target))

        try:
            state, metrics = step_fn(state, batch_jnp)
        except Exception as e:
            print(f"\n[ERREUR] step={step}")
            traceback.print_exc()
            return

        if step % 50 == 0:
            print(f"  step={step:4d}  loss={float(metrics['loss_total']):.4f}"
                  f"  jepa={float(metrics['loss_jepa']):.4f}"
                  f"  var={float(metrics['loss_var']):.4f}")

    # --- Evaluation ---
    from data.moving_mnist import get_dataloaders
    _, _, test_fn = get_dataloaders(data_config)

    enc_w   = state.encoder_weights
    trans_w = state.transformer_weights
    all_nmse = []

    for i, batch in enumerate(test_fn()):
        if i >= 20:
            break
        z_ctx  = apply_encoder(enc_w, jnp.array(batch.context))
        z_tgt  = apply_encoder(enc_w, jnp.array(batch.target))
        z_pred = apply_transformer_predictor(trans_w, z_ctx, config)
        z_tgt_k = z_tgt[:, :config.pred_K, :]
        all_nmse.append(float(nmse(z_pred, z_tgt_k)))

    import numpy as np
    mean_nmse = float(np.mean(all_nmse))
    print(f"\n  NMSE finale (20 batches) = {mean_nmse:.4f}")
    print("[test_transformer] OK" if not (mean_nmse != mean_nmse) else "[test_transformer] NaN détecté")


if __name__ == '__main__':
    main()
