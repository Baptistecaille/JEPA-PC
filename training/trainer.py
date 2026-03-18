"""
=== SPEC [trainer] ===
Rôle : Orchestration des deux boucles, logging, interface pour experiments/
Invariants:
  - ORDRE IMMUABLE dans train_step :
      1. Encoder contexte
      2. Encoder cible avec STOP GRADIENT (R3)
      3. Boucle 1 — inférence PC (états libres, poids figés)
      4. Predictor
      5. Calcul des pertes
      6. Boucle 2 — mise à jour poids (optax)
      7. Logging
  - TrainState est un NamedTuple (R4)
  - Reproductibilité via key JAX (R5)
=== FIN SPEC ===
"""
import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple, Iterator
from functools import partial

from config import ModelConfig
from models.encoder import init_encoder, apply_encoder
from models.predictor import init_predictor, apply_predictor
from models.pc_nodes import (
    PCWeights, init_pc_weights, init_pc_from_encoding, run_inference_loop,
    pc_weights_to_flat_dict, flat_dict_to_pc_weights,
)
from training.losses import total_loss, loss_jepa
from data.moving_mnist import Batch, DataConfig


# ---------------------------------------------------------------------------
# État d'entraînement (R4 — NamedTuple)
# ---------------------------------------------------------------------------

class TrainState(NamedTuple):
    """État global de l'entraînement — tout ce qui change au fil du temps."""
    encoder_weights:   dict
    predictor_weights: dict
    pc_weights_flat:   dict          # dict plat (compatible optax)
    optimizer_state:   optax.OptState
    step:              int
    key:               jax.random.PRNGKey


def _build_optimizer(config: ModelConfig) -> optax.GradientTransformation:
    """Construit l'optimizer avec warmup + cosine decay."""
    schedule = optax.warmup_cosine_decay_schedule(
        init_value  = 0.0,
        peak_value  = config.learning_rate,
        warmup_steps= config.warmup_steps,
        decay_steps = config.n_epochs * 312,   # ~steps per epoch pour n=10000, bs=32
        end_value   = config.learning_rate * 0.01,
    )
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule),
    )


def create_train_state(config: ModelConfig) -> TrainState:
    """Initialise tous les poids et l'optimizer. (R5 — seed unique)"""
    key = jax.random.PRNGKey(config.seed)
    key, sk_enc, sk_pred, sk_pc = jax.random.split(key, 4)

    enc_w  = init_encoder(sk_enc,  config)
    pred_w = init_predictor(sk_pred, config)
    pc_w   = init_pc_weights(sk_pc, config)
    pc_flat = pc_weights_to_flat_dict(pc_w)

    # Les poids à optimiser : encodeur + predictor + pc (hiérarchie)
    all_weights = {'enc': enc_w, 'pred': pred_w, 'pc': pc_flat}

    optimizer = _build_optimizer(config)
    opt_state = optimizer.init(all_weights)

    return TrainState(
        encoder_weights   = enc_w,
        predictor_weights = pred_w,
        pc_weights_flat   = pc_flat,
        optimizer_state   = opt_state,
        step              = 0,
        key               = key,
    )


# ---------------------------------------------------------------------------
# train_step — JIT compilé
# ---------------------------------------------------------------------------

def make_train_step(config: ModelConfig, optimizer: optax.GradientTransformation):
    """
    Factory qui retourne un train_step JIT-compilé.
    On sépare la construction de l'optimizer pour éviter de le capturer en closure.
    """

    @jax.jit
    def train_step(state: TrainState, batch: Batch) -> tuple:
        """
        Un pas complet d'entraînement.

        Ordre des opérations (IMMUABLE) :
        1. Encoder les frames de contexte  : z_context = f_θ(batch.context)
        2. Encoder les frames cibles avec STOP_GRADIENT (R3)
        3. Boucle 1 — inférence PC sur z_context → pc_state_converged
        4. Predictor : z_pred = g_φ(z_context)
        5. Calcul des pertes
        6. Boucle 2 — mise à jour des poids via optax
        7. Logging — retourner les métriques

        Returns: (new_state, metrics_dict)
        """
        enc_w  = state.encoder_weights
        pred_w = state.predictor_weights
        pc_w   = flat_dict_to_pc_weights(state.pc_weights_flat, config)

        def loss_fn(all_weights):
            enc_w_  = all_weights['enc']
            pred_w_ = all_weights['pred']
            pc_flat_ = all_weights['pc']
            pc_w_   = flat_dict_to_pc_weights(pc_flat_, config)

            # Étape 1 — Encodage contexte
            z_context = apply_encoder(enc_w_, batch.context)   # (B, T_in, d_z)

            # Étape 2 — Encodage cible avec STOP GRADIENT (R3)
            z_target = jax.lax.stop_gradient(
                apply_encoder(enc_w_, batch.target)            # (B, T_pred, d_z)
            )

            # Étape 3 — Boucle 1 : inférence PC (poids figés, états libres)
            init_pc_state = init_pc_from_encoding(
                z_context[:, -1, :], pc_w_, config
            )
            pc_converged, T_conv, final_err = run_inference_loop(
                init_pc_state, pc_w_, z_context[:, -1, :], config
            )

            # Étape 4 — Prédiction dans l'espace latent
            z_pred = apply_predictor(pred_w_, z_context)   # (B, K, d_z)

            # Aligner les horizons (T_pred peut être > K)
            K = config.pred_K
            z_target_k = z_target[:, :K, :]               # (B, K, d_z)

            # Étape 5 — Pertes
            loss, loss_details = total_loss(
                all_weights, pc_converged, z_pred, z_target_k, config
            )

            aux = {**loss_details, 'T_conv': T_conv, 'pc_error': final_err}
            return loss, aux

        all_weights = {
            'enc':  enc_w,
            'pred': pred_w,
            'pc':   state.pc_weights_flat,
        }

        # Étape 5+6 — Calcul gradients + mise à jour (Boucle 2)
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(all_weights)

        updates, new_opt_state = optimizer.update(
            grads, state.optimizer_state, all_weights
        )
        new_weights = optax.apply_updates(all_weights, updates)

        new_state = TrainState(
            encoder_weights   = new_weights['enc'],
            predictor_weights = new_weights['pred'],
            pc_weights_flat   = new_weights['pc'],
            optimizer_state   = new_opt_state,
            step              = state.step + 1,
            key               = state.key,
        )
        return new_state, metrics

    return train_step


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def evaluate(
    state: TrainState,
    data_iter: Iterator[Batch],
    config: ModelConfig,
    n_batches: int = 50,
) -> dict:
    """
    Évalue sur n_batches batches.
    Retourne les métriques moyennées.
    """
    from eval.metrics import compute_all_metrics

    enc_w  = state.encoder_weights
    pred_w = state.predictor_weights
    pc_w   = flat_dict_to_pc_weights(state.pc_weights_flat, config)

    all_metrics = []
    for i, batch in enumerate(data_iter()):
        if i >= n_batches:
            break

        z_context = apply_encoder(enc_w, batch.context)
        z_target  = apply_encoder(enc_w, batch.target)

        init_pc_state = init_pc_from_encoding(z_context[:, -1, :], pc_w, config)
        pc_converged, T_conv, _ = run_inference_loop(
            init_pc_state, pc_w, z_context[:, -1, :], config
        )

        z_pred = apply_predictor(pred_w, z_context)
        z_target_k = z_target[:, :config.pred_K, :]

        metrics = compute_all_metrics(z_pred, z_target_k, pc_converged, T_conv)
        all_metrics.append(metrics)

    # Moyenne sur les batches
    averaged = {}
    for key in all_metrics[0]:
        averaged[key] = float(jnp.mean(jnp.stack([m[key] for m in all_metrics])))
    return averaged


# ---------------------------------------------------------------------------
# Boucle d'entraînement principale
# ---------------------------------------------------------------------------

def train(
    config: ModelConfig,
    train_loader_fn,
    val_loader_fn,
    n_steps: int,
    log_every: int = 100,
    eval_every: int = 500,
) -> tuple:
    """
    Boucle d'entraînement principale.

    train_loader_fn : callable () → Iterator[Batch]
    val_loader_fn   : callable () → Iterator[Batch]
    n_steps         : nombre total de pas d'entraînement
    Returns: (final_state, history_dict)
    """
    state    = create_train_state(config)
    optimizer = _build_optimizer(config)
    train_step = make_train_step(config, optimizer)

    history = {'train_loss': [], 'val_loss': [], 'T_conv': []}
    train_iter = train_loader_fn()

    for step in range(n_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = train_loader_fn()
            batch = next(train_iter)

        # Convertir en jnp arrays
        batch_jnp = Batch(
            context = jnp.array(batch.context),
            target  = jnp.array(batch.target),
        )

        state, metrics = train_step(state, batch_jnp)

        if step % log_every == 0:
            history['train_loss'].append(float(metrics['loss_total']))
            history['T_conv'].append(float(metrics['T_conv']))
            print(
                f"  step={step:5d}  "
                f"loss={float(metrics['loss_total']):.4f}  "
                f"jepa={float(metrics['loss_jepa']):.4f}  "
                f"pc={float(metrics['loss_pc']):.4f}  "
                f"var={float(metrics['loss_var']):.4f}  "
                f"T_conv={int(metrics['T_conv']):3d}"
            )

        if step % eval_every == 0 and step > 0:
            val_metrics = evaluate(state, val_loader_fn, config)
            history['val_loss'].append(val_metrics.get('nmse', 0.0))
            print(f"  [Eval step={step}] nmse={val_metrics.get('nmse', 0.0):.4f}")

    return state, history
