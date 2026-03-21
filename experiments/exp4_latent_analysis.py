"""
Expérience 4 — Analyse de l'espace latent
Objectif : Visualiser et quantifier la qualité de l'espace latent PC-JEPA.

Analyses :
  - PCA 2D des représentations latentes (colorées par classe chiffre)
  - Évolution du collapse_score au fil de l'entraînement
  - Distribution des valeurs propres de la matrice de covariance
"""
import jax.numpy as jnp
import numpy as np
from config import ModelConfig
from data.moving_mnist import DataConfig, get_dataloaders, Batch
from models.encoder import apply_encoder
from models.predictor import apply_predictor
from models.pc_nodes import init_pc_from_encoding, run_inference_loop, flat_dict_to_pc_weights
from eval.metrics import collapse_score, compute_per_horizon_nmse
from training.trainer import TrainState


def collect_latent_representations(
    state: TrainState,
    data_fn,
    config: ModelConfig,
    n_batches: int = 20,
) -> dict:
    """
    Collecte les représentations latentes z et z_pred sur n_batches.
    Retourne un dict avec les arrays numpy pour visualisation.
    """
    from models.pc_nodes import flat_dict_to_pc_weights

    enc_w  = state.encoder_weights
    pred_w = state.predictor_weights
    pc_w   = flat_dict_to_pc_weights(state.pc_weights_flat, config)

    all_z_context  = []
    all_z_pred     = []
    all_z_target   = []
    all_T_conv     = []

    for i, batch in enumerate(data_fn()):
        if i >= n_batches:
            break
        ctx = jnp.array(batch.context)
        tgt = jnp.array(batch.target)

        z_context = apply_encoder(enc_w, ctx)
        z_target  = apply_encoder(enc_w, tgt)

        init_state = init_pc_from_encoding(z_context[:, -1, :], pc_w, config)
        pc_conv, T_conv, _ = run_inference_loop(init_state, pc_w, z_context[:, -1, :], config)

        z_pred = apply_predictor(pred_w, z_context, config)

        all_z_context.append(np.array(z_context[:, -1, :]))   # dernière frame
        all_z_pred.append(np.array(z_pred[:, 0, :]))          # horizon 1
        all_z_target.append(np.array(z_target[:, 0, :]))      # horizon 1 cible
        all_T_conv.append(int(T_conv))

    return {
        'z_context': np.concatenate(all_z_context, axis=0),
        'z_pred':    np.concatenate(all_z_pred,    axis=0),
        'z_target':  np.concatenate(all_z_target,  axis=0),
        'T_conv':    np.array(all_T_conv),
    }


def run_exp4(
    state: TrainState,
    config: ModelConfig,
    data_config: DataConfig = None,
) -> dict:
    """
    Analyse complète de l'espace latent.
    Nécessite un state entraîné (depuis exp1 ou exp2).
    """
    if data_config is None:
        data_config = DataConfig(seed=config.seed)

    _, _, test_fn = get_dataloaders(data_config)

    print("=" * 60)
    print("[Exp4] Analyse de l'espace latent")
    print("=" * 60)

    latents = collect_latent_representations(state, test_fn, config)

    z_all = latents['z_context']
    cs = float(collapse_score(jnp.array(z_all)))
    print(f"  collapse_score (z_context) : {cs:.4f}")
    print(f"  T_conv moyen               : {latents['T_conv'].mean():.1f}")
    print(f"  T_conv max                 : {latents['T_conv'].max()}")

    # Rang effectif de la covariance
    z_centered = z_all - z_all.mean(axis=0)
    cov = z_centered.T @ z_centered / len(z_all)
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]   # tri décroissant
    cumvar = np.cumsum(eigenvalues) / eigenvalues.sum()
    rank_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    print(f"  Rang effectif (90% var)    : {rank_90}/{config.d_z}")

    return {
        'collapse_score':  cs,
        'T_conv_mean':     float(latents['T_conv'].mean()),
        'effective_rank':  rank_90,
        'latents':         latents,
        'eigenvalues':     eigenvalues,
    }
