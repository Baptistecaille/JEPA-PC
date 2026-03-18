"""
Point d'entrée unique pour Google Colab.

Utilisation :
  Cellule 1 (setup) :
    !pip install -q optax equinox
    !python run_all.py --mode sanity

  Cellule 2 (expérience principale) :
    !python run_all.py --mode exp2

  Cellule 3 (tout enchaîner) :
    !python run_all.py --mode all
"""
import argparse
import sys
import os

# Ajouter le répertoire courant au path (nécessaire dans Colab)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description='PC-JEPA — Point d\'entrée Colab')
    parser.add_argument(
        '--mode',
        choices=['sanity', 'exp1', 'exp2', 'exp3', 'exp4', 'all'],
        default='sanity',
        help='Mode d\'exécution',
    )
    parser.add_argument('--seed', type=int, default=42, help='Seed global')
    args = parser.parse_args()

    from config import ModelConfig
    from data.moving_mnist import DataConfig, run_sanity_checks

    data_config  = DataConfig(seed=args.seed)
    model_config = ModelConfig(seed=args.seed)

    # -------------------------------------------------------------------------
    # Sanity checks — toujours exécutés en premier
    # -------------------------------------------------------------------------
    if args.mode in ('sanity', 'all'):
        print("\n" + "=" * 60)
        print("SANITY CHECKS — vérification de tous les modules")
        print("=" * 60 + "\n")

        # Étape 1 — Data
        run_sanity_checks(data_config)

        # Étape 2 — Encoder
        from models.encoder import run_sanity_checks_encoder
        run_sanity_checks_encoder(model_config)

        # Étape 3 — PC Nodes
        from models.pc_nodes import run_sanity_checks_pc
        run_sanity_checks_pc(model_config)

        # Étape 4 — Predictor
        from models.predictor import run_sanity_checks_predictor
        run_sanity_checks_predictor(model_config)

        # Étape 5 — Parité des paramètres Transformer vs PC-JEPA
        from models.predictor import init_predictor, count_predictor_params
        from models.transformer_baseline import (
            init_transformer_predictor, check_parameter_parity
        )
        import jax
        key = jax.random.PRNGKey(model_config.seed)
        key, sk1, sk2 = jax.random.split(key, 3)
        pred_w  = init_predictor(sk1, model_config)
        trans_w = init_transformer_predictor(sk2, model_config)
        print("[Sanity] Parité des paramètres Transformer / PC-JEPA predictor")
        check_parameter_parity(pred_w, trans_w)
        print()

        print("\n" + "=" * 60)
        print("✅ Tous les sanity checks passés. Prêt pour les expériences.")
        print("=" * 60 + "\n")

    # -------------------------------------------------------------------------
    # Expérience 1 — Qualité absolue
    # -------------------------------------------------------------------------
    if args.mode in ('exp1', 'all'):
        from experiments.exp1_full_data import run_exp1
        results1 = run_exp1(model_config, data_config)

    # -------------------------------------------------------------------------
    # Expérience 2 — Sample Efficiency (expérience principale)
    # -------------------------------------------------------------------------
    if args.mode in ('exp2', 'all'):
        from experiments.exp2_sample_efficiency import run_exp2
        results2 = run_exp2(model_config, data_config)

        # Visualisation
        from experiments.exp2_sample_efficiency import EFFICIENCY_NS, SEEDS
        from eval.visualize import plot_efficiency_curves
        plot_efficiency_curves(
            results2,
            efficiency_ns=EFFICIENCY_NS,
            seeds=SEEDS,
            save_path='results/exp2_efficiency_curves.png',
        )

    # -------------------------------------------------------------------------
    # Expérience 3 — Ablations
    # -------------------------------------------------------------------------
    if args.mode in ('exp3', 'all'):
        from experiments.exp3_ablations import run_exp3
        results3 = run_exp3(model_config, data_config)

        from eval.visualize import plot_ablation_bars
        plot_ablation_bars(results3, save_path='results/exp3_ablations.png')

    # -------------------------------------------------------------------------
    # Expérience 4 — Analyse latente (nécessite un state entraîné)
    # -------------------------------------------------------------------------
    if args.mode == 'exp4':
        print("[Exp4] Nécessite un state pré-entraîné (depuis exp1 ou exp2).")
        print("       Utiliser run_exp4(state, config) depuis un notebook.")

    if args.mode == 'all':
        print("\n✅ Toutes les expériences terminées.")


if __name__ == '__main__':
    main()
