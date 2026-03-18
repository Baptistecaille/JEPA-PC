"""
=== SPEC [visualize] ===
Entrées  : dicts de résultats, arrays numpy de représentations latentes
Sorties  : figures matplotlib (affichées dans Colab ou sauvegardées)
Invariants:
  - Pas de dépendances JAX (pure numpy/matplotlib)
  - Sauvegarde optionnelle en PDF/PNG
  - Compatible Colab (inline display)
=== FIN SPEC ===
"""
import numpy as np
import os


def _get_plt():
    """Import matplotlib lazily pour éviter les erreurs sur serveurs sans display."""
    import matplotlib
    matplotlib.use('Agg')   # backend non-interactif par défaut
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# Courbes de loss
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: dict,
    title: str = "Courbes d'entraînement",
    save_path: str = None,
) -> None:
    """
    Trace les courbes de loss au fil des steps.
    history : dict avec clés 'train_loss', 'val_loss', 'T_conv'
    """
    plt = _get_plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if 'train_loss' in history and history['train_loss']:
        axes[0].plot(history['train_loss'], label='Train loss')
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(
            np.linspace(0, len(history.get('train_loss', [1])), len(history['val_loss'])),
            history['val_loss'], label='Val NMSE', linestyle='--'
        )
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss / NMSE')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].set_yscale('log')

    if 'T_conv' in history and history['T_conv']:
        axes[1].plot(history['T_conv'], color='orange')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('T_conv')
        axes[1].set_title('Convergence PC')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Courbe sample efficiency (exp2)
# ---------------------------------------------------------------------------

def plot_efficiency_curves(
    results: dict,
    efficiency_ns: tuple,
    seeds: tuple,
    save_path: str = None,
) -> None:
    """
    Courbe principale NMSE(n) pour PC-JEPA et Transformer.
    results : dict avec clés f'{model_type}_n{n}_seed{seed}'
    """
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_type, color, label in [
        ('pc_jepa',     'steelblue', 'PC-JEPA'),
        ('transformer', 'tomato',    'Transformer Baseline'),
    ]:
        means, stds = [], []
        for n in efficiency_ns:
            vals = [
                results.get(f'{model_type}_n{n}_seed{s}', {}).get('nmse', float('nan'))
                for s in seeds
            ]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(float('nan'))
                stds.append(0.0)

        means = np.array(means)
        stds  = np.array(stds)
        ax.plot(efficiency_ns, means, 'o-', color=color, label=label, linewidth=2)
        ax.fill_between(
            efficiency_ns,
            means - stds, means + stds,
            alpha=0.2, color=color,
        )

    ax.set_xlabel('Nombre d\'exemples d\'entraînement (n)', fontsize=12)
    ax.set_ylabel('NMSE (test)', fontsize=12)
    ax.set_title('Sample Efficiency — PC-JEPA vs Transformer', fontsize=13)
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# PCA 2D de l'espace latent
# ---------------------------------------------------------------------------

def plot_latent_pca(
    z: np.ndarray,
    labels: np.ndarray = None,
    title: str = "PCA espace latent",
    save_path: str = None,
    n_components: int = 2,
) -> None:
    """
    Visualisation PCA 2D des représentations latentes.
    z      : (N, d_z) numpy array
    labels : (N,) entiers optionnels (ex : classe chiffre) pour la couleur
    """
    plt = _get_plt()

    # PCA manuelle via SVD (pas de sklearn requis)
    z_centered = z - z.mean(axis=0)
    U, S, Vt = np.linalg.svd(z_centered, full_matrices=False)
    z_pca = z_centered @ Vt[:n_components].T   # (N, 2)

    var_explained = S[:n_components] ** 2 / (S ** 2).sum()

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_kwargs = {'s': 5, 'alpha': 0.5}
    if labels is not None:
        sc = ax.scatter(z_pca[:, 0], z_pca[:, 1], c=labels,
                        cmap='tab10', **scatter_kwargs)
        plt.colorbar(sc, ax=ax, label='Classe')
    else:
        ax.scatter(z_pca[:, 0], z_pca[:, 1], **scatter_kwargs)

    ax.set_xlabel(f'PC1 ({100*var_explained[0]:.1f}% var)', fontsize=11)
    ax.set_ylabel(f'PC2 ({100*var_explained[1]:.1f}% var)', fontsize=11)
    ax.set_title(title, fontsize=12)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Distribution des valeurs propres de la covariance
# ---------------------------------------------------------------------------

def plot_eigenvalue_spectrum(
    eigenvalues: np.ndarray,
    title: str = "Spectre de la covariance latente",
    save_path: str = None,
) -> None:
    """
    Trace la distribution des valeurs propres (décroissant).
    Un espace latent non-collapsed a une distribution étalée.
    """
    plt = _get_plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sorted_ev = np.sort(eigenvalues)[::-1]
    cumvar = np.cumsum(sorted_ev) / sorted_ev.sum()

    axes[0].bar(range(min(64, len(sorted_ev))), sorted_ev[:64], color='steelblue')
    axes[0].set_xlabel('Composante principale')
    axes[0].set_ylabel('Valeur propre')
    axes[0].set_title('Spectre (top 64)')

    axes[1].plot(cumvar * 100, color='steelblue')
    axes[1].axhline(90, color='red', linestyle='--', label='90% variance')
    axes[1].set_xlabel('Nombre de composantes')
    axes[1].set_ylabel('Variance expliquée cumulée (%)')
    axes[1].set_title('Variance cumulée')
    axes[1].legend()
    axes[1].set_title(title)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Résumé ablations
# ---------------------------------------------------------------------------

def plot_ablation_bars(
    ablation_results: dict,
    metric: str = 'nmse',
    save_path: str = None,
) -> None:
    """Barplot de comparaison des ablations."""
    plt = _get_plt()

    names  = list(ablation_results.keys())
    values = [ablation_results[n].get(metric, float('nan')) for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['steelblue' if n == 'full_model' else 'lightcoral' for n in names]
    ax.bar(names, values, color=colors)
    ax.set_xlabel('Ablation')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'Ablations PC-JEPA — {metric.upper()}')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
