"""
=== SPEC [moving_mnist] ===
Entrées  : Configuration DataConfig (seed, split sizes, séquence lengths)
Sorties  : Itérateurs de batches Batch(context, target)
           context : (B, T_in, H, W, 1) float32 in [0,1]
           target  : (B, T_pred, H, W, 1) float32 in [0,1]
Invariants:
  - Reproductibilité totale via seed (R5)
  - Pas de fuite entre train/val/test
  - Frames normalisées dans [0,1]
  - Compatible JAX (arrays numpy convertibles en jnp)
=== FIN SPEC ===
"""
from typing import NamedTuple, Iterator
import numpy as np


class DataConfig(NamedTuple):
    # Dimensions spatiales
    img_size: int  = 64      # H = W = 64 (Moving MNIST standard)
    n_digits: int  = 2       # nombre de chiffres par séquence

    # Découpage temporel
    T_in:   int = 10         # frames de contexte
    T_pred: int = 5          # frames à prédire (= pred_K)

    # Tailles des splits
    n_train: int = 10000
    n_val:   int = 1000
    n_test:  int = 1000

    # Entraînement
    batch_size: int = 32

    # Reproductibilité (R5)
    seed: int = 42


class Batch(NamedTuple):
    context: np.ndarray   # (B, T_in,   H, W, 1) float32
    target:  np.ndarray   # (B, T_pred, H, W, 1) float32


# ---------------------------------------------------------------------------
# Génération procédurale de Moving MNIST
# ---------------------------------------------------------------------------

# Cache module-level : MNIST chargé une seule fois par processus
_MNIST_DIGITS_CACHE: np.ndarray | None = None


def _load_mnist_digits() -> np.ndarray:
    """
    Retourne les chiffres MNIST.
    Ordre de priorité :
      1. Cache mémoire (même processus)
      2. ~/.keras/datasets/mnist.npz (téléchargé par keras, chargé via numpy pur)
      3. tensorflow.keras.datasets.mnist
      4. torchvision MNIST
      5. Placeholders synthétiques (CI / pas d'internet)
    Shape : (N, 28, 28) uint8

    N'avance JAMAIS le rng de l'appelant : la graine des placeholders est fixe (0),
    ce qui garantit la reproductibilité de D4 quelle que soit la fréquence d'appel.
    """
    global _MNIST_DIGITS_CACHE
    if _MNIST_DIGITS_CACHE is not None:
        return _MNIST_DIGITS_CACHE

    import os

    # Essai 1 : fichier keras déjà sur disque → numpy pur, pas de TF requis
    keras_path = os.path.expanduser('~/.keras/datasets/mnist.npz')
    if os.path.exists(keras_path):
        try:
            data   = np.load(keras_path)
            digits = np.concatenate([data['x_train'], data['x_test']], axis=0)
            _MNIST_DIGITS_CACHE = digits.astype(np.uint8)
            return _MNIST_DIGITS_CACHE
        except Exception:
            pass   # fichier corrompu → essais suivants

    # Essai 2 : TensorFlow (télécharge et met en cache dans ~/.keras/)
    try:
        import tensorflow as tf
        (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
        digits = np.concatenate([x_train, x_test], axis=0)
    except Exception:
        # Essai 3 : torchvision
        try:
            from torchvision import datasets as tvd
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                ds = tvd.MNIST(root=tmp, train=True, download=True)
                digits = ds.data.numpy()   # (60000, 28, 28)
        except Exception:
            # Fallback : carrés synthétiques (CI / sans internet)
            # Graine fixe (0) : indépendant du rng de l'appelant → reproductible
            print("[WARNING] MNIST non disponible — utilisation de placeholders synthétiques.")
            digits = np.random.default_rng(0).integers(
                0, 255, size=(1000, 28, 28), dtype=np.uint8
            )

    _MNIST_DIGITS_CACHE = digits.astype(np.uint8)
    return _MNIST_DIGITS_CACHE


def _generate_sequence(
    digits_pool: np.ndarray,
    n_digits: int,
    img_size: int,
    T_total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Génère une séquence Moving MNIST de longueur T_total.
    Retourne : (T_total, img_size, img_size) float32 in [0, 1]
    """
    canvas = np.zeros((T_total, img_size, img_size), dtype=np.float32)
    digit_size = 28

    for _ in range(n_digits):
        idx = rng.integers(0, len(digits_pool))
        digit = digits_pool[idx].astype(np.float32) / 255.0  # (28, 28)

        # Position initiale aléatoire
        x = rng.integers(0, img_size - digit_size)
        y = rng.integers(0, img_size - digit_size)

        # Vitesse aléatoire en pixels/frame
        vx = rng.integers(-3, 4)
        vy = rng.integers(-3, 4)
        if vx == 0 and vy == 0:
            vx = 1

        for t in range(T_total):
            x_t = int(x + vx * t) % (img_size - digit_size)
            y_t = int(y + vy * t) % (img_size - digit_size)
            canvas[t, y_t:y_t + digit_size, x_t:x_t + digit_size] = np.clip(
                canvas[t, y_t:y_t + digit_size, x_t:x_t + digit_size] + digit, 0, 1
            )

    return canvas  # (T, H, W)


def _build_dataset(
    digits_pool: np.ndarray,
    config: DataConfig,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Génère n_samples séquences.
    Retourne : (n_samples, T_total, H, W, 1) float32
    """
    T_total = config.T_in + config.T_pred
    seqs = np.stack([
        _generate_sequence(digits_pool, config.n_digits, config.img_size, T_total, rng)
        for _ in range(n_samples)
    ], axis=0)  # (N, T, H, W)
    return seqs[:, :, :, :, np.newaxis]  # (N, T, H, W, 1)


# ---------------------------------------------------------------------------
# Interface publique
# ---------------------------------------------------------------------------

def get_dataloaders(config: DataConfig) -> tuple:
    """
    Retourne (train_iter, val_iter, test_iter).
    Chaque itérateur est une fonction () -> Iterator[Batch].
    """
    rng = np.random.default_rng(config.seed)
    digits_pool = _load_mnist_digits()

    train_data = _build_dataset(digits_pool, config, config.n_train, rng)
    val_data   = _build_dataset(digits_pool, config, config.n_val,   rng)
    test_data  = _build_dataset(digits_pool, config, config.n_test,  rng)

    def make_iter(data: np.ndarray, shuffle: bool = True) -> Iterator[Batch]:
        rng_iter = np.random.default_rng(config.seed)
        while True:
            idx = np.arange(len(data))
            if shuffle:
                rng_iter.shuffle(idx)
            for start in range(0, len(data), config.batch_size):
                batch_idx = idx[start:start + config.batch_size]
                if len(batch_idx) < config.batch_size:
                    continue   # drop last pour garder des batches uniformes
                seqs = data[batch_idx]   # (B, T_total, H, W, 1)
                yield Batch(
                    context = seqs[:, :config.T_in,  :, :, :],
                    target  = seqs[:, config.T_in:,  :, :, :],
                )

    return (
        lambda: make_iter(train_data, shuffle=True),
        lambda: make_iter(val_data,   shuffle=False),
        lambda: make_iter(test_data,  shuffle=False),
    )


def get_subset_dataloader(
    config: DataConfig,
    n_samples: int,
    seed: int,
) -> Iterator[Batch]:
    """
    Crée un loader d'entraînement sur n_samples exemples (pour exp2).
    Seed distinct pour reproductibilité inter-expériences.
    """
    rng = np.random.default_rng(seed)
    digits_pool = _load_mnist_digits()
    data = _build_dataset(digits_pool, config, n_samples, rng)

    rng_iter = np.random.default_rng(seed)

    def _iter() -> Iterator[Batch]:
        while True:
            idx = np.arange(len(data))
            rng_iter.shuffle(idx)
            for start in range(0, len(data), config.batch_size):
                batch_idx = idx[start:start + config.batch_size]
                if len(batch_idx) < config.batch_size:
                    continue
                seqs = data[batch_idx]
                yield Batch(
                    context = seqs[:, :config.T_in,  :, :, :],
                    target  = seqs[:, config.T_in:,  :, :, :],
                )

    return _iter


def run_sanity_checks(config: DataConfig) -> None:
    """
    R6 — Sanity checks du module data.
    Vérifie :
      [D1] Les batches ont les bonnes shapes
      [D2] Les valeurs sont dans [0, 1]
      [D3] context et target sont distincts (pas d'overlap)
      [D4] Reproductibilité : deux loaders avec le même seed produisent les mêmes données
    """
    print("[Sanity] data.moving_mnist — début")

    # Config minimale pour les tests (rapide)
    test_cfg = DataConfig(n_train=10, n_val=5, n_test=5, batch_size=4)
    train_fn, val_fn, test_fn = get_dataloaders(test_cfg)

    # [D1] Shapes
    batch = next(train_fn())
    B, T_in, H, W, C = batch.context.shape
    assert B  == test_cfg.batch_size,    f"[D1] batch_size: {B} ≠ {test_cfg.batch_size}"
    assert T_in == test_cfg.T_in,        f"[D1] T_in: {T_in} ≠ {test_cfg.T_in}"
    assert H  == test_cfg.img_size,      f"[D1] H: {H} ≠ {test_cfg.img_size}"
    assert W  == test_cfg.img_size,      f"[D1] W: {W} ≠ {test_cfg.img_size}"
    assert C  == 1,                      f"[D1] C: {C} ≠ 1"
    assert batch.target.shape == (B, test_cfg.T_pred, H, W, C), \
        f"[D1] target shape: {batch.target.shape}"
    print(f"  [D1] ✓ context: {batch.context.shape}, target: {batch.target.shape}")

    # [D2] Valeurs dans [0, 1]
    assert batch.context.min() >= 0.0, f"[D2] context min < 0: {batch.context.min()}"
    assert batch.context.max() <= 1.0, f"[D2] context max > 1: {batch.context.max()}"
    assert batch.target.min()  >= 0.0, f"[D2] target min < 0"
    assert batch.target.max()  <= 1.0, f"[D2] target max > 1"
    print("  [D2] ✓ valeurs dans [0, 1]")

    # [D3] context ≠ target — frames temporellement distinctes
    # On compare la première frame du contexte vs la première frame de la cible
    assert not np.allclose(batch.context[:, 0], batch.target[:, 0]), \
        "[D3] context et target semblent identiques"
    print("  [D3] ✓ context et target temporellement distincts")

    # [D4] Reproductibilité
    train_fn2, _, _ = get_dataloaders(test_cfg)
    batch2 = next(train_fn2())
    assert np.allclose(batch.context, batch2.context), "[D4] non-reproductible"
    print("  [D4] ✓ reproductibilité avec même seed")

    print("[Sanity] data.moving_mnist — OK\n")
