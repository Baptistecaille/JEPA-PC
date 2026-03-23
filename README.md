# PC-JEPA — Predictive Coding + Joint Embedding Predictive Architecture

Implémentation JAX d'un modèle PC-JEPA pour Moving MNIST. Le modèle combine une hiérarchie de Predictive Coding (inférence variationnelle à deux boucles) avec un prédicteur Transformer pour apprendre des représentations vidéo par auto-supervision.

**Hypothèse principale :** la boucle d'inférence PC impose une contrainte de cohérence hiérarchique qui améliore l'efficacité en données (sample efficiency) par rapport à un Transformer seul, en particulier dans le régime peu de données (n < 1000 exemples).

---

## Architecture

```
Frames (B, T, 64, 64, 1)
        │
        ▼
  CNN Encoder f_θ           3 couches conv → linear → LayerNorm → (B, T, 256)
        │
        ├── z_context (B, T_in, 256)
        │         │
        │         ├── obs = z_context[:, -1, :]
        │         │         │
        │         │    ┌────▼──────────────────────────────────┐
        │         │    │  PC Hierarchy (Boucle 1 — inférence)  │
        │         │    │  L=3 niveaux, x ← x - α·∂F/∂x        │
        │         │    │  Convergence : MSE < pc_tol            │
        │         │    └────────────────┬──────────────────────┘
        │         │                     │ T_conv, x* (stop_gradient)
        │         │
        │         └──► Transformer Predictor g_φ  →  z_pred (B, K, 256)
        │
        └── z_target (B, T_pred, 256)  [stop_gradient — R3]

Loss = L_JEPA + λ_pc · F(x*, W) + λ_var · L_var
     └─ pc_loss_hybrid(z_pred - z_target, PrecisionParams, α)
```

### Precision Module (normalisation divisive)

Inspiré des interneurones PV de L2/3 (König & Negrello, 2026) :

```
e_norm[i] = e[i] / (ε + Σ_j W_inhib[i,j] · |e[j]|)
```

Curriculum : `α = 0 → 1` sur les 30 premiers % du budget. Les paramètres `W_inhib` (256×256) et `log_ε` sont entraînés par descente de gradient avec contrainte `enforce_pv_constraints` (positivité, diag=0, normalisation par ligne) après chaque update.

---

## Structure des fichiers

```
JEPA-PC/
├── config.py                    # ModelConfig — tous les hyperparamètres (R7)
├── run_all.py                   # Point d'entrée Colab (--mode sanity|exp1|exp2|...)
├── check_tconv.py               # Diagnostic : T_conv est-il variable ?
│
├── models/
│   ├── encoder.py               # CNN encoder : (B,T,H,W,1) → (B,T,d_z)
│   ├── pc_nodes.py              # Hiérarchie PC : boucle d'inférence (while_loop JIT)
│   ├── predictor.py             # Transformer + MLP multi-horizon : → (B,K,d_z)
│   └── transformer_baseline.py  # Baseline Transformer seul (même budget de params)
│
├── precision/
│   ├── module.py                # divisive_normalize, PrecisionParams, enforce_pv_constraints
│   ├── losses.py                # pc_loss_standard / pc_loss_divisive / pc_loss_hybrid
│   └── tests/test_module.py     # 6 tests unitaires (gradient, JIT, vmap, contraintes PV)
│
├── training/
│   ├── trainer.py               # TrainState, train_step (JIT), evaluate, train
│   └── losses.py                # loss_jepa, loss_variance
│
├── data/
│   └── moving_mnist.py          # Générateur procédural Moving MNIST (64×64, 2 chiffres)
│
├── eval/
│   ├── metrics.py               # nmse, collapse_score, compute_all_metrics
│   └── visualize.py             # Courbes d'efficacité, barres d'ablation
│
└── experiments/
    ├── exp1_full_data.py        # Qualité absolue (n=10000)
    ├── exp2_sample_efficiency.py # Courbe NMSE(n) — expérience principale
    ├── exp3_ablations.py        # Ablations composantes (λ_pc, λ_var, pc_n_layers)
    └── exp4_latent_analysis.py  # Visualisation espace latent
```

---

## Démarrage rapide (Google Colab)

```python
# Cellule 1 — Setup
!git clone <repo> && cd JEPA-PC
!pip install -q optax

# Cellule 2 — Sanity checks (< 2 min)
!python run_all.py --mode sanity

# Cellule 3 — Expérience principale
!python run_all.py --mode exp2

# Ou tout enchaîner
!python run_all.py --mode all
```

---

## Hyperparamètres clés

| Paramètre | Valeur | Description |
|---|---|---|
| `d_z` | 256 | Dimension latente |
| `pc_n_layers` | 3 | Profondeur hiérarchie PC |
| `pc_alpha` | 0.1 | Pas d'inférence (stable : α < 2/‖H‖₂ ≈ 0.22) |
| `pc_tol` | 0.15 | Seuil MSE de convergence (atteignable en ~82 pas) |
| `pc_max_iter` | 100 | Itérations max par boucle d'inférence |
| `prec_alpha` | 0.0 | Défaut curriculum (géré dynamiquement dans train_step) |
| `lambda_pc` | 0.1 | Poids de la perte PC |
| `lambda_var` | 0.01 | Poids anti-collapse (sur z_pred + z_context) |
| `learning_rate` | 3e-4 | Pic LR (Adam + cosine decay proportionnel au budget) |

---

## Design

**Deux boucles explicitement séparées (R2) :**

- **Boucle 1 (inférence)** : poids figés, états `x` libres. Minimise `F = Σ_l ‖ε^l‖²` par gradient descent. Implémentée avec `jax.lax.while_loop` (JIT-compatible). Résultat `x*` passé avec `stop_gradient` à la Boucle 2.
- **Boucle 2 (apprentissage)** : poids libres, états figés. Gradient des poids PC via `F(sg(x*), W, sg(obs))` — règle Hebbienne. Gradient du prédicteur et de `PrecisionParams` via `pc_loss_hybrid`.

**Autres invariants :**
- `z_target` toujours `stop_gradient` avant le calcul des pertes (R3)
- Tous les états sont des `NamedTuple` (pytrees JAX natifs, R4)
- Seed unique propagé via `jax.random.PRNGKey` (R5)
- Zéro magic number — tout dans `config.py` (R7)

---

## Résultats attendus (exp2)

Après correction des 3 bugs de divergence (warmup fixe, loss scale-invariante, collapse encodeur) :

| n | PC-JEPA NMSE | Transformer NMSE |
|---|---|---|
| 100 | ~15–25 | ~20–35 |
| 500 | ~4–7 | ~8–15 |
| 1000 | ~2–4 | ~5–10 |
| 10000 | ~0.5–1.5 | ~1–3 |

T_conv doit décroître avec n (plus de données → meilleure convergence PC) et avec le nombre de steps d'entraînement.
