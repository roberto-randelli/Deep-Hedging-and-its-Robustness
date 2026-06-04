# Deep Hedging and its Robustness

MSc Mathematical Finance dissertation — University of Oxford, 2026.
**Author**: Roberto Randelli

---

## Overview

This repository contains two distinct bodies of work:

1. **Phase 1 — Replication** of Buehler, Gonon, Teichmann & Wood (2019) and He, Sutter & Gonon (2025). Code is adapted to run with MPS on macOS.
2. **Phase 2 — Novel Extensions**: robustness analysis of deep hedging under the Bates model (stochastic volatility with jumps), extending the adversarial training framework of He et al. to a richer market dynamics setting.

---

## Repository Structure

```
.
├── ATTRIBUTION.md                    # Credits and licensing
├── data/                             # Pre-generated simulation paths (.pt + .json)
│   ├── gbm_paths.*
│   ├── heston_paths.*
│   └── bates_paths.*
│
├── src/                              # All custom code, written from scratch
│   ├── gbm_simulator.py              # Black-Scholes (GBM) path simulator
│   ├── heston_simulator.py           # Heston stochastic volatility simulator
│   ├── bates_simulator.py            # Bates model (SV + jumps) simulator
│   ├── train_vanilla.py              # Vanilla deep hedging training (GBM)
│   ├── train_buehler_benchmark.py    # Buehler et al. benchmark replication
│   ├── train_heston.py               # Heston model training
│   ├── train_adv_heston.py           # Adversarial training on Heston (He et al.)
│   ├── bates_experiment_train.py     # Bates: clean / S-attack / SV-attack training
│   ├── bates_experiment_evaluate.py  # Bates: cross-model robustness evaluation
│   ├── bates_experiment_perf_vs_delta.py  # Bates: performance vs attack budget sweep
│   ├── models/                       # (reserved for model wrappers)
│   └── hedging/                      # Core hedging primitives
│       ├── hedge_network.py          # Recurrent hedging neural network
│       ├── loss.py                   # OCE / Expected Shortfall loss
│       ├── trainer.py                # Vanilla training loop
│       ├── heston_trainer.py         # Heston training loop
│       ├── adv_trainer.py            # Adversarial (W-DRO) training loop
│       ├── attacker.py               # Heston distributional attacker (FGSM / PGD)
│       ├── bates_network.py          # Bates hedging network (S + V + jump features)
│       ├── bates_trainer.py          # Bates training loop
│       ├── bates_attacker.py         # Bates distributional attacker
│       ├── bates_loss.py             # Bates OCE loss with variance-swap auxiliary
│       └── theoretical.py            # Analytical delta benchmarks
│
├── results/                          # Saved models, training logs, figures
│   ├── adv_nets/                     # Adversarial nets across N and seeds
│   ├── comparison/                   # Bates robustness comparison outputs
│   └── *.pt / *.png                  # Buehler & Heston replication artefacts
│
└── notebooks_outputs/                # Jupyter notebooks with rendered outputs
    ├── buehler_benchmark.ipynb
    ├── buehler_benchmark_heston.ipynb
    ├── vanilla_delta_comparison.ipynb
    ├── heston_table1_attacks.ipynb
    ├── adv_heston_analysis.ipynb
    ├── bates_visual_check.ipynb
    ├── bates_experiments.ipynb
    └── bates_robustness_comparison.ipynb
```

---

## Phase 1 — Replication

**Paper**: Buehler, L., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep Hedging. *Quantitative Finance*, 19(8), 1271–1291.

**Paper**: He, G., Sutter, T., & Gonon, L. (2025). *Distributional Adversarial Attacks and Training in Deep Hedging*. arXiv:2508.14757v2. NeurIPS 2025.

Replicated results include Buehler et al. Figures 3.1, 6, 7 and He et al. Table 1 (adversarial attack comparison under Heston dynamics). Reference code: the official `Distributional-Adversarial-Attacks-and-Training-in-Deep-Hedging` repository.

---

## Phase 2 — Novel Extensions

Extends the He et al. adversarial framework to the **Bates model** — a stochastic volatility model augmented with compound Poisson jumps. Key contributions:

- **Bates simulator** with Euler–Maruyama discretisation, log-normal jump sizes, and variance-swap price tracking.
- **Bates hedging network** taking stock price, variance, and variance-swap price as features.
- **Bates distributional attacker** perturbing drift, volatility-of-volatility, and jump intensity simultaneously.
- **Robustness comparison**: clean, S-attacked, and SV-attacked hedgers evaluated in-distribution (Bates) and out-of-distribution (clean Heston, attacked Heston).
- **Performance vs. attack budget sweep** characterising the clean/robust trade-off as a function of Wasserstein radius δ.

---

## Key References

- Buehler, L., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep Hedging. *Quantitative Finance*, 19(8), 1271–1291.
- He, G., Sutter, T., & Gonon, L. (2025). Distributional Adversarial Attacks and Training in Deep Hedging. arXiv:2508.14757v2.
