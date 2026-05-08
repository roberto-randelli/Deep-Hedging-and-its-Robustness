# Deep Hedging and its Robustness

MSc Mathematical Finance dissertation — University of Oxford, 2026.
**Author**: Roberto Randelli

---

## Overview

This repository contains two distinct bodies of work:

1. **Phase 1 — Replication** of Buehler, Gonon, Teichmann, and Wood (2018) Deep Hedging and He, Sutter & Gonon (2025). Code is adapetd to work with MPS on MacOS.
2. **Phase 2 — Novel Extensions**:

---

## Repository Structure

```
.
├── ATTRIBUTION.md          # Credits and licensing for the cloned replication code
│
├── data/                   # data generated through src code
│
├── src/                    # MY CODE — written entirely from scratch
│   ├── models/             # GBM, Heston, Merton, Regime-switching simulators
│   ├── hedging/            # Neural network, OCE loss, vanilla + W-DRO training loops
│   ├── attacks/            # Distributional FGSM and PGD (reimplemented from paper equations)
│   ├── experiments/        # Cross-model evaluation, ablation studies, mechanistic analysis
│   └── utils/              # Config, plotting, metrics
│
├── notebooks_outputs/
│   ├── replication/        # Phase 1 results (He et al. code outputs)
│   ├── extension/          # Phase 2 results (my novel experiments)
│   └── notebooks/              # Exploratory Jupyter notebooks
└── 
```

---

## Phase 1 — Replication

**Paper**: Buehler, L., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep Hedging. *Quantitative Finance*, 19(8), 1271–1291.

**Paper**: He, G., Sutter, T., & Gonon, L. (2025). *Distributional Adversarial Attacks and Training in Deep Hedging*. arXiv:2508.14757v2. NeurIPS 2025.

**Code source**: `Distributional-Adversarial-Attacks-and-Training-in-Deep-Hedging` is the official repository. The code is used as reference to replicate the results in the papers.


---

## Phase 2 — Novel Extensions


---

## Key References

- Buehler, L., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep Hedging. *Quantitative Finance*, 19(8), 1271–1291.
- He, G., Sutter, T., & Gonon, L. (2025). Distributional Adversarial Attacks and Training in Deep Hedging. arXiv:2508.14757v2.

## Rules

- **Always ask clarifying questions before starting a complex task.** Do not assume intent.
- **Show your plan and all steps before executing any code.** Wait for confirmation on plan before proceeding.
- **Keep reports and summaries concise** — bullet points over paragraphs unless I ask otherwise.
- **At the end of each task, explain step-by-step what you did.** This explanation is critical: I will use it to write my dissertation methodology section.
- **Cite sources** when doing research or referencing equations (use paper section/equation numbers).
- **If something fails**, explain the error clearly and propose a fix before retrying.
- **Never delete or overwrite existing results** without confirming first.
