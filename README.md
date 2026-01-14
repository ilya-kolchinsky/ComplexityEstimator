# Complexity-Based Router for Large Language Models

**Complexity-based routing** is a pragmatic way to cut inference cost and latency without sacrificing quality. Instead of sending every request to your strongest (and most expensive) LLM, you first **estimate the difficulty of the prompt**, then **route** it to an appropriate model tier (fast/cheap vs. strong/expensive).

This repository includes:
- A **trainable difficulty regressor** that maps a prompt to a **score in [0,1]** (higher = harder).
- A **simple routing API** for making routing decisions based on the estimated difficulty scores.
- A **demo app** to explore the idea interactively.
- An **evaluation framework** to quantify cost/quality tradeoffs against single-model baselines.

---

## Why complexity estimation?

Most prompts don’t need your strongest LLM. By recognizing easy vs. hard requests, you can:
- **Save cost**: route simpler tasks to cheaper models.
- **Improve throughput & latency**: keep heavy models available for the few prompts that truly need them.
- **Maintain or improve quality**: only escalate when the estimated difficulty is high.

For routing, two main properties matter:
1) **Ordering** (rank): harder prompts should get higher scores. We track this with **Spearman ρ**.
2) **Closeness** (calibration-like): predicted scores should reflect empirical difficulty. We track **MAE** on [0,1].

---

## What is the complexity estimator?

A small neural model that maps raw text to **difficulty score** in [0,1].
- **Encoder** (e.g., DeBERTa/ModernBERT): converts the prompt into a fixed-size vector using mask-aware mean pooling + LayerNorm.
- **Regressor head**: a compact MLP (dropout + 2 linear layers) produces a score in \[0,1\].
- **Training data**: publicly available corpora with **explicit or proxy difficulty** labels (e.g., MATH (1–5), ARC (Easy/Challenge), APPS (Intro/Interview/Competition), RACE (Easy/Medium/Hard), ScienceQA (grade levels)). Labels are normalized to \[0,1\].
- **Metrics**: **MAE** for absolute error, **Spearman ρ** for ranking quality.

The model doesn't make actual routing decisions - it only scores the prompt complexity. 
To translate that score into model ID to route the request to, a separate **router** component is needed (e.g., based on static rules).
We provide an initial prototype of such a routing API.

---

## Repository Layout

```
.
├─ demo/                    # Streamlit app for interactive exploration
└─ eval/                    # cost/accuracy evaluation framework
├─ routing/                 # sample routing API for decision making based on the trained estimator
├─ train/                   # training the complexity estimator (from scratch)
├─ README.md                # (this file) project overview
```

### `train/`
Code to train the difficulty regressor from scratch. 
In addition to the training loop implementation, contains all the necessary steps for producing the training/validation/test datasets, i.e., synthetic data generation, preprocessing data from existing datasets, data normalization, and assembling the unified dataset from multiple sources.

### `routing/`
A **minimal PoC routing API** that converts difficulty scores into routing decisions via predefined constant thresholds. It is intended as a reference only and is used by the demo UI and the evaluation framework.

### `demo/`
A **Streamlit-based UI**: paste prompts, see the difficulty score, and how the router chooses a model. Helpful for demos and quick validation.

### `eval/`
A small **evaluation framework** that compares the total cost + accuracy of mixture-of-models using the complexity router vs. single-model baselines.

---


## License & acknowledgements

This project is licensed under the [Apache License 2.0](./LICENSE).

---

## Acknowledgements
Thanks to open datasets (MATH, ARC, APPS, RACE, ScienceQA, E2H, Cross-Difficulty).
