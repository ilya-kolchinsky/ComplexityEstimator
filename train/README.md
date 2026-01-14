# Complexity Estimator — Training (`train/`)

This directory contains the code to train and validate a model-independent difficulty regressor that maps a prompt to a score in [0,1]. The model is later used by the router (see `../routing/`) to select an LLM tier.


---

## Data Pipeline

### Overview

The training framework consists of the following components:
- **Floor dataset generator** (`train/data/preprocess/generate_floor_pack.py`) generates a diverse dataset of basic, very simple prompts such as 'Hi there', '2+3=' and so on. This dataset is necessary in order to teach the model to assign low complexity scores.
- **Custom per-datasets preprocessors** (`train/data/preprocess/preprocess_*.py`) convert data from existing datasets into unified format and heuristically assign complexity scores if missing.
- **Dataset join/mixing script** (`train/data/join_datasets.py`) builds unified CSVs from separate per-dataset preprocessed CSVs, using the mixing schema in `train/config.yaml`.

### CSV Schema

The training framework expects train/validation/test CSV files with the following columns:

```
id,dataset,split,prompt_text,label_raw
```

- `id`: unique row id (stable)
- `dataset`: short source name (e.g., `crossdiff`, `e2h`, `math`, `arc`, `race`, `anli`, `floorpack`)
- `split`: one of `train|dev|test` (local to the dataset preprocessor)
- `prompt_text`: rendered prompt (with optional `<opts> ... </opts>` tags for MC options)
- `label_raw`: the source-specific difficulty label (may require normalization)

### Supported Datasets

- **Cross-Difficulty (BatsResearch/Cross-Difficulty):** uses `1pl_diff` (absolute difficulty). Adds randomized MC templates, `<opts>` tagging, **option-dependence** flags and meta sidecar.  
- **Easy2Hard-Bench (E2H)** *(without chess subset)*: uses `rating` (absolute). Joins and resplits, handles subsets that only have `eval`.  
- **MATH (EleutherAI/hendrycks_math):** maps “Level 1–5” to difficulty bands.  
- **ARC, RACE:** MC-heavy reasoning/reading comprehension.  
- **ANLI (R1/R2/R3):** adversarial NLI with increasing difficulty.

> You can include/exclude sources via the mix config and/or adjust their contribution via ratios.

### Unifying Datasets (`train/data/join_datasets.py`)

**What it does**

1. **Pools per dataset:** loads that dataset’s `*_train.csv`, `*_dev.csv`, `*_test.csv` and joins them, optionally de-duplicated by prompt text.
2. **Total size:**  
   - Given via `--total_size` parameter, or  
   - AUTO: compute the maximum feasible total so the dataset mix ratios can be honored given availability.  
3. **Split sizes:** compute `train/dev/test` sizes from `--train_ratio`, `--dev_ratio`, `--test_ratio`.  
4. **Allocate & sample:** computes per-split per-dataset counts via Hamilton rounding with capacity checks, samples without replacement, shuffles deterministically, and writes to the specified output paths.

**YAML mix example**

```yaml
datasets:
  - name: e2h
    dir: data/raw/e2h
    prefix: e2h
    weight: 0.30  # ratios; normalized across available datasets
    enabled: true
  - name: crossdiff
    dir: data/raw/crossdiff
    prefix: crossdiff
    weight: 0.30
    enabled: true
  - name: math
    dir: data/raw/math
    prefix: math
    weight: 0.20
    enabled: true
  - name: arc
    dir: data/raw/arc
    prefix: arc
    weight: 0.20
    enabled: true
```

---

## Training

The training loop implementation can be found in `train/run_train.py`. 
Use `train/model_eval.py` to evaluate the trained model on the test dataset.

The settings for both the training and the evaluation stages must be provided as a YAML configuration file (see usage example below).
The following training settings are supported:
```yaml
model:
  name: "microsoft/deberta-v3-base"
  dropout: 0.1
  max_length: 512
train:
  loss: huber
  rank_lambda: 0.0
  batch_size: 16
  lr_encoder: 1.0e-5
  lr_head: 7.5e-5
  weight_decay_encoder: 0.01
  weight_decay_head: 0.0
  num_epochs: 3
  warmup_ratio: 0.06
  grad_accum_steps: 2
  fp16: true
  freeze_encoder: false
  encoder_unfrozen_layers: 3
```

---


## Usage

### Dataset Creation
```bash
python train/data/join_datasets.py --mix train/config.yaml
```

### Train
```bash
python train/run_train.py --config train/config.yaml
```

### Evaluate
```bash
python train/model_eval.py --config train/config.yaml --ckpt runs/exp1/best.pt
```

