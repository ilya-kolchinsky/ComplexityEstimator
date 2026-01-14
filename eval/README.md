# Prompt Complexity–Aware Semantic Router - Evaluation Framework

This repository contains a small, purpose-built framework for evaluating the gains and tradeoffs of complexity-based routing in multi-model setups.
To minimize costs and redundant computation, this framework:
- uses data from [HELM Lite](https://crfm.stanford.edu/helm/lite/latest/#/) instead of directly querying frontier models;
- maintains a permanent cache to avoid duplicate execution of queries on locally deployed models.

By utilizing the complexity estimator (see `train/`) and the complexity-aware routing API based on it (see `routing/routing.py`), our evaluation framework allows to compare single-model runs (with either small/cheap locally deployed models or large/expensive API-based ones) to sophisticated routing schemes where prompts of different complexities are routed to different inference endpoints.

The final evaluation report includes the estimated accuracy and the total cost (based on user-provided cost per token per model). Statistics are produced for each model in isolation and for the provided routing scheme.

## How It Works

1. **One model per query:**  
   An optional routing function `route(example) -> model_id` chooses *exactly one* model for each example.

2. **HELM data first:**  
   For frontier models that are already in HELM, we read:
   - Input prompt
   - Model output
   - Correctness  
   directly from `scenario_state.json`. No API calls needed.

3. **Local models only when needed:**  
   If the routing function chooses a model that is not in HELM:
   - We run the query on the local model.
   - We evaluate correctness (either via a judge model or a deterministic rule).
   - We cache the correctness so we never re-evaluate that triple again.

4. **MMLU is special (and easy):**  
   For MMLU-like multiple-choice datasets:
   - HELM prompts instruct models to output a choice index (A/B/C/etc.).
   - We derive the correct letter from the scenario state.
   - For local models, we just compare the letter — no judge LLM required.

5. **Cost-aware analysis:**  
   Given per-token prices and a tokenizer, we can estimate:
   - Token count per example,
   - Total cost of a routing policy,
   - Cost–accuracy tradeoffs across experiments.


## Data Layout

The framework assumes you manually download HELM scenario state files and organize them locally as follows:

```text
<helm_root>/
  <dataset_id>/
    <model_id>_scenario_state.json
    <other_model_id>_scenario_state.json
    ...
  <another_dataset_id>/
    ...
```

Examples:

```text
data/helm/
  mmlu/
    openai_gpt-4-0613_scenario_state.json
    meta_llama-2-70b_scenario_state.json
    ...
  mmlu_high_school_mathematics/
    openai_gpt-4-0613_scenario_state.json
    ...
```

Please use the `EVAL_ROOT_DIR` environment variable to provide the root directory path to the evaluation framework.

Each `*_scenario_state.json` is a serialized HELM `ScenarioState` containing:

- `request_states`: a list of entries, each with:
  - `instance` (question metadata)
  - `request.prompt` (full prompt sent to the model)
  - `result.completions[0].text` (model output)
  - `output_mapping` for multiple-choice tasks like MMLU
  - correctness-related information we can infer



## Configuration File

The evaluation framework expects a YAML configuration file to read the settings from. 
The following settings must be provided:
 - Definitions of the models participating in the evaluation (model ID, cost per token, URL for a locally deployed model). *NOTE: as of this version, only two-model evaluation mode is supported, where exactly one model is local and exactly one is remote (API-based)! In this very limited case, the routing rule is defined by a single "local vs. remote" threshold number in the [0,1] interval. More advanced evaluation scenarios will be supported in the near future.*
 - Binary thresholds to build the routing rules from (e.g., given a threshold 0.5, the rule *'route all requests with complexity below 0.5 to the local model, otherwise to the remote model'* will be constructed). A dedicated evaluation procedure will be executed for each threshold.
 - The HELM dataset ID.

Please see `eval/sample_config.yaml` for more information.

## Usage

After downloading the dataset and creating the configuration file, simply run:

```bash
python eval/run_eval.py --config eval/my_config.yaml
```

## Limitations & Notes

- **Manual data download:**  
  This framework assumes you manually download `scenario_state.json` files and place them under the expected directory layout. There is no automatic GCS / HTTP downloader in this repo.

- **Dataset assumptions:**  
  As of now, only the HELM Lite MMLU dataset is supported. In particular, `MmluHelmStore` assumes MMLU-like multiple-choice tasks where models are instructed to output a choice index (A/B/C/D). For other datasets, you’ll need custom logic.

- **Judge model:**  
  For MMLU and similar multiple-choice tasks, we avoid using any judge LLM and rely on deterministic choice-letter matching. For other datasets, the correctness depends on the judge you supply (`LlmJudge` or your own).

- **Tokenization & pricing:**  
  Cost estimation depends on your token_counter and cost_per_token values. There is no built-in integration with billing APIs; everything is user-specified.

- **No parallelism yet:**  
  The current `ExperimentRunner` runs examples sequentially. If needed, you can add batching or concurrency around local model evaluation.
