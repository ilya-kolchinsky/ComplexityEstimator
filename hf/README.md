---
library_name: transformers
pipeline_tag: text-classification
tags:
  - regression
  - prompt
  - complexity-estimation
  - semantic-routing
  - llm-routing
base_model: microsoft/deberta-v3-base
license: apache-2.0
---

# PromptComplexityEstimator

A lightweight regressor that estimates the complexity of an LLM prompt on a scale between 0 and 1.

- **Input:** a string prompt  
- **Output:** a scalar score in [0, 1] (higher = more complex)

The model is designed primarily to be used as a core building block for semantic routing systems, especially LLM vs. SLM (Small Language Model) routers.  
Any router that aims to intelligently decide *which model should handle a request* needs a reliable signal for *how complex the request is*. This is the gap this model aims to close.

---

## Intended use

### Primary use case: LLM vs. SLM routing

This model is intended to be used as part of a semantic router, where:
- *Simple* prompts are handled by a **small / fast / cheap model**
- *Complex* prompts are routed to a **large / capable / expensive model**

The complexity score provides a learned signal for this decision.

### Additional use cases
- Prompt analytics and monitoring
- Dataset stratification by difficulty
- Adaptive compute allocation
- Cost-aware or latency-aware inference pipelines

### Not intended for
- Safety classification, toxicity detection, or policy enforcement
- Guaranteed difficulty estimation for a specific target model
- Multimodal inputs or tool-augmented workflows (RAG/tools)

---

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModel

repo_id = "ilya-kolchinsky/PromptComplexityEstimator"

tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).eval()

prompt = "Design a distributed consensus protocol with Byzantine fault tolerance..."
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    score = model(**inputs).logits.squeeze(-1).item()

print(float(score))
```


### Example: Simple LLM vs. SLM routing

```python
THRESHOLD = 0.45  # chosen empirically

def route_prompt(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        complexity = model(**inputs).logits.squeeze(-1).item()

    return "LLM" if complexity > THRESHOLD else "SLM"
```

---


## Model and Training Details

### Datasets
- Cross-Difficulty (https://huggingface.co/datasets/BatsResearch/Cross-Difficulty)
- [Easy2Hard-Bench](https://huggingface.co/datasets/furonghuang-lab/Easy2Hard-Bench)
- [MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math)
- [ARC](https://huggingface.co/datasets/allenai/ai2_arc)
- [RACE](https://huggingface.co/datasets/ehovy/race)
- [ANLI (R1/R2/R3)](https://huggingface.co/datasets/facebook/anli)

### Training Configuration
- **Epochs:** 3
- **Batch Size:** 16
- **Loss:** huber
- **Regressor Learning Rate:** 7.5e-5
- **Encoder Learning Rate:** 1.0e-5
- **Encoder Weight Decay:** 0.01
- **Optimizer**: AdamW
- **Schedule**: Cosine (warmup_ratio=0.06)
- **Dropout**: 0.1

### Model
- **Backbone encoder:** microsoft/deberta-v3-base
- Mask-aware **mean pooling** over token embeddings + **LayerNorm**
- **Regression head:** Dropout(0.1) → Linear → ReLU → Linear → Sigmoid
- **Max input length:** 512 tokens
- The model outputs a bounded score in [0, 1]. In the examples below, the score is read from `outputs.logits` (shape `[batch, 1]`).


Full training code and configuration are available at https://github.com/ilya-kolchinsky/ComplexityEstimator.

---

## Performance

On the held-out evaluation set used during development, the released checkpoint achieved:

- **MAE:** **0.0855**
- **Spearman correlation:** **0.735**

---

## Citation

```bibtex
@misc{kolchinsky_promptcomplexityestimator_2026,
  title        = {PromptComplexityEstimator},
  author       = {Ilya Kolchinsky},
  year         = {2026},
  howpublished = {Hugging Face Hub model: ilya-kolchinsky/PromptComplexityEstimator}
}
```
