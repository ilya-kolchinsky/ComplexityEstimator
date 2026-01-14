# Prompt Complexity–Aware Semantic Router - Streamlit Demo

This repository contains a Streamlit demo showcasing a prompt complexity–aware semantic router based on the complexity estimator model trained under `../train/`
and a routing API defined in `../routing/routing.py`.
This UI can be used for experimentation and presentation.

---

## How It Works

1. Define environment variables for the `.pt` file containing the pretrained model and the `.yaml` configuration file - see `.env.example` and `train/README.md` for more information.
2. Load a YAML file containing the model definitions (see `demo/models_example.yaml`).
3. Optionally, define custom routing rules by specifying non-overlapping complexity intervals for your models. By default, models are assigned equally sized consecutive intervals.
4. Enter any prompt and click 'Send'.
5. The request is routed to the appropriate LLM using the complexity estimator. The chosen model and its response will be displayed.

---


## Running the Demo

```bash
streamlit run demo/app.py
```
