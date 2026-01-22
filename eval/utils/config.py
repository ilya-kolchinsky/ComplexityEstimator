from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import yaml


@dataclass
class ModelConfig:
    id: str
    base_url: Optional[str]  # should not be specified for a model which we only fetch HELM data for but never run
    cost_per_token: Optional[float]  # for API-based models with the price available from the provider
    inference_scenario: Optional[Dict[str, Any]]  # for local models, simulation data for cost per token estimation


@dataclass
class EvalConfig:
    models: List[ModelConfig]
    binary_thresholds: List[float]
    dataset_id: str
    helm_root_dir: Optional[str]


def load_config(path: str) -> EvalConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return EvalConfig(**cfg)
