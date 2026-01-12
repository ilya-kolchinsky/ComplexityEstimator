from dataclasses import dataclass
from typing import List

import yaml


@dataclass
class ModelConfig:
    id: str
    base_url: str  # None for a model which we only fetch HELM data for but never run
    cost_per_token: float


@dataclass
class EvalConfig:
    models: List[ModelConfig]
    binary_thresholds: List[float]
    dataset_id: str


def load_config(path: str) -> EvalConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return EvalConfig(**cfg)
