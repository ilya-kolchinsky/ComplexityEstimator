from dataclasses import dataclass
import yaml


@dataclass
class Config:
    seed: int
    device: str
    model: dict
    train: dict
    data: dict
    datasets: list
    logging: dict
    eval: dict


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)
