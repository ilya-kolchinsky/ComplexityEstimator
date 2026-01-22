import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable

import torch
from dotenv import load_dotenv

from train.models.encoder import HFEncoder
from train.models.regressor import Regressor
from train.utils.config import load_config
from train.utils.utils import get_device

load_dotenv()


@dataclass
class RoutingRule:
    model_name: str
    lower: float
    upper: float


RouteFn = Callable[[str], str]


def load_complexity_model():
    cfg = load_config(os.getenv("ROUTER_MODEL_CONFIG_PATH"))
    encoder = HFEncoder(cfg.model["name"], init_weights=False)
    model = Regressor(encoder)
    model.load_state_dict(torch.load(os.getenv("ROUTER_MODEL_PATH"), map_location="cpu", weights_only=True))
    device = get_device(cfg)
    model.to(device).eval()
    return model, device, cfg.model["max_length"]


def estimate_complexity(model: Regressor, device: str, max_length: int, prompt: str) -> float:
    encoded = model.enc.tokenize([prompt], max_length)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        p = model(input_ids, attention_mask)

    return float(p.squeeze(0).cpu().item())


def route_to_model(rules: List[RoutingRule], difficulty: float) -> Optional[str]:
    # Find rule whose interval contains difficulty
    for r in rules:
        if r.lower <= difficulty <= r.upper:
            return r.model_name
    return None


def validate_routing_rules(rules: List[RoutingRule]) -> Tuple[bool, str]:
    """
    Validate that:
    - 0 <= lower < upper <= 1 for each rule
    - Intervals cover [0,1] exactly with no gaps/overlaps
    """
    if not rules:
        return False, "No routing rules defined."

    # Basic sanity per rule
    for rule in rules:
        if not (0.0 <= rule.lower < rule.upper <= 1.0):
            return False, f"Invalid interval for {rule.model_name}: [{rule.lower}, {rule.upper}]"

    # Sort by lower bound
    sorted_rules = sorted(rules, key=lambda r: r.lower)

    eps = 1e-6

    # Check coverage start and end
    if abs(sorted_rules[0].lower - 0.0) > eps:
        return False, f"Rules must start at 0.0, first rule starts at {sorted_rules[0].lower:.3f}"

    if abs(sorted_rules[-1].upper - 1.0) > eps:
        return False, f"Rules must end at 1.0, last rule ends at {sorted_rules[-1].upper:.3f}"

    # Check adjacency (no gaps / overlaps)
    for prev, curr in zip(sorted_rules, sorted_rules[1:]):
        if abs(prev.upper - curr.lower) > eps:
            return False, (
                "Intervals must touch exactly with no gaps or overlaps. "
                f"{prev.model_name} ends at {prev.upper:.3f}, "
                f"{curr.model_name} starts at {curr.lower:.3f}"
            )

    return True, "Routing configuration is valid."


def create_single_model_route_function(model_id: str) -> RouteFn:
    return lambda e: model_id


def create_complexity_model_route_function(rules: List[RoutingRule]) -> Optional[RouteFn]:
    is_valid, reason = validate_routing_rules(rules)
    if not is_valid:
        print(reason)
        return None

    model, device, max_length = load_complexity_model()

    def route(prompt: str):
        complexity = estimate_complexity(model, device, max_length, prompt)
        return route_to_model(rules, complexity)

    return route


def create_single_threshold_route_function(small_model_id: str, large_model_id: str, threshold: float) -> RouteFn:
    rules = [
        RoutingRule(model_name=large_model_id, lower=threshold, upper=1.0),
        RoutingRule(model_name=small_model_id, lower=0.0, upper=threshold),
    ]
    return create_complexity_model_route_function(rules)
