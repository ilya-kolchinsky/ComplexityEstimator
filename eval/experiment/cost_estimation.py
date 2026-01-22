from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Mapping


class GPUCategory(str, Enum):
    """
    High-level GPU categories for demo scenarios.

    The defaults are calibrated for ~7B models and are *approximate*,
    based on public cloud pricing and vLLM-style benchmarks:

    - CONSUMER: e.g. RTX 3060 / laptop GPU
    - L4:       NVIDIA L4-class GPU
    - A10:      NVIDIA A10-class GPU
    - A100:     NVIDIA A100 40GB/80GB-class GPU

    You can always override the assumed cost_per_hour or base_tps_7b
    if you want to fine-tune the numbers for your org.
    """
    CONSUMER = "consumer"
    L4 = "l4"
    A10 = "a10"
    A100 = "a100"


class ThroughputMode(str, Enum):
    """
    Rough operating regimes:

    - CONSERVATIVE: interactive, minimal batching, worst-case-ish
    - TYPICAL:      realistic internal deployment with some batching
    - THROUGHPUT:   high-throughput, heavy batching (demo upper bound)
    """
    CONSERVATIVE = "conservative"
    TYPICAL = "typical"
    HIGH_THROUGHPUT = "high-throughput"


@dataclass
class InferenceScenario:
    """
    Input description of a local inference setup.
    """
    gpu_category: GPUCategory
    throughput_mode: ThroughputMode = ThroughputMode.TYPICAL
    model_size_billion_params: float = 7.0  # 7B by default
    cost_per_hour: float | None = None,
    base_tps_7b: float | None = None,

    def scale_relative_to_7b(self) -> float:
        """
        Simple scaling factor for tokens/sec: assume decode throughput
        is inversely proportional to model size.

        Example:
          - 7B  -> scale = 1.0
          - 14B -> scale = 7 / 14 = 0.5 (half the tokens/sec)
          - 3B  -> scale = 7 / 3  ≈ 2.33 (more tokens/sec)
        """
        if self.model_size_billion_params <= 0:
            return 1.0
        return 7.0 / self.model_size_billion_params


@dataclass
class InferenceEconomics:
    """
    Output: derived economics for a given scenario.
    """
    scenario: InferenceScenario
    cost_per_hour: float          # USD / GPU-hour
    tokens_per_second: float      # estimated decode tokens per second
    cost_per_token: float         # USD / token
    cost_per_million_tokens: float  # USD / 1M tokens


# Approximate *on-demand* GPU prices (USD / hour) for 2025-ish, mid-range values:
# - L4:   ~0.8–2.0 $/h on various clouds and platforms -> use 0.90 as a round, conservative-ish default
# - A10:  ~0.3–1.1 $/h depending on provider -> use 1.00
# - A100: ~1.0–2.5 $/h on specialized platforms -> use 2.00 as a simple default
# - CONSUMER: very rough amortized + electricity cost -> use 0.30
#
# These are intentionally rounded for story-telling, not billing.
DEFAULT_COST_PER_HOUR: Dict[GPUCategory, float] = {
    GPUCategory.CONSUMER: 0.30,
    GPUCategory.L4: 0.90,
    GPUCategory.A10: 1.00,
    GPUCategory.A100: 2.00,
}

# Base tokens/sec for a ~7B model in "TYPICAL" mode, per GPU category.
# These are ballpark values derived from public vLLM benchmarks and community reports:
# - CONSUMER:  ~15 tok/s (desktop / gaming GPU / laptop)
# - L4:        ~80 tok/s
# - A10:       ~100 tok/s
# - A100:      ~250 tok/s  (well below the 1.8k+ tok/s max-throughput numbers)
DEFAULT_BASE_TPS_7B_TYPICAL: Dict[GPUCategory, float] = {
    GPUCategory.CONSUMER: 15.0,
    GPUCategory.L4: 80.0,
    GPUCategory.A10: 100.0,
    GPUCategory.A100: 250.0,
}

# Throughput multipliers on top of "TYPICAL":
THROUGHPUT_MODE_MULTIPLIER: Dict[ThroughputMode, float] = {
    ThroughputMode.CONSERVATIVE: 0.5,   # half the typical throughput
    ThroughputMode.TYPICAL: 1.0,        # baseline
    ThroughputMode.HIGH_THROUGHPUT: 2.5,     # optimistic batching / tuning
}


def _parse_gpu_category(value: str) -> GPUCategory:
    v = value.strip().lower()
    mapping = {
        "consumer": GPUCategory.CONSUMER,
        "desktop": GPUCategory.CONSUMER,
        "rtx": GPUCategory.CONSUMER,
        "l4": GPUCategory.L4,
        "a10": GPUCategory.A10,
        "a100": GPUCategory.A100,
    }
    if v not in mapping:
        raise ValueError(
            f"Unknown gpu_category={value!r}. "
            f"Expected one of: {', '.join(sorted(set(mapping.keys())))}"
        )
    return mapping[v]


def _parse_throughput_mode(value: str) -> ThroughputMode:
    v = value.strip().lower()
    mapping = {
        "conservative": ThroughputMode.CONSERVATIVE,
        "typical": ThroughputMode.TYPICAL,
        "high-throughput": ThroughputMode.HIGH_THROUGHPUT,
        "max": ThroughputMode.HIGH_THROUGHPUT,
        "high": ThroughputMode.HIGH_THROUGHPUT,
    }
    if v not in mapping:
        raise ValueError(
            f"Unknown throughput_mode={value!r}. "
            f"Expected one of: {', '.join(sorted(set(mapping.keys())))}"
        )
    return mapping[v]


def inference_scenario_from_dict(config: Mapping[str, Any]) -> InferenceScenario:
    """
    Construct an InferenceScenario from a simple dict-like object.

    Expected keys in `config`:
      - gpu_category: str (required)
      - throughput_mode: str (optional, default "typical")
      - model_size_billion_params: float (optional, default 7.0)
      - cost_per_hour: float (optional)
      - base_tps_7b: float (optional)
    """
    if "gpu_category" not in config:
        raise ValueError(
            "Missing required key 'gpu_category' in inference_scenario config."
        )

    gpu_cat = _parse_gpu_category(str(config["gpu_category"]))

    throughput_mode_str = str(
        config.get("throughput_mode", "typical")
    )
    throughput_mode = _parse_throughput_mode(throughput_mode_str)

    model_size = float(config.get("model_size_billion_params", 7.0))
    cost_per_hour = float(config.get("cost_per_hour")) if config.get("cost_per_hour") is not None else None
    base_tps_7b = float(config.get("base_tps_7b")) if config.get("base_tps_7b") is not None else None

    return InferenceScenario(
        gpu_category=gpu_cat,
        throughput_mode=throughput_mode,
        model_size_billion_params=model_size,
        cost_per_hour=cost_per_hour,
        base_tps_7b=base_tps_7b,
    )


def estimate_inference_cost(scenario: InferenceScenario) -> InferenceEconomics:
    """
    Estimate (cost_per_hour, tokens_per_second) and derived quantities
    for a given local model deployment scenario.

    Parameters
    ----------
    scenario:
        InferenceScenario describing GPU category, throughput mode,
        model size, const per hour and baseline tokens/s for a 7B model.

    Returns
    -------
    InferenceEconomics:
        cost_per_hour, tokens_per_second, cost_per_token, cost_per_million_tokens

    Notes
    -----
    - All numbers are *approximate*, intended for demos and what-if analyses, not billing.
    - Tokens/sec scales inversely with model size: bigger model -> fewer tokens/sec.
    """
    gpu = scenario.gpu_category

    # 1) Cost per hour
    if scenario.cost_per_hour is not None:
        cost_per_hour = float(scenario.cost_per_hour)
    else:
        if gpu not in DEFAULT_COST_PER_HOUR:
            raise ValueError(f"No default cost_per_hour for GPU category {gpu!r}")
        cost_per_hour = DEFAULT_COST_PER_HOUR[gpu]

    # 2) Base tokens/sec for 7B in "typical" mode
    if scenario.base_tps_7b is not None:
        base_tps_7b_typical = float(scenario.base_tps_7b)
    else:
        if gpu not in DEFAULT_BASE_TPS_7B_TYPICAL:
            raise ValueError(f"No default base TPS for GPU category {gpu!r}")
        base_tps_7b_typical = DEFAULT_BASE_TPS_7B_TYPICAL[gpu]

    # 3) Adjust for throughput mode
    mode_mult = THROUGHPUT_MODE_MULTIPLIER.get(
        scenario.throughput_mode, 1.0
    )
    tps_7b = base_tps_7b_typical * mode_mult

    # 4) Scale for model size (inverse proportional to parameter count)
    size_scale = scenario.scale_relative_to_7b()
    tokens_per_second = tps_7b * size_scale

    # 5) Cost per token derived from cost/hour and tokens/sec
    tokens_per_hour = max(tokens_per_second, 1e-9) * 3600.0
    cost_per_token = cost_per_hour / tokens_per_hour
    cost_per_million_tokens = cost_per_token * 1_000_000.0

    return InferenceEconomics(
        scenario=scenario,
        cost_per_hour=cost_per_hour,
        tokens_per_second=tokens_per_second,
        cost_per_token=cost_per_token,
        cost_per_million_tokens=cost_per_million_tokens,
    )


def get_cost_per_token_from_scenario_dict(config: Mapping[str, Any]) -> float:
    scenario = inference_scenario_from_dict(config)
    inference_economics = estimate_inference_cost(scenario)
    return inference_economics.cost_per_token
