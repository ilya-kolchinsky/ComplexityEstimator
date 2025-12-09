import asyncio
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import httpx
import streamlit as st
import torch
import yaml
from dotenv import load_dotenv

from src.config import load_config
from src.models.encoder import HFEncoder
from src.models.regressor import Regressor
from src.utils.utils import get_device

load_dotenv()


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class ModelConfig:
    name: str
    model_id: str
    url: str
    api_key: Optional[str] = None
    cost: str = "moderate"  # cheap | moderate | expensive
    extra: dict = None  # anything else from YAML


@dataclass
class RoutingRule:
    model_name: str
    lower: float
    upper: float


# -----------------------------
# Complexity model integration
# -----------------------------

@st.cache_resource
def load_complexity_model():
    cfg = load_config(os.getenv("DEMO_MODEL_CONFIG_PATH"))
    encoder = HFEncoder(cfg.model["name"])
    model = Regressor(encoder)
    model.load_state_dict(torch.load(os.getenv("DEMO_MODEL_PATH"), map_location="cpu"))
    device = get_device(cfg)
    model.to(device).eval()
    return model, device, cfg.model["max_length"]


def estimate_complexity(prompt: str) -> float:
    model, device, max_length = load_complexity_model()

    encoded = model.enc.tokenize([prompt], max_length)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        p = model(input_ids, attention_mask)

    complexity = float(p.squeeze(0).cpu().item())

    # clamp to [0, 1] just to be safe
    complexity = max(0.0, min(1.0, complexity))
    return complexity


# -----------------------------
# Model invocation
# -----------------------------

async def call_target_model(model: ModelConfig, prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    if model.api_key:
        headers["Authorization"] = f"Bearer {model.api_key}"

    payload = {
        "model": model.model_id,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{model.url}/v1/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error calling model '{model.name}': {e}"


# -----------------------------
# Routing validation
# -----------------------------

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

    return True, "Routing configuration is valid ✅"


def route_to_model(rules: List[RoutingRule], difficulty: float) -> Optional[RoutingRule]:
    # Find rule whose interval contains difficulty
    for r in rules:
        if r.lower <= difficulty <= r.upper:
            return r
    return None


# -----------------------------
# YAML loading
# -----------------------------

def parse_models_from_yaml(yaml_text: str) -> List[ModelConfig]:
    data = yaml.safe_load(yaml_text)
    if not isinstance(data, dict) or "models" not in data:
        raise ValueError("YAML must contain a top-level 'models' list.")

    allowed_costs = {"cheap", "moderate", "expensive"}

    models = []
    for entry in data["models"]:
        name = entry.get("name")
        model_id = entry.get("model_id")
        url = entry.get("url")
        api_key = entry.get("api_key")
        cost = entry.get("cost", "moderate")

        if not name or not model_id or not url:
            raise ValueError(
                "Each model must have 'name', 'model_id', and 'url' fields."
            )

        if cost not in allowed_costs:
            raise ValueError(
                f"Invalid cost '{cost}' for model '{name}'. "
                f"Allowed values: {sorted(allowed_costs)}"
            )

        extra = {
            k: v for k, v in entry.items()
            if k not in {"name", "model_id", "url", "api_key", "cost"}
        }

        models.append(
            ModelConfig(
                name=name,
                model_id=model_id,
                url=url,
                api_key=api_key,
                cost=cost,
                extra=extra,
            )
        )

    return models


# -----------------------------
# UI helpers
# -----------------------------

def init_routing_defaults(models: List[ModelConfig]) -> List[RoutingRule]:
    n = len(models)
    if n == 0:
        return []

    step = 1.0 / n
    rules = []
    for i, m in enumerate(models):
        lower = round(i * step, 3)
        upper = round((i + 1) * step, 3) if i < n - 1 else 1.0
        rules.append(RoutingRule(model_name=m.name, lower=lower, upper=upper))
    return rules


def cost_badge(cost: str) -> str:
    mapping = {
        "cheap": "🟢 Cheap",
        "moderate": "🟡 Moderate",
        "expensive": "🔴 Expensive",
    }
    return mapping.get(cost, "🟡 Moderate")


# -----------------------------
# Streamlit App
# -----------------------------

def main():
    st.set_page_config(
        page_title="Route-By-Complexity Demo",
        page_icon="🧭",
        layout="wide",
    )

    st.title("Route-By-Complexity Demo")

    # ---- Sidebar: YAML upload ----
    st.sidebar.header("Load Models (YAML)")
    yaml_file = st.sidebar.file_uploader(
        "Upload YAML with model definitions",
        type=["yml", "yaml"],
        help="Must contain a top-level 'models' list.",
    )

    if yaml_file is not None:
        yaml_text = yaml_file.read().decode("utf-8")

        try:
            models = parse_models_from_yaml(yaml_text)
            st.session_state["models"] = models
        except Exception as e:
            st.sidebar.error(f"Failed to parse YAML: {e}")
            models = []
    else:
        models = st.session_state.get("models", [])

    st.subheader("Available Models & Routing Rules")

    if not models:
        st.info("Upload a YAML file in the sidebar to see available models.")
    else:
        # Initialize routing in session state if not present / size changed
        if "routing_rules" not in st.session_state or len(st.session_state["routing_rules"]) != len(models):
            st.session_state["routing_rules"] = init_routing_defaults(models)

        rules = st.session_state["routing_rules"]

        updated_rules: List[RoutingRule] = []

        for i, m in enumerate(models):

            # Show a nice list
            with st.container(border=True):
                st.markdown(f"### 🧩 {m.name}")
                st.markdown(f"- **Model ID:** `{m.model_id}`")
                st.markdown(f"- **URL:** `{m.url}`")
                st.markdown(f"- **Cost:** {cost_badge(m.cost)}")

                if m.api_key:
                    st.markdown(f"- **API key:** `••••••••`")
                if m.extra:
                    with st.expander("Extra config"):
                        st.json(m.extra)

                rule = rules[i]
                c1, c2 = st.columns(2)
                lower = c1.number_input(
                    "Lower bound (inclusive)",
                    min_value=0.0,
                    max_value=1.0,
                    value=rule.lower,
                    step=0.01,
                    key=f"lower_{i}",
                )
                upper = c2.number_input(
                    "Upper bound (inclusive for last model)",
                    min_value=0.0,
                    max_value=1.0,
                    value=rule.upper,
                    step=0.01,
                    key=f"upper_{i}",
                )
                updated_rules.append(
                    RoutingRule(model_name=m.name, lower=float(lower), upper=float(upper))
                )

        # Validate configuration
        is_valid, msg = validate_routing_rules(updated_rules)
        st.session_state["routing_rules"] = updated_rules

        if is_valid:
            st.success(msg)
            # Show a small visual of ranges
            st.markdown("**Coverage:**")
            coverage_str = " | ".join(
                f"{r.model_name}: [{r.lower:.2f}, {r.upper:.2f}]"
                for r in updated_rules
            )
            st.code(coverage_str)
        else:
            st.error(msg)

    st.markdown("---")

    # ---- Prompt input & routing ----
    st.subheader("Try It Out")

    prompt = st.text_area(
        "Enter a prompt:",
        height=180,
        placeholder="Ask a question or type a complex instruction to see how it is routed...",
    )

    disabled = not (models and st.session_state.get("routing_rules"))
    if st.button("Send", use_container_width=True, disabled=disabled):
        if not models:
            st.error("Please upload a valid models YAML first.")
        elif not prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            rules = st.session_state["routing_rules"]
            is_valid, msg = validate_routing_rules(rules)
            if not is_valid:
                st.error(f"Invalid routing config: {msg}")
            else:
                with st.spinner("Estimating query complexity & routing..."):
                    complexity = estimate_complexity(prompt)
                    routed_rule = route_to_model(rules, complexity)

                if routed_rule is None:
                    st.error(
                        f"No routing rule matched complexity {complexity:.3f}. "
                        "Check your intervals."
                    )
                else:
                    model = next(m for m in models if m.name == routed_rule.model_name)

                    # Display complexity & choice
                    st.markdown("### Routing Result")
                    c1, c2 = st.columns(2)
                    c1.metric("Estimated Complexity", f"{complexity:.3f}", help="0 = easy, 1 = very hard")
                    with c1:
                        st.progress(min(1.0, max(0.0, complexity)))

                    c2.metric("Chosen Model", model.name)

                    # Call the model
                    with st.spinner("Awaiting reply from the chosen model..."):
                        reply = asyncio.run(call_target_model(model, prompt))
                        st.markdown("### Model Reply")
                        st.write(reply)


if __name__ == "__main__":
    main()
