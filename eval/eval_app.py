from typing import List

import streamlit as st

from eval.experiment.cost_estimation import GPUCategory, ThroughputMode, InferenceScenario, estimate_inference_cost
from eval.run_eval import eval_two_model_setup
from eval.utils.config import EvalConfig, ModelConfig
from routing.routing import load_complexity_model

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Inference Cost Evaluation",
    layout="wide",
)

st.title("Frontier vs. Local vs. Hybrid Cost Evaluation")

MAX_THRESHOLDS = 10

if "num_thresholds" not in st.session_state:
    st.session_state.num_thresholds = 1

with st.sidebar:
    st.header("Dataset & HELM setup")

    helm_root_dir = st.text_input(
        "HELM root directory",
        value="data/eval",
        help="Directory that contains the HELM dataset subdirectories.",
    )

    dataset_id = st.text_input(
        "Dataset ID",
        value="mmlu",
        help="Name of the dataset subdirectory under the HELM root.",
    )

    st.markdown("---")
    st.header("Routing")

    st.write("Use the sliders to define one or more thresholds in [0,1].")

    # Render sliders based on num_thresholds
    current_thresholds: List[float] = []
    for i in range(st.session_state.num_thresholds):
        threshold = st.slider(
            f"Threshold {i + 1}",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            key=f"threshold_{i}",
        )
        current_thresholds.append(threshold)

    col_add, col_remove = st.columns(2)
    with col_add:
        if st.button("Add threshold"):
            if st.session_state.num_thresholds < MAX_THRESHOLDS:
                st.session_state.num_thresholds += 1
                st.rerun()
            else:
                st.warning(f"Maximum of {MAX_THRESHOLDS} thresholds reached.", icon="⚠️")
    with col_remove:
        if st.button("Remove threshold"):
            if st.session_state.num_thresholds > 1:
                st.session_state.num_thresholds -= 1
                st.rerun()
            else:
                st.warning("At least one threshold must remain.", icon="⚠️")

st.subheader("Frontier model configuration")

col_f1, col_f2 = st.columns(2)

with col_f1:
    frontier_model_id = st.text_input(
        "Frontier model ID",
        value="claude_3.5_sonnet",
        help="Model identifier as it appears in HELM scenario_state filenames.",
    )

with col_f2:
    frontier_cpm = st.number_input(
        "Frontier: cost per 1M input tokens (USD)",
        min_value=0.0,
        value=10.0,
        step=0.1,
        help="Use the published price from your provider.",
    )
    frontier_cost_per_token = frontier_cpm / 1_000_000.0

st.caption(
    f"Frontier model effective cost/token: {frontier_cost_per_token:.2e} USD "
    f"(≈ {frontier_cpm:.2f} USD / 1M tokens)."
)

st.markdown("---")

st.subheader("Local model configuration")

col_l1, col_l2 = st.columns(2)

with col_l1:
    local_model_id = st.text_input(
        "Local model ID",
        value="granite32-8b",
        help="Identifier you use in your vLLM server.",
    )

    local_model_url = st.text_input(
        "Local vLLM base URL",
        value="http://localhost:8000/v1",
        help="Base URL of the vLLM OpenAI-compatible endpoint.",
    )

    local_api_key = st.text_input(
        "Local API key (if needed)",
        value="dummy",
        help="API key to access vLLM server (ignored if your setup doesn't require it).",
    )

with col_l2:
    gpu_category = st.selectbox(
        "GPU category",
        options=[c.value for c in GPUCategory],
        index=1,  # default to "l4"
        help="Hardware class for the local model deployment (approximate).",
    )

    throughput_mode = st.selectbox(
        "Throughput mode",
        options=[m.value for m in ThroughputMode],
        index=1,  # default to "typical"
        help=(
            "conservative: low batching, interactive\n"
            "typical: realistic internal deployment\n"
            "throughput: tuned for high batch throughput"
        ),
    )

    model_size_billion_params = st.number_input(
        "Model size (billions of parameters)",
        min_value=0.1,
        value=8.0,
        step=0.5,
    )

    override_cost_per_hour = st.number_input(
        "Override cost/hour (USD) - leave at 0.0 to use the built-in defaults for this GPU category",
        min_value=0.0,
        value=0.0,
        step=0.1,
        help="Leave at 0.0 to use the built-in defaults for this GPU category.",
    )

    override_base_tps_7b = st.number_input(
        "Override base TPS for 7B (tokens/sec) - leave at 0.0 to use default throughput assumptions",
        min_value=0.0,
        value=0.0,
        step=10.0,
        help="Leave at 0.0 to use default throughput assumptions.",
    )

# Compute local inference economics based on the inputs
scenario = InferenceScenario(
    gpu_category=gpu_category,
    throughput_mode=throughput_mode,
    model_size_billion_params=model_size_billion_params,
    cost_per_hour=override_cost_per_hour if override_cost_per_hour else None,
    base_tps_7b=override_base_tps_7b if override_base_tps_7b else None,
)

local_cost = estimate_inference_cost(scenario)

local_cost_per_token = local_cost.cost_per_token
local_cpm = local_cost.cost_per_million_tokens

st.markdown(
    f"Local model estimated cost/token: {local_cost_per_token:.2e} USD "
    f"(≈ {local_cpm:.2f} USD / 1M tokens, "
    f"{local_cost.tokens_per_second:.1f} tokens/s, "
    f"{local_cost.cost_per_hour:.2f} USD/hour)."
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

st.subheader("Run evaluation")

run_button = st.button("Run evaluation")
summary_container = st.container()
log_container = st.container()

with log_container:
    # Add some CSS for a dark, terminal-like box
    st.markdown(
        """
        <style>
        .log-box {
            background-color: #111111;
            color: #f1f1f1;
            padding: 0.75rem 1rem;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.85rem;
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #333333;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # This placeholder will be updated with the log text
    log_placeholder = st.empty()

if run_button:
    # Basic sanity checks
    if not frontier_model_id or not local_model_id:
        st.error("Please specify both a frontier model ID and a local model ID.")
    elif not dataset_id:
        st.error("Please specify a dataset ID.")
    else:
        # log buffer for this run
        log_lines: list[str] = []

        def log_to_streamlit(message: str) -> None:
            log_lines.append(message)
            text = "\n".join(log_lines[-50:])
            # Update the placeholder
            log_placeholder.code(text, language="bash")

        # loading the complexity estimation model can take a while the first time,
        # hence we do it explicitly before starting the evaluation
        with st.spinner("Loading complexity estimation model..."):
            load_complexity_model()

        with st.spinner("Loading HELM data and running evaluation..."):
            try:
                frontier_model_config = ModelConfig(
                    id=frontier_model_id,
                    base_url=None,
                    cost_per_token=frontier_cost_per_token,
                    inference_scenario=None
                )
                local_model_config = ModelConfig(
                    id=local_model_id,
                    base_url=local_model_url,
                    cost_per_token=None,
                    inference_scenario=scenario.__dict__,
                )
                models = [frontier_model_config.__dict__, local_model_config.__dict__]

                eval_config = EvalConfig(
                    models=models,
                    binary_thresholds=current_thresholds,
                    dataset_id=dataset_id,
                    helm_root_dir=helm_root_dir,
                )

                results = eval_two_model_setup(eval_config, log_callback=log_to_streamlit)

            except Exception as e:
                st.exception(e)
            else:
                st.success("Evaluation complete.")

                with summary_container:
                    columns = st.columns(len(results))
                    for i, column in enumerate(columns):
                        with column:
                            curr_result = results[i]
                            st.subheader(curr_result.label)
                            st.metric(
                                "Total accuracy",
                                f"{curr_result.accuracy * 100:.2f} %",
                            )
                            st.metric(
                                "Total cost (USD)",
                                f"{curr_result.cost:.4f}",
                            )
