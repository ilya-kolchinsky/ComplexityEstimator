import argparse
import os
from typing import List, Dict, Tuple, Optional, Callable

from dotenv import load_dotenv

from eval.experiment.base import ExperimentMetrics
from eval.experiment.cost_estimation import get_cost_per_token_from_scenario_dict
from eval.helm.helm_store import HelmLiteStore
from eval.utils.cache import JsonEvalCache
from eval.utils.config import EvalConfig, load_config
from eval.utils.judge import SimpleStringJudge, Judge
from eval.experiment.local_model import VllmLocalModel, LocalModel
from eval.utils.tokenize_prompt import make_st_tokenizer_token_counter
from eval.experiment.experiment_runner import ExperimentRunner, RouteFn
from eval.helm.mmlu_helm_store import MmluHelmStore
from routing.routing import load_complexity_model, create_single_model_route_function, \
    create_single_threshold_route_function


load_dotenv()


def run_experiment(experiment_label: str,
                   store: HelmLiteStore,
                   dataset_id: str,
                   model_ids: List[str],
                   local_models: Dict[str, LocalModel],
                   cost_per_token: Dict[str, float],
                   route: RouteFn,
                   judge: Judge,
                   log_callback: Optional[Callable[[str], None]] = None,
) -> ExperimentMetrics:
    """
    Executed a single experiment using the given parameters.
    Calculates and returns the total accuracy and cost of the run (number of tokens * cost per token).
    """

    # Persistent cache file (shared across experiments)
    eval_cache = JsonEvalCache("data/cache/local_eval_cache.json")

    runner = ExperimentRunner(
        helm_store=store,
        local_models=local_models,
        judge=judge,
        eval_cache=eval_cache,
        verbose=True,
        log_callback=log_callback
    )

    result = runner.run_dataset(dataset_id=dataset_id, model_ids=model_ids, route=route)

    # Accuracy
    overall_acc = result.total_accuracy()

    # Total cost
    log_callback("Calculating cost & accuracy metrics..")
    model, _, _ = load_complexity_model()
    token_counter = make_st_tokenizer_token_counter(model.enc.tokenizer)
    total_cost = result.total_cost(
        cost_per_token=cost_per_token,
        token_counter=token_counter,
    )

    return ExperimentMetrics(label=experiment_label, accuracy=overall_acc, cost=total_cost)


def validate_two_model_eval_config(config: EvalConfig) -> Tuple[str, LocalModel]:
    """
    If the provided evaluation configuration is valid, returns the ID of the frontier model and a wrapper object
    for accessing the local model. Raises ValueError if the configuration is invalid.
    """
    if len(config.models) != 2:
        raise ValueError("Please provide exactly two models")
    if config.models[0]["id"] == config.models[1]["id"]:
        raise ValueError("Please provide two distinct models")
    if config.models[0]["base_url"] is not None and config.models[1]["base_url"] is not None:
        raise ValueError("Two local models provided. Please provide one local and one frontier model.")
    if config.models[0]["base_url"] is None and config.models[1]["base_url"] is None:
        raise ValueError("Two frontier models provided. Please provide one local and one frontier model.")

    if config.models[0]["base_url"] is not None:
        # the first model is local
        small_model = VllmLocalModel(
            model_id=config.models[0]["id"],
            base_url=config.models[0]["base_url"],
            api_key="dummy",
        )
        return config.models[1]["id"], small_model

    # the second model is local
    small_model = VllmLocalModel(
        model_id=config.models[1]["id"],
        base_url=config.models[1]["base_url"],
        api_key="dummy",
    )
    return config.models[0]["id"], small_model


def eval_two_model_setup(config: EvalConfig, log_callback: Optional[Callable[[str], None]] = None) -> List[ExperimentMetrics]:
    """
    Run a simple evaluation scenario considering two models: a "cheap" local model and an "expensive" frontier model.
    The local model will be invoked during evaluation (unless the results are already cached) whereas for the
    frontier model HELM results will be used.
    The following experiments will be executed:
    - cheap model only;
    - expensive model only;
    - for each of the specified thresholds, route prompts with the estimated difficulty below the threshold to the
    cheap model and those above the threshold to the expensive model.
    """

    eval_root_dir = config.helm_root_dir if config.helm_root_dir else os.getenv("EVAL_ROOT_DIR")
    store = MmluHelmStore(eval_root_dir)

    model_ids = [model["id"] for model in config.models]

    cost_per_token = {}
    for model in config.models:
        model_id = model["id"]
        current_cost_per_token = model.get("cost_per_token")
        if current_cost_per_token is not None:
            # cost per token is explicitly specified
            cost_per_token[model_id] = float(current_cost_per_token)
        else:
            # should calculate the cost per token from scenario specification
            inference_scenario_config = model.get("inference_scenario")
            if inference_scenario_config is None or not isinstance(inference_scenario_config, Dict):
                raise ValueError(f"Neither cost per token nor inference scenario specified for model {model_id}")
            estimated_cost_per_token = get_cost_per_token_from_scenario_dict(inference_scenario_config)
            cost_per_token[model_id] = estimated_cost_per_token

    remote_model_id, local_model = validate_two_model_eval_config(config)
    local_model_id = local_model.model_id

    local_models = {
        local_model_id: local_model,
    }
    dataset_id = config.dataset_id
    judge = SimpleStringJudge()  # should switch to the real judge if non-multi-choice question dataset is used

    results: List[ExperimentMetrics] = []

    # Expensive model only
    label = f"{remote_model_id} only"
    route = create_single_model_route_function(remote_model_id)
    results.append(
        run_experiment(
            label, store, dataset_id, model_ids, local_models, cost_per_token, route, judge, log_callback
        )
    )

    # Cheap model only
    label = f"{local_model_id} only"
    route = create_single_model_route_function(local_model_id)
    results.append(
        run_experiment(
            label, store, dataset_id, model_ids, local_models, cost_per_token, route, judge, log_callback
        )
    )

    # Threshold experiments
    for threshold in config.binary_thresholds:
        label = f"threshold {threshold}"
        route = create_single_threshold_route_function(local_model_id, remote_model_id, threshold)
        results.append(
            run_experiment(
                label, store, dataset_id, model_ids, local_models, cost_per_token, route, judge, log_callback
            )
        )

    for result in results:
        print(f"{result.label}: accuracy {result.accuracy}, cost {result.cost}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="eval/config.yaml")
    args = ap.parse_args()

    config = load_config(args.config)
    eval_two_model_setup(config)


if __name__ == "__main__":
    main()
