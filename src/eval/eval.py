from typing import List

from dotenv import load_dotenv

from experiment_runner import ExperimentRunner, RouteFn
from mmlu_helm_store import MmluHelmStore
from src.eval.base import ExperimentResult, Example
from src.eval.cache import JsonEvalCache
from src.eval.judge import SimpleStringJudge
from src.eval.local_model import VllmLocalModel
from src.eval.tokenize_prompt import make_st_tokenizer_token_counter
from src.routing import load_complexity_model, estimate_complexity, route_to_model, RoutingRule

load_dotenv()


def create_single_model_route_function(model_id: str) -> RouteFn:
    return lambda e: model_id


def create_complexity_model_route_function(rules: List[RoutingRule]) -> RouteFn:
    model, device, max_length = load_complexity_model()

    def route(example: Example):
        prompt = example.query
        complexity = estimate_complexity(model, device, max_length, prompt)
        routed_rule = route_to_model(rules, complexity)
        return routed_rule.model_name

    return route


def create_token_counter():
    model, _, _ = load_complexity_model()
    return make_st_tokenizer_token_counter(model.enc.tokenizer)


def main():

    store = MmluHelmStore("data/eval")

    # vLLM-served small model
    small_model = VllmLocalModel(
        model_id="granite32-8b",
        base_url="https://granite32-8b-llama-serve.apps.ocp-beta-test.nerc.mghpcc.org/v1",
        api_key="dummy",
    )

    local_models = {
        small_model.model_id: small_model,
    }

    # Persistent cache file (shared across experiments)
    eval_cache = JsonEvalCache("data/cache/local_eval_cache.json")

    runner = ExperimentRunner(
        helm_store=store,
        local_models=local_models,
        judge=SimpleStringJudge(),  # no real judge is needed for multi-choice question datasets
        eval_cache=eval_cache,
        verbose=True,
    )

    # route = create_single_model_route_function("claude_3.5_sonnet")

    threshold = 0.5
    rules = [
        RoutingRule(model_name="claude_3.5_sonnet", lower=threshold, upper=1.0),
        RoutingRule(model_name="granite32-8b", lower=0.0, upper=threshold),
    ]
    route = create_complexity_model_route_function(rules)

    result: ExperimentResult = runner.run_dataset(
        dataset_id="mmlu",
        model_ids=["claude_3.5_sonnet", "granite32-8b"],
        route=route,
    )

    # 1) Total accuracy:
    overall_acc = result.total_accuracy()
    print("Total accuracy:", overall_acc)

    # 2) Total cost with a simple whitespace token counter:
    cost_per_token = {
        "claude_3.5_sonnet": 1.0,
        "granite32-8b": 0.1,
    }

    token_counter = create_token_counter()

    total_cost = result.total_cost(
        cost_per_token=cost_per_token,
        token_counter=token_counter,
    )
    print("Total cost:", total_cost)


if __name__ == "__main__":
    main()
