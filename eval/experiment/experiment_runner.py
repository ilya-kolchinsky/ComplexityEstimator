from typing import (
    Dict,
    Iterable,
    List,
    Optional,
)

from eval.experiment.base import Example, ExperimentResult, InstanceEval, ModelResult
from eval.utils.cache import EvalCache
from eval.helm.helm_store import HelmLiteStore
from eval.utils.judge import Judge, MmluChoiceJudge
from eval.experiment.local_model import LocalModel
from routing.routing import RouteFn


class ExperimentRunner:
    """
    Orchestrates the simplified logic with mandatory routing and optional caching.

    Features:
      - HELM data for frontier models via HelmLiteStore
      - Local models via LocalModel + Judge
      - Optional persistent correctness cache via EvalCache
      - Automatic use of MmluChoiceJudge for multiple-choice datasets (e.g. MMLU)
      - Optional progress prints via `verbose`
    """

    def __init__(
        self,
        helm_store: HelmLiteStore,
        local_models: Dict[str, LocalModel],
        judge: Judge,
        eval_cache: Optional[EvalCache] = None,
        verbose: bool = False,
    ) -> None:
        self.helm_store = helm_store
        self.local_models = local_models
        self.judge = judge
        self.eval_cache = eval_cache
        self.verbose = verbose

    # -------------------------------------------------------------
    # Helper: detect multiple-choice datasets (MMLU-style)
    # -------------------------------------------------------------
    @staticmethod
    def _is_multiple_choice_dataset(examples: List[Example]) -> bool:
        """
        Heuristic: treat dataset as multiple-choice if ALL examples have
        a reference_answer that looks like a single choice letter (A-Z).

        This covers MMLU and similar tasks where reference_answer is "A"/"B"/...
        """
        def is_choice_letter(ans: str) -> bool:
            if not ans:
                return False
            ans = str(ans).strip()
            letters = [ch for ch in ans if ch.isalpha()]
            if len(letters) != 1:
                return False
            return letters[0].isalpha()

        if not examples:
            return False

        return all(is_choice_letter(ex.reference_answer) for ex in examples)

    # -------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------
    def run_dataset(
        self,
        dataset_id: str,
        model_ids: Iterable[str],
        route: RouteFn,
    ) -> ExperimentResult:
        """
        Evaluate all examples in `dataset_id`, routing each example to exactly
        one model chosen by `route(example)`.

        Behavior:
          - HELM data: use (response, is_correct) from scenario_state.json.
          - Local models on *multiple-choice datasets* (e.g. MMLU):
               -> use MmluChoiceJudge (no LLM judge needed).
          - Local models on other datasets:
               -> use the generic Judge passed to the ExperimentRunner.

        If eval_cache is provided, local (model, judge) evaluation for a given
        (dataset_id, example_id, model_id) triple will only be done once; later
        runs reuse the cached correctness (source="local_cache").
        """
        examples = self.helm_store.load_dataset(dataset_id)
        allowed_models = set(model_ids)
        num_examples = len(examples)

        # Detect if this is a multiple-choice dataset (like MMLU)
        is_mc_dataset = self._is_multiple_choice_dataset(examples)
        mc_judge = MmluChoiceJudge() if is_mc_dataset else None

        if self.verbose:
            print(
                f"[ExperimentRunner] Starting evaluation for dataset={dataset_id!r} "
                f"with {num_examples} examples and models={sorted(allowed_models)!r}."
            )
            if is_mc_dataset:
                print("[ExperimentRunner] Detected multiple-choice dataset; "
                      "using MmluChoiceJudge for local evaluations.")
            else:
                print("[ExperimentRunner] Non-multiple-choice dataset; "
                      "using provided Judge for local evaluations.")

        instance_evals: List[InstanceEval] = []

        for idx, ex in enumerate(examples, start=1):
            if self.verbose and (
                idx == 1 or idx % 50 == 0 or idx == num_examples
            ):
                print(
                    f"[ExperimentRunner] Processing example {idx}/{num_examples} "
                    f"(id={ex.id!r})..."
                )

            # Decide which model to use for this example.
            chosen_model = route(ex.query)

            if chosen_model not in allowed_models:
                raise ValueError(
                    f"Route function returned model_id {chosen_model!r} for example "
                    f"{ex.id}, but it is not in allowed model_ids={sorted(allowed_models)!r}"
                )

            # 1. Try HELM data first
            helm_answer = self.helm_store.get_helm_answer(
                dataset_id=dataset_id,
                model_id=chosen_model,
                example_id=ex.id,
            )

            if helm_answer is not None:
                response, is_correct = helm_answer
                if self.verbose:
                    print(
                        f"[ExperimentRunner] Example {ex.id!r} routed to model "
                        f"{chosen_model!r}: using HELM prediction (no local inference)."
                    )
                model_result = ModelResult(
                    example_id=ex.id,
                    model_id=chosen_model,
                    response=response,
                    is_correct=is_correct,
                    source="helm",
                )
            else:
                # 2. Not in HELM: try local cache, then local model + judge
                cached_correct: Optional[bool] = None
                if self.eval_cache is not None:
                    cached_correct = self.eval_cache.get_is_correct(
                        dataset_id=dataset_id,
                        example_id=ex.id,
                        model_id=chosen_model,
                    )

                if cached_correct is not None:
                    if self.verbose:
                        print(
                            f"[ExperimentRunner] Example {ex.id!r} routed to model "
                            f"{chosen_model!r}: using cached correctness "
                            f"(no local inference)."
                        )
                    model_result = ModelResult(
                        example_id=ex.id,
                        model_id=chosen_model,
                        response="<cached>",
                        is_correct=cached_correct,
                        source="local_cache",
                    )
                else:
                    # Need to actually run the local model and judge
                    local_model = self.local_models.get(chosen_model)
                    if local_model is None:
                        raise ValueError(
                            f"Model {chosen_model!r} for example {ex.id} is not available "
                            f"in HELM and no LocalModel was provided for it."
                        )

                    if self.verbose:
                        print(
                            f"[ExperimentRunner] Example {ex.id!r} routed to model "
                            f"{chosen_model!r}: running local model + judge..."
                        )

                    candidate = local_model.generate(ex.query)
                    ref = self.helm_store.get_reference_answer(dataset_id, ex.id)

                    if is_mc_dataset and mc_judge is not None:
                        # MMLU-style multiple choice: use choice-letter judge
                        is_correct = mc_judge.is_correct(ex.query, ref, candidate)
                    else:
                        # Generic dataset: use the Judge supplied to ExperimentRunner
                        is_correct = self.judge.is_correct(ex.query, ref, candidate)

                    # Store in cache (correctness only)
                    if self.eval_cache is not None:
                        self.eval_cache.set_is_correct(
                            dataset_id=dataset_id,
                            example_id=ex.id,
                            model_id=chosen_model,
                            is_correct=is_correct,
                        )

                    model_result = ModelResult(
                        example_id=ex.id,
                        model_id=chosen_model,
                        response=candidate,
                        is_correct=is_correct,
                        source="local",
                    )

            instance_evals.append(
                InstanceEval(
                    example=ex,
                    model_result=model_result,
                )
            )

        # Persist cache at the end (if present)
        if self.eval_cache is not None:
            if self.verbose:
                print("[ExperimentRunner] Flushing evaluation cache to disk...")
            self.eval_cache.flush()
            if self.verbose:
                print("[ExperimentRunner] Cache flush complete.")

        if self.verbose:
            print(
                f"[ExperimentRunner] Finished evaluation for dataset={dataset_id!r}. "
                f"Processed {num_examples} examples."
            )

        return ExperimentResult(dataset_id=dataset_id, instance_evals=instance_evals)
