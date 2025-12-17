# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any


@dataclass
class Example:
    """
    A single evaluation example.

    IMPORTANT:
    - `query` is the **full prompt** used for the LLM request (taken from scenario_state.request.prompt),
      not just the bare question text.
    - The original question text is stored in metadata["question_text"].
    """
    id: str
    query: str                    # full prompt sent to the model
    reference_answer: str         # expected output used for correctness checking
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResult:
    """
    Result for one (example, model) pair.
    """
    example_id: str
    model_id: str
    response: str
    is_correct: Optional[bool]
    # "helm" means taken directly from HELM data;
    # "local"        -> freshly computed via local model + judge
    # "local_cache"  -> correctness loaded from local cache, response is dummy
    source: str


@dataclass
class InstanceEval:
    """Evaluation result for a single example routed to exactly one model."""
    example: Example
    model_result: ModelResult


@dataclass
class ExperimentResult:
    """Collection of per-example evaluations for a dataset."""
    dataset_id: str
    instance_evals: List[InstanceEval]

    # -------------------------------------------------------------
    # 1) TOTAL accuracy (across all models and all examples)
    # -------------------------------------------------------------
    def total_accuracy(self) -> float:
        """
        Overall accuracy across all InstanceEval entries.

        Only counts ModelResults where is_correct is not None.
        """
        num_correct = 0
        num_total = 0

        for ie in self.instance_evals:
            r = ie.model_result
            if r.is_correct is None:
                continue
            num_total += 1
            if r.is_correct:
                num_correct += 1

        if num_total == 0:
            return 0.0

        return num_correct / num_total

    # -------------------------------------------------------------
    # 2) TOTAL cost using a token counter + cost_per_token dict
    # -------------------------------------------------------------
    def total_cost(
        self,
        cost_per_token: Dict[str, float],
        token_counter: Callable[[str, str], int],
    ) -> float:
        """
        Compute total cost of the experiment.

        Args:
            cost_per_token:
                Mapping model_id -> cost per token (float).
                Should include all model_ids that appear in model_result.model_id.
            token_counter:
                Function token_counter(text, model_id) -> int
                that returns the number of tokens for the input text with
                respect to that model.

        Returns:
            Total cost = sum over examples of (num_tokens * cost_per_token[model_id])

        Raises:
            KeyError if a model_id is missing in cost_per_token.
        """
        total = 0.0

        for ie in self.instance_evals:
            r = ie.model_result
            model_id = r.model_id

            if model_id not in cost_per_token:
                raise KeyError(
                    f"Missing cost_per_token entry for model_id={model_id!r}"
                )

            tokens = token_counter(ie.example.query, model_id)
            total += tokens * cost_per_token[model_id]

        return total

    # (existing helper, still useful)
    def per_model_accuracy(self) -> Dict[str, float]:
        """
        Convenience helper: compute accuracy per model_id.

        Only counts ModelResults where is_correct is not None.
        """
        totals: Dict[str, int] = {}
        corrects: Dict[str, int] = {}

        for ie in self.instance_evals:
            r = ie.model_result
            if r.is_correct is None:
                continue
            totals[r.model_id] = totals.get(r.model_id, 0) + 1
            if r.is_correct:
                corrects[r.model_id] = corrects.get(r.model_id, 0) + 1

        return {
            mid: (corrects.get(mid, 0) / totals[mid]) if totals[mid] > 0 else 0.0
            for mid in totals
        }
