import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from eval.experiment.base import Example
from eval.helm.helm_store import HelmLiteStore


class MmluHelmStore(HelmLiteStore):
    """
    HELM-Lite store specialized for MMLU (multiple_choice_joint).

    Assumes local directory structure:

        root_dir/
          <dataset_id>/
            <model_id>_scenario_state.json
            <other_model_id>_scenario_state.json
            ...

    Where each `*_scenario_state.json` is a serialized ScenarioState containing:
      - adapter_spec
      - request_states: [
          {
            "instance": {
              "id": <example_id>,
              "input": {"text": <question text>},
              "references": [
                 {"output": {"text": <choice_text>}, "tags": [...]},
                 ...
              ],
              ...
            },
            "output_mapping": {"A": <choice_text>, "B": <choice_text>, ...},
            "request": {"prompt": <full prompt string>, ...},
            "result": {
               "completions": [
                  {"text": <model_output>, ...},
                  ...
               ],
               ...
            },
            ...
          },
          ...
        ]

    For MMLU, the model is instructed to answer with a single choice index (A/B/C/D).
    We therefore:
      - Use `request.prompt` as the `Example.query` (for local model evaluation).
      - Derive the **correct choice letter** from `references` + `output_mapping`.
      - Derive HELM model correctness by comparing the model's letter (from completions[0].text)
        against the correct letter.
    """

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()

        # Cache: dataset_id -> List[Example]
        self._examples_cache: Dict[str, List[Example]] = {}
        # Cache: dataset_id -> {example_id -> Example}
        self._example_index: Dict[str, Dict[str, Example]] = {}
        # Cache: (dataset_id, model_id) -> parsed scenario_state dict
        self._scenario_state_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dataset_dir(self, dataset_id: str) -> Path:
        return self.root_dir / dataset_id

    @staticmethod
    def _model_id_from_filename(filename: str) -> str:
        suffix = "_scenario_state.json"
        if not filename.endswith(suffix):
            raise ValueError(
                f"Unexpected scenario_state filename {filename!r}; "
                f"expected it to end with {suffix!r}"
            )
        return filename[: -len(suffix)]

    def _load_scenario_state(self, dataset_id: str, model_id: str) -> dict[str, Any] | None:
        """
        Load and cache scenario_state.json for (dataset_id, model_id).
        """
        key = (dataset_id, model_id)
        if key in self._scenario_state_cache:
            return self._scenario_state_cache[key]

        dataset_dir = self._dataset_dir(dataset_id)
        path = dataset_dir / f"{model_id}_scenario_state.json"
        if not path.is_file():
            # Scenario state file not found for this model
            return None

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if "request_states" not in data:
            raise ValueError(
                f"scenario_state.json for dataset={dataset_id!r}, model={model_id!r} "
                f"does not contain 'request_states'"
            )

        self._scenario_state_cache[key] = data
        return data

    def _ensure_examples_loaded(self, dataset_id: str) -> None:
        """
        Build Examples for a dataset from one canonical scenario_state.json
        (we assume all models used the same instances & prompts).
        """
        if dataset_id in self._examples_cache:
            return

        # Use the first available model file as canonical for instances/prompts.
        model_ids = sorted(self.get_helm_models(dataset_id))
        if not model_ids:
            raise FileNotFoundError(
                f"No scenario_state.json files found under dataset {dataset_id!r} "
                f"in root_dir={self.root_dir!s}"
            )

        canonical_model = model_ids[0]
        scenario_state = self._load_scenario_state(dataset_id, canonical_model)

        instructions = scenario_state["adapter_spec"]["instructions"]
        request_states = scenario_state["request_states"]

        examples: List[Example] = []
        index: Dict[str, Example] = {}

        for rs in request_states:
            instance = rs.get("instance", {})
            instance_id = instance.get("id")
            if instance_id is None:
                # Skip malformed entries
                continue

            correct_letter = self._extract_correct_choice_letter(rs)

            input_obj = instance.get("input", {})
            question_text = input_obj.get("text", "")
            output_mapping = rs.get("output_mapping", {})

            prompt = self._compile_closed_form_question(question_text, output_mapping)
            if not isinstance(prompt, str):
                prompt = str(prompt)
            prompt_with_instructions = f"{instructions}\n{prompt}"

            ex = Example(
                id=instance_id,
                query=prompt,
                query_with_instructions=prompt_with_instructions,
                # For MMLU, the reference answer is the choice index (e.g., "A")
                reference_answer=correct_letter or "",
                metadata={
                    "question_text": question_text,
                    "split": instance.get("split"),
                    "output_mapping": output_mapping,
                },
            )

            examples.append(ex)
            index[instance_id] = ex

        self._examples_cache[dataset_id] = examples
        self._example_index[dataset_id] = index

    @staticmethod
    def _compile_closed_form_question(question: str, answers: dict[str, str]) -> str:
        lines = [question]

        for key in sorted(answers.keys()):
            lines.append(f"{key}. {answers[key]}")

        return "\n".join(lines)

    @staticmethod
    def _extract_correct_choice_letter(request_state: Dict[str, Any]) -> Optional[str]:
        """
        For a single request_state, derive the correct multiple-choice letter.

        Uses:
          - instance.references[*].tags (looks for "correct")
          - output_mapping: { "A": choice_text, ... }
        """
        instance = request_state.get("instance", {})
        references = instance.get("references", []) or []
        output_mapping = request_state.get("output_mapping", {}) or {}

        # Find the correct reference text.
        correct_text: Optional[str] = None
        for ref in references:
            tags = ref.get("tags", []) or []
            if "correct" in tags:
                correct_text = ref.get("output", {}).get("text")
                if correct_text is not None:
                    break

        if correct_text is None:
            return None

        correct_text_norm = str(correct_text).strip()

        # Map back from reference text to letter using output_mapping.
        for letter, text in output_mapping.items():
            if str(text).strip() == correct_text_norm:
                letter_str = str(letter).strip()
                if not letter_str:
                    return None
                return letter_str[0].upper()

        # If we can't map, fall back to None (caller can decide what to do).
        return None

    @staticmethod
    def _extract_model_choice(
        request_state: Dict[str, Any],
    ) -> Tuple[str, Optional[str]]:
        """
        Extract the model's raw completion text and inferred choice letter.

        For MMLU multiple_choice_joint, completions[0].text is usually "A", "B", "C", or "D".
        We:
          - return the raw text
          - also derive the first alphabetic character as uppercase letter (if any)
        """
        result = request_state.get("result", {}) or {}
        completions = result.get("completions") or []
        if not completions:
            return "", None

        raw_text = str(completions[0].get("text", ""))
        s = raw_text.strip()

        letter: Optional[str] = None
        for ch in s:
            if ch.isalpha():
                letter = ch.upper()
                break

        return raw_text, letter

    # ------------------------------------------------------------------
    # HelmLiteStore API implementation
    # ------------------------------------------------------------------

    def load_dataset(self, dataset_id: str) -> List[Example]:
        """
        Load all MMLU examples for the given dataset_id.

        - Example.query is the full `request.prompt` (what you should send to a local model).
        - Example.reference_answer is the correct choice letter (A/B/C/…).
        - question_text (pure question without prefix/options) is kept in metadata["question_text"].
        """
        self._ensure_examples_loaded(dataset_id)
        # Return a copy to avoid accidental external mutation.
        return list(self._examples_cache[dataset_id])

    def get_reference_answer(self, dataset_id: str, example_id: str) -> str:
        """
        Return the correct choice letter (e.g., "A") for a given example.
        """
        self._ensure_examples_loaded(dataset_id)
        try:
            ex = self._example_index[dataset_id][example_id]
        except KeyError as exc:
            raise KeyError(
                f"Example with id={example_id!r} not found in dataset {dataset_id!r}"
            ) from exc
        return ex.reference_answer

    def get_helm_models(self, dataset_id: str) -> Set[str]:
        """
        List all models that have HELM MMLU runs for the given dataset.

        Implementation: look for `<model_id>_scenario_state.json` under root_dir/dataset_id.
        """
        dataset_dir = self._dataset_dir(dataset_id)
        if not dataset_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset directory {dataset_dir!s} does not exist "
                f"(dataset_id={dataset_id!r})"
            )

        model_ids: Set[str] = set()
        for path in dataset_dir.glob("*_scenario_state.json"):
            model_ids.add(self._model_id_from_filename(path.name))

        return model_ids

    def get_helm_answer(
        self,
        dataset_id: str,
        model_id: str,
        example_id: str,
    ) -> Optional[Tuple[str, Optional[bool]]]:
        """
        Return (raw_model_output, is_correct) for a HELM model on MMLU.

        - raw_model_output: completions[0].text from scenario_state.json.
        - is_correct: True/False if we can derive correctness (choice letter matches),
                      None if the instance isn't found.
        """
        scenario_state = self._load_scenario_state(dataset_id, model_id)
        if scenario_state is None:
            # this model has no HELM data; we will have to resort to actual evaluation
            return None

        for rs in scenario_state["request_states"]:
            instance = rs.get("instance", {})
            if instance.get("id") != example_id:
                continue

            raw_output, model_letter = self._extract_model_choice(rs)
            correct_letter = self._extract_correct_choice_letter(rs)

            is_correct: Optional[bool] = None
            if model_letter is not None and correct_letter is not None:
                is_correct = (model_letter == correct_letter)

            return raw_output, is_correct

        # No request_state with that example_id for this model.
        return None
