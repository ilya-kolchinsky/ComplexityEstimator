# ---------------------------------------------------------------------------
# Judge abstraction (LLM-as-a-judge, or anything else)
# ---------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Any, Optional


class Judge(ABC):
    """
    Judge decides whether candidate answer is correct given query + reference.
    """

    @abstractmethod
    def is_correct(self, query: str, reference: str, candidate: str) -> bool:
        ...


class SimpleStringJudge(Judge):
    """
    Extremely naive default judge: trims whitespace, does a case-insensitive
    comparison. Useful as a placeholder for datasets like MMLU where the
    answer is a single letter (A/B/C/D).
    """

    def is_correct(self, query: str, reference: str, candidate: str) -> bool:
        ref = reference.strip().lower()
        cand = candidate.strip().lower()
        return ref == cand


# If you want an LLM-as-a-judge, you can implement something like:

class LlmJudge(Judge):
    """
    Example LLM-as-a-judge, using an OpenAI-like client.

    This is intentionally generic; plug in your own client.
    """

    def __init__(
        self,
        client: Any,
        judge_model: str,
        system_prompt: Optional[str] = None,
    ) -> None:
        self._client = client
        self._judge_model = judge_model
        self._system_prompt = (
            system_prompt
            or "You are a strict evaluator that answers ONLY with 'yes' or 'no'. "
               "Given a question, a reference answer, and a candidate answer, "
               "decide if the candidate is fully correct."
        )

    def is_correct(self, query: str, reference: str, candidate: str) -> bool:
        prompt = (
            "Question:\n"
            f"{query}\n\n"
            "Reference answer:\n"
            f"{reference}\n\n"
            "Candidate answer:\n"
            f"{candidate}\n\n"
            "Is the candidate answer fully correct? Reply with exactly 'yes' or 'no'."
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]
        resp = self._client.chat.completions.create(
            model=self._judge_model,
            messages=messages,
            max_tokens=1,
            temperature=0.0,
        )
        text = (resp.choices[0].message.content or "").strip().lower()
        return text.startswith("y")


class MmluChoiceJudge(Judge):
    """
    Judge for multiple-choice datasets where the reference answer is a
    choice letter (A, B, C, D, ...), like MMLU.

    It:
      - Extracts the first alphabetic character from reference and candidate
      - Uppercases them
      - Compares equality

    This means:
      reference: "A"        -> "A"
      candidate: "A"        -> correct
      candidate: "Answer: A" -> also correct (extracts 'A')
    """

    @staticmethod
    def _extract_choice_letter(text: str) -> Optional[str]:
        if text is None:
            return None
        s = str(text)
        for ch in s:
            if ch.isalpha():
                return ch.upper()
        return None

    def is_correct(self, query: str, reference: str, candidate: str) -> bool:
        ref_letter = self._extract_choice_letter(reference)
        cand_letter = self._extract_choice_letter(candidate)

        if ref_letter is None or cand_letter is None:
            # If we can't parse one of them as a letter, treat as incorrect.
            return False

        return ref_letter == cand_letter
