# ---------------------------------------------------------------------------
# Local model abstraction + vLLM implementation
# ---------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Any


class LocalModel(ABC):
    """
    Abstract local model that we can actually query.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Synchronous one-shot generation for a single prompt.
        """
        ...


class VllmLocalModel(LocalModel):
    """
    LocalModel wrapper around a vLLM OpenAI-compatible endpoint.

    vLLM exposes an OpenAI-style HTTP API, e.g.:

        python -m vllm.entrypoints.openai.api_server \\
            --model /path/to/model \\
            --port 8000

    Then you can talk to it using the official OpenAI Python client by
    providing base_url and an arbitrary api_key.

    Example usage:

        from helm_experiment_runner import VllmLocalModel

        small_model = VllmLocalModel(
            model_id="mistral-7b-instruct",
            base_url="http://localhost:8000/v1",
            api_key="not-used-but-required",
        )
    """

    def __init__(
        self,
        model_id: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "dummy-key",
        max_tokens: int = None,
        temperature: float = 0.0,
        client: Any = None,
    ) -> None:
        super().__init__(model_id=model_id)

        # Lazy import so this file can be imported without openai installed
        if client is None:
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:  # pragma: no cover - dependency issue
                raise RuntimeError(
                    "openai Python package is required for VllmLocalModel. "
                    "Install it with `pip install openai`."
                ) from exc

            client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)

        self._client = client
        self._max_tokens = max_tokens
        self._temperature = temperature

    def generate(self, prompt: str, **kwargs: Any) -> str:
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        if max_tokens is not None:
            max_tokens = int(max_tokens)
        temperature = float(kwargs.get("temperature", self._temperature))

        # We use the Chat Completions API; vLLM supports it for most models.
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = resp.choices[0].message.content
        return content or ""
