from typing import Callable, Dict, Any

TokenCounter = Callable[[str, str], int]  # (text, model_id) -> num_tokens


def whitespace_token_counter(text: str, model_id: str) -> int:
    """
    Very simple token counter: splits on whitespace.

    This is NOT a good approximation of real tokenizer behavior, but it is:
    - dependency-free
    - deterministic
    - good enough for rough relative comparisons if you keep it consistent.
    """
    return 0 if not text else len(text.split())


def make_tiktoken_token_counter(
    default_model_name: str = "gpt-4o",
) -> TokenCounter:
    """
    Create a token_counter(text, model_id) that uses tiktoken.

    - If tiktoken has a dedicated encoding for the given model_id, we use it.
    - Otherwise we fall back to a default model name (e.g., gpt-4o).

    Requires: `pip install tiktoken`
    """

    try:
        import tiktoken  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "tiktoken is not installed. Install it with `pip install tiktoken` "
            "or use whitespace_token_counter instead."
        ) from exc

    # Cache encodings per model_id
    enc_cache: Dict[str, Any] = {}

    def get_encoding_for_model(model_name: str):
        if model_name in enc_cache:
            return enc_cache[model_name]
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except Exception:
            enc = tiktoken.encoding_for_model(default_model_name)
        enc_cache[model_name] = enc
        return enc

    def token_counter(text: str, model_id: str) -> int:
        enc = get_encoding_for_model(model_id)
        return len(enc.encode(text))

    return token_counter


def make_st_tokenizer_token_counter(tokenizer: Callable) -> TokenCounter:
    """
    Use the Sequence Transformer tokenizer to estimate token counts for *all* models.
    """

    def token_counter(text: str, model_id: str) -> int:
        enc = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        # enc["input_ids"] is a list (or list of lists) depending on batching
        input_ids = enc["input_ids"]
        if isinstance(input_ids[0], list):  # batched case
            return len(input_ids[0])
        return len(input_ids)

    return token_counter
