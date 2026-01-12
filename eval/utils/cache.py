from abc import ABC, abstractmethod
from typing import Optional
import json
from pathlib import Path
import os


# ---------------------------------------------------------------------------
# Evaluation cache: only stores correctness, not responses
# ---------------------------------------------------------------------------

class EvalCache(ABC):
    """
    Persistent cache for local evaluations.

    We only store a boolean correctness flag for each
    (dataset_id, example_id, model_id) triple. Responses are *not* stored.
    """

    @abstractmethod
    def get_is_correct(
        self,
        dataset_id: str,
        example_id: str,
        model_id: str,
    ) -> Optional[bool]:
        """
        Return cached correctness if available, else None.
        """
        ...

    @abstractmethod
    def set_is_correct(
        self,
        dataset_id: str,
        example_id: str,
        model_id: str,
        is_correct: bool,
    ) -> None:
        """
        Store correctness flag in the cache (in memory).
        """
        ...

    @abstractmethod
    def flush(self) -> None:
        """
        Persist in-memory cache to disk.
        """
        ...


class JsonEvalCache(EvalCache):
    """
    Simple JSON-based correctness cache.

    File format (one JSON object):

        {
          "dataset_id": {
            "model_id": {
              "example_id": true/false
            }
          }
        }

    Only correctness is stored; we never store the local model's response.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._data: dict[str, dict[str, dict[str, bool]]] = {}
        self._loaded = False
        self._dirty = False  # track whether we changed anything since last load/flush

    # ---------------- internal helpers ----------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.path.is_file():
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    self._data = raw
            except Exception:
                # Corrupt or unreadable file -> start fresh
                self._data = {}
        else:
            self._data = {}
        self._loaded = True
        self._dirty = False

    def _ensure_dataset_model(self, dataset_id: str, model_id: str) -> dict[str, bool]:
        self._ensure_loaded()
        ds = self._data.setdefault(dataset_id, {})
        return ds.setdefault(model_id, {})

    # ---------------- public API ----------------

    def get_is_correct(
        self,
        dataset_id: str,
        example_id: str,
        model_id: str,
    ) -> Optional[bool]:
        self._ensure_loaded()
        return (
            self._data.get(dataset_id, {})
            .get(model_id, {})
            .get(example_id)
        )

    def set_is_correct(
        self,
        dataset_id: str,
        example_id: str,
        model_id: str,
        is_correct: bool,
    ) -> None:
        bucket = self._ensure_dataset_model(dataset_id, model_id)
        prev = bucket.get(example_id)
        new_val = bool(is_correct)
        if prev != new_val:
            bucket[example_id] = new_val
            self._dirty = True

    def flush(self) -> None:
        """
        Write to disk. If nothing changed, do nothing.

        Uses a simple temp-file + os.replace for atomic-ish writes.
        """
        self._ensure_loaded()
        if not self._dirty:
            return  # nothing changed, no need to touch disk

        self.path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

        # Atomic replace on most platforms
        os.replace(tmp_path, self.path)
        self._dirty = False
