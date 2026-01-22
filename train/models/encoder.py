from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizerBase, AutoConfig, PretrainedConfig

# Sentence-Transformers is optional
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    SentenceTransformer = None
    _HAS_ST = False


class HFEncoder(nn.Module):
    """
    Flexible encoder wrapper producing a single [B, H] sequence embedding.

    Pooling priority (when pooling_mode='auto'):
      1) Sentence-Transformers pooling (if enabled + available)
      2) BERT-style pooler_output (if enabled + available)
      3) Mask-aware mean pooling + LayerNorm (default fallback)

    Args:
        model_name: HF or Sentence-Transformers model name/path.
        pooling_mode: 'auto' | 'st' | 'pooler' | 'mean'
        enable_sentence_transformers: if True and sentence-transformers is installed,
            use SentenceTransformer pipeline when pooling_mode in {'auto','st'} and the
            model looks like an ST model (name contains 'sentence-transformers/' or you force 'st').
        prefer_pooler_output: if True and pooling_mode in {'auto','pooler'}, use
            outputs.pooler_output when present.
        layernorm_after_pool: apply LayerNorm after pooling (recommended).
    """
    def __init__(
        self,
        model_name: str,
        init_weights: bool = True,
        pooling_mode: str = "auto",
        enable_sentence_transformers: bool = True,
        prefer_pooler_output: bool = False,
        layernorm_after_pool: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.pooling_mode = pooling_mode
        self.enable_st = enable_sentence_transformers and _HAS_ST
        self.prefer_pooler = prefer_pooler_output

        self.is_st_candidate = (
            ("sentence-transformers/" in (model_name or "").lower())
            or self.pooling_mode == "st"
        )

        self._init_backbone(model_name, init_weights)

        hidden = self.hidden_size
        self.post_ln = nn.LayerNorm(hidden) if layernorm_after_pool else nn.Identity()
        self.out_dim = hidden

    # ---------- public API ----------

    def tokenize(self, texts, max_length: int):
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.mode == "st":
            # Sentence-Transformers forward
            # SentenceTransformer.forward expects a dict named "features"
            features = {"input_ids": input_ids, "attention_mask": attention_mask}
            out = self.st_model.forward(features)  # returns dict with 'sentence_embedding'
            emb = out["sentence_embedding"]  # [B, H]
            return self.post_ln(emb)

        # Hugging Face AutoModel forward
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        if self.mode == "pooler" and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output  # [B, H]
            return self.post_ln(emb)

        # Fallback: mask-aware mean pooling
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # [B, T, 1]
        summed = (last_hidden * mask).sum(dim=1)                   # [B, H]
        lengths = mask.sum(dim=1).clamp_min(1e-6)                  # [B, 1]
        mean_pooled = summed / lengths                             # [B, H]
        return self.post_ln(mean_pooled)

    # ---------- internals ----------

    def _init_backbone(self, model_name: str, init_weights: bool):
        """
        Decide which backend to use and initialize tokenizer/backbone accordingly.
        """
        # Prefer Sentence-Transformers if requested/allowed and available
        if self.pooling_mode in ("auto", "st") and self.enable_st and self.is_st_candidate:
            try:
                self.st_model = SentenceTransformer(model_name)
                # SentenceTransformer exposes .tokenizer via ._first_module().tokenizer in recent versions;
                # fall back to an AutoTokenizer if not found.
                try:
                    self.tokenizer = self.st_model.tokenizer
                except Exception:
                    self.tokenizer = self.load_or_download_tokenizer(model_name)
                # Determine hidden size from the pooling module or the transformer config
                self.hidden_size = self._infer_st_hidden(self.st_model)
                self.mode = "st"
                return
            except Exception:
                # Fall back to HF if ST load fails
                pass

        self.tokenizer = self.load_or_download_tokenizer(model_name)

        if init_weights:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            config = self.load_or_download_config(model_name)
            self.backbone = AutoModel.from_config(config)

        self.hidden_size = self.backbone.config.hidden_size

        # Decide pooling behavior for HF
        if self.pooling_mode == "pooler" or (
            self.pooling_mode == "auto" and self.prefer_pooler and hasattr(self.backbone.config, "pooler_fc_size")
        ):
            # We'll still check for pooler presence at forward time
            self.mode = "pooler"
        else:
            self.mode = "mean"

    @staticmethod
    def load_or_download_tokenizer(
            model_name: str,
            local_dir: Optional[str] = None,
            use_fast: bool = True,
    ) -> PreTrainedTokenizerBase:
        """
        Load a tokenizer for `model_name`, preferring a local directory.
        If the local directory does not exist or is incomplete, download
        from Hugging Face Hub, then save to the local directory for
        future runs.

        Args:
            model_name:  Hugging Face model id, e.g. "distilroberta-base"
            local_dir:   Local directory to load from / save to.
                         If None, defaults to "./router_tokenizer".
            use_fast:    Whether to prefer the fast tokenizer implementation.

        Returns:
            A Hugging Face tokenizer (PreTrainedTokenizerBase subclass).
        """
        if local_dir is None:
            local_dir = "data/hf/router_tokenizer"

        local_path = Path(local_dir)

        # 1) Try local load first (no network)
        if local_path.is_dir():
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    local_path,
                    use_fast=use_fast,
                    local_files_only=True,
                )
                return tokenizer
            except Exception as e:
                # Local dir exists but is broken/incomplete; fall back to HF.
                print(
                    f"[load_or_download_tokenizer] Failed to load tokenizer from "
                    f"{local_path} ({type(e).__name__}: {e}). Falling back to HF download."
                )

        # 2) Download from HF Hub
        print(f"Downloading tokenizer for {model_name!r} from Hugging Face Hub...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=use_fast,
        )

        # 3) Save locally for next time
        local_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(local_path)
        print(f"Saved tokenizer for {model_name!r} to {local_path}.")

        return tokenizer

    @staticmethod
    def load_or_download_config(
            model_name: str,
            local_dir: Optional[str] = None,
    ) -> PretrainedConfig:
        """
        Load a model config for `model_name`, preferring a local directory.
        If the local directory does not exist or is incomplete, download
        from Hugging Face Hub, then save to the local directory for
        future runs.

        Args:
            model_name:
                Hugging Face model id, e.g. "distilroberta-base".
            local_dir:
                Local directory to load from / save to.
                If None, defaults to "./router_config".

        Returns:
            A Hugging Face PretrainedConfig instance.
        """
        if local_dir is None:
            local_dir = "data/hf/router_config"

        local_path = Path(local_dir)

        # 1) Try local load first (no network)
        if local_path.is_dir():
            try:
                cfg = AutoConfig.from_pretrained(
                    local_path,
                    local_files_only=True,
                )
                return cfg
            except Exception as e:
                print(f"Failed to load config from {local_path} ({type(e).__name__}: {e}). Falling back to HF download.")

        # 2) Download from HF Hub
        print(f"Downloading config for {model_name!r} from Hugging Face Hub...")
        cfg = AutoConfig.from_pretrained(model_name)

        # 3) Save locally for next time
        local_path.mkdir(parents=True, exist_ok=True)
        cfg.save_pretrained(local_path)
        print(f"Saved config for {model_name!r} to {local_path}.")

        return cfg

    @staticmethod
    def _infer_st_hidden(st_model) -> int:
        # Try to infer embedding dimension from the pooling module or last transformer
        try:
            # Sentence-Transformers often has a 'modules' list: [Transformer, Pooling, ...]
            for mod in reversed(st_model.modules()):
                # We want the final sentence embedding size; the Pooling module has .word_embedding_dimension
                if hasattr(mod, "word_embedding_dimension"):
                    return int(mod.word_embedding_dimension)
        except Exception:
            pass
        # Fallback: try underlying transformer config
        try:
            return int(st_model[0].auto_model.config.hidden_size)
        except Exception:
            pass
        raise RuntimeError("Could not infer Sentence-Transformers embedding size.")
