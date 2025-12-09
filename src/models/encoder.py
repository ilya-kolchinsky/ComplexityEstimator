import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

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

        self._init_backbone(model_name)

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

    def _init_backbone(self, model_name: str):
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
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                # Determine hidden size from the pooling module or the transformer config
                self.hidden_size = self._infer_st_hidden(self.st_model)
                self.mode = "st"
                return
            except Exception:
                # Fall back to HF if ST load fails
                pass

        # Hugging Face AutoModel path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.backbone = AutoModel.from_pretrained(model_name)
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
