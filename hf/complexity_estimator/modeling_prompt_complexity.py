import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from hf.complexity_estimator.configuration_prompt_complexity import PromptComplexityConfig


class PromptComplexityModel(PreTrainedModel):
    config_class = PromptComplexityConfig

    def __init__(self, config: PromptComplexityConfig):
        super().__init__(config)

        self.encoder = AutoModel.from_pretrained(config.base_model_name)
        h = self.encoder.config.hidden_size

        self.post_ln = nn.LayerNorm(h) if config.layernorm_after_pool else nn.Identity()

        if config.use_projection:
            ph = int(h * config.proj_hidden_ratio)
            self.proj = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(h, ph),
                nn.ReLU(),
                nn.Linear(ph, h),
                nn.ReLU(),
            )
        else:
            self.proj = nn.Identity()

        hidden = config.hidden if config.hidden is not None else max(h // 2, 128)

        layers = [
            nn.Dropout(config.dropout),
            nn.Linear(h, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        ]
        if config.output_sigmoid:
            layers.append(nn.Sigmoid())
        self.head = nn.Sequential(*layers)

        self.post_init()

    def _mean_pool(self, last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        return summed / denom

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled = self._mean_pool(out.last_hidden_state, attention_mask)
        pooled = self.post_ln(pooled)
        pooled = self.proj(pooled)

        scores = self.head(pooled).squeeze(-1)  # [B] in [0,1]

        loss = None
        if labels is not None:
            labels = labels.to(scores.dtype).view(-1)
            loss = torch.nn.functional.mse_loss(scores, labels)

        # We’ll store scores inside logits for compatibility (shape [B,1]).
        return SequenceClassifierOutput(loss=loss, logits=scores.unsqueeze(-1))

    @torch.no_grad()
    def predict(self, texts, tokenizer, device=None):
        if isinstance(texts, str):
            texts = [texts]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        if device is not None:
            self.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
        self.eval()
        scores = self(**inputs).logits.squeeze(-1)
        out = scores.detach().cpu().tolist()
        return out[0] if len(out) == 1 else out
