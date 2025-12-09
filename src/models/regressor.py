import torch
import torch.nn as nn


class Regressor(nn.Module):
    """
    Prompt complexity estimation regression head:
      - By default: a single 2-layer MLP head (dropout + Linear→ReLU→Linear→Sigmoid).
      - Optionally: prepend a projection MLP if you later want more capacity.

    Arguments:
      enc: HFEncoder instance
      hidden: width of the main MLP hidden layer (default: enc.out_dim // 2)
      dropout: dropout rate
      use_projection: if True, add a lightweight projection block before the head
      proj_hidden_ratio: width multiplier for projection (only used if use_projection=True)
    """
    def __init__(
        self,
        enc,
        hidden: int = None,
        dropout: float = 0.1,
        use_projection: bool = False,
        proj_hidden_ratio: float = 1.0,
    ):
        super().__init__()
        self.enc = enc
        h = enc.out_dim
        if hidden is None:
            hidden = max(h // 2, 128)  # reasonable default

        # Optional projection block (off by default)
        if use_projection:
            ph = int(h * proj_hidden_ratio)
            self.proj = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(h, ph),
                nn.ReLU(),
                nn.Linear(ph, h),
                nn.ReLU(),
            )
        else:
            self.proj = nn.Identity()

        # Main prediction head (2 linear layers total)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(h, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        z = self.enc(input_ids, attention_mask)  # [B, H]
        z = self.proj(z)                         # [B, H]
        y = self.head(z).squeeze(-1)             # [B]
        return y
