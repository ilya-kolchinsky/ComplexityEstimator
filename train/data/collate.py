from dataclasses import dataclass
from typing import List
import torch


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def make_collate():
    def collate_fn(examples: List[dict]):
        input_ids = torch.stack([e["input_ids"] for e in examples])         # [B, T]
        attention_mask = torch.stack([e["attention_mask"] for e in examples])  # [B, T]
        labels = torch.tensor([e["label"] for e in examples], dtype=torch.float32)  # [B]
        return Batch(input_ids=input_ids.long(), attention_mask=attention_mask.long(), labels=labels)
    return collate_fn
