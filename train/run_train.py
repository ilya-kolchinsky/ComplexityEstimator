import argparse
import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softplus
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from data.collate import make_collate
from models.encoder import HFEncoder
from models.regressor import Regressor
from train.utils.config import load_config
from train.utils.sampler import BalancedRatioBatchSampler
from train.utils.utils import mae, spearman, get_device, set_seed
from train.utils.logging import Logger


class CSVRegDataset(Dataset):
    def __init__(self, df, enc, max_len, text_col, label_col):
        self.df = df.reset_index(drop=True)
        self.enc = enc
        self.max_len = max_len
        self.text_col = text_col
        self.label_col = label_col

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        toks = self.enc.tokenize([str(row[self.text_col])], self.max_len)
        item = {k: v[0] for k, v in toks.items()}
        item["label"] = float(row[self.label_col])
        return item


def pairwise_rank_loss(preds: torch.Tensor, targets: torch.Tensor, num_pairs: int = 256):
    """
    Sampled logistic pairwise ranking loss, promoting correct ordering.
    preds, targets: [B]
    """
    b = preds.size(0)
    if b < 2:
        return preds.new_tensor(0.0)
    # sample pairs (i,j) with y_i != y_j
    i = torch.randint(0, b, (num_pairs,), device=preds.device)
    j = torch.randint(0, b, (num_pairs,), device=preds.device)
    mask = (targets[i] > targets[j])  # True means i should rank above j
    if mask.sum() == 0:
        return preds.new_tensor(0.0)
    s = preds[i] - preds[j]
    # For i>j we want s large; for i<j we flip sign
    s = torch.where(mask, s, -s)
    return softplus(-s).mean()  # log(1+exp(-s))


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def unfreeze_top_k_layers(model, k: int):
    # DeBERTa-like: encoder.layer.<idx>.*  (largest idx = top)
    all_layers = [m for n, m in model.enc.named_modules() if "encoder.layer." in n and n.endswith(".output")]
    num_layers = len(all_layers)
    top_idxs = set(range(num_layers - k, num_layers))
    for n, p in model.enc.named_parameters():
        # parse layer index from name
        if "encoder.layer." in n:
            idx = int(n.split("encoder.layer.")[1].split(".")[0])
            p.requires_grad = (idx in top_idxs)
        else:
            # non-layer params (embeddings, pooler) stay frozen
            p.requires_grad = False


def train_epoch(model, loader, optim, sched, device, loss, rank_lambda):
    model.train()

    if loss == "mse":
        ls = torch.nn.MSELoss()
    elif loss == "huber":
        ls = torch.nn.SmoothL1Loss(beta=0.1)
    else:
        raise ValueError(f"Unsupported loss: {loss}")

    losses = []
    for step, batch in enumerate(tqdm(loader, desc="train")):
        for k in ["input_ids", "attention_mask", "labels"]:
            batch.__dict__[k] = batch.__dict__[k].to(device)
        pred = model(batch.input_ids, batch.attention_mask)
        main_loss = ls(pred, batch.labels)

        if rank_lambda > 0.0:
            final_loss = main_loss + (rank_lambda * pairwise_rank_loss(pred, batch.labels))
        else:
            final_loss = main_loss

        final_loss.backward()

        optim.step()
        optim.zero_grad(set_to_none=True)
        if sched is not None:
            sched.step()
        losses.append(final_loss.item())
        # free GPU memory of batch
        del batch
        gc.collect()
        if device == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()

    # evaluate metrics on the training set
    training_metrics = eval_epoch(model, loader, device, "train-eval")

    return float(np.mean(losses)), training_metrics


@torch.no_grad()
def eval_epoch(model, loader, device, label="eval"):
    model.eval()
    y_true, y_pred = [], []
    for batch in tqdm(loader, desc=label):
        for k in ["input_ids", "attention_mask", "labels"]:
            batch.__dict__[k] = batch.__dict__[k].to(device)
        p = model(batch.input_ids, batch.attention_mask)
        y_true.extend(batch.labels.detach().cpu().tolist())
        y_pred.extend(p.detach().cpu().tolist())
        del batch
        gc.collect()
        if device == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
    return {"mae": mae(y_true, y_pred), "spearman": spearman(y_true, y_pred)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="train/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    device = get_device(cfg)
    print(f"Using device: {device}")

    os.makedirs(cfg.logging["out_dir"], exist_ok=True)
    logger = Logger(cfg.logging["out_dir"])

    enc = HFEncoder(cfg.model["name"])
    model = Regressor(enc, dropout=cfg.model["dropout"]).to(device)

    # Optionally freeze full encoder
    if cfg.train.get("freeze_encoder", False):
        set_requires_grad(model.enc, False)
    elif cfg.train.get("encoder_unfrozen_layers"):
        unfreeze_top_k_layers(model, k=cfg.train["encoder_unfrozen_layers"])

    # data
    train_df = pd.read_csv(cfg.data["train_csv"])
    dev_df = pd.read_csv(cfg.data["dev_csv"])
    collate = make_collate()
    train_set = CSVRegDataset(train_df, enc, cfg.model["max_length"], cfg.data["text_col"], cfg.data["label_col"])
    dev_set = CSVRegDataset(dev_df, enc, cfg.model["max_length"], cfg.data["text_col"], cfg.data["label_col"])

    if cfg.data["balance_batches"]:
        per_batch_mix = {d["name"]: d["weight"] for d in cfg.datasets}
        batch_sampler = BalancedRatioBatchSampler(
            dataset_names=list(per_batch_mix.keys()),
            ratios=per_batch_mix,
            per_batch_size=cfg.train["batch_size"],
            num_batches=None,  # default: floor(N / batch)
            shuffle_each_epoch=True,
            replacement_after_exhaustion=True,
            seed=cfg.seed,
            drop_last_incomplete=True,
        )
        train_loader = DataLoader(train_set, batch_sampler=batch_sampler, collate_fn=collate)
        dev_loader = DataLoader(dev_set, batch_sampler=batch_sampler, collate_fn=collate)
    else:
        train_loader = DataLoader(train_set, batch_size=cfg.train["batch_size"], shuffle=True, collate_fn=collate)
        dev_loader = DataLoader(dev_set, batch_size=cfg.eval["batch_size"], shuffle=False, collate_fn=collate)

    # optim
    no_decay = ["bias", "LayerNorm.weight"]
    enc_params = [(n, p) for n, p in model.named_parameters() if "enc." in n]
    head_params = [(n, p) for n, p in model.named_parameters() if "enc." not in n]
    params = [
        {"params": [p for n, p in enc_params if not any(nd in n for nd in no_decay)],
         "lr": float(cfg.train["lr_encoder"]),
         "weight_decay": float(cfg.train["weight_decay_encoder"])},
        {"params": [p for n, p in enc_params if any(nd in n for nd in no_decay)], "lr": float(cfg.train["lr_encoder"]),
         "weight_decay": 0.0},
        {"params": [p for n, p in head_params], "lr": float(cfg.train["lr_head"]),
         "weight_decay": float(cfg.train["weight_decay_head"])},
    ]
    optim = torch.optim.AdamW(params)
    total_steps = len(train_loader) * cfg.train["num_epochs"]
    warmup = int(total_steps * cfg.train["warmup_ratio"])
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=total_steps)

    main_loss = cfg.train["loss"].lower()
    rank_lambda = float(cfg.train["rank_lambda"])

    best = {"mae": 1e9, "spearman": -1.0}
    for epoch in range(cfg.train["num_epochs"]):
        tr_loss, tr_metrics = train_epoch(model, train_loader, optim, sched, device, main_loss, rank_lambda)
        eval_metrics = eval_epoch(model, dev_loader, device)
        print(
            f"epoch {epoch}: train_loss={tr_loss:.4f} "
            f"train_mae={tr_metrics['mae']:.4f} train_spear={tr_metrics['spearman']:.4f} "
            f"dev_mae={eval_metrics['mae']:.4f} dev_spear={eval_metrics['spearman']:.4f}")
        if eval_metrics["mae"] < best["mae"]:
            best = eval_metrics
            output_model_path = Path(cfg.logging["out_dir"]) / "best.pt"
            torch.save(model.state_dict(), output_model_path)
            logger.write_json("best_metrics.json", best)


if __name__ == "__main__":
    main()
