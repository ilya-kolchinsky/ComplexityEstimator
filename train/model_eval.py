import argparse
import pandas as pd
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from data.collate import make_collate
from models.encoder import HFEncoder
from models.regressor import Regressor
from train.utils.config import load_config
from train.utils.sampler import BalancedRatioBatchSampler
from train.utils.utils import get_device, mae, spearman


class CSVRegDatasetEval(torch.utils.data.Dataset):
    def __init__(self, df, enc, max_len, text_col, label_col):
        self.df, self.enc, self.max_len = df.reset_index(drop=True), enc, max_len
        self.text_col, self.label_col = text_col, label_col

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        toks = self.enc.tokenize([str(row[self.text_col])], self.max_len)
        item = {k: v[0] for k, v in toks.items()}
        item["label"] = float(row[self.label_col])
        return item


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="train/config.yaml")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--csv", type=str, help="override test CSV", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    enc = HFEncoder(cfg.model["name"])
    model = Regressor(enc, dropout=cfg.model["dropout"])
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    device = get_device(cfg)
    model.to(device).eval()

    csv_path = args.csv or cfg.data["test_csv"]
    df = pd.read_csv(csv_path)
    ds = CSVRegDatasetEval(df, enc, cfg.model["max_length"], cfg.data["text_col"], cfg.data["label_col"])

    if cfg.data["balance_batches"]:
        per_batch_mix = {d["name"]: d["weight"] for d in cfg.datasets}
        batch_sampler = BalancedRatioBatchSampler(
            dataset_names=list(per_batch_mix.keys()),
            ratios=per_batch_mix,
            per_batch_size=cfg.eval["batch_size"],
            num_batches=None,            # default: floor(N / batch)
            shuffle_each_epoch=True,
            replacement_after_exhaustion=True,
            seed=cfg.seed,
            drop_last_incomplete=True,
        )
        loader = DataLoader(ds, batch_sampler=batch_sampler, collate_fn=make_collate())
    else:
        loader = DataLoader(ds, batch_size=cfg.eval["batch_size"], shuffle=False, collate_fn=make_collate())

    y_true, y_pred = [], []
    for batch in tqdm(loader, desc="test"):
        for k in ["input_ids", "attention_mask", "labels"]:
            batch.__dict__[k] = batch.__dict__[k].to(device)
        p = model(batch.input_ids, batch.attention_mask)
        y_true.extend(batch.labels.detach().cpu().tolist())
        y_pred.extend(p.detach().cpu().tolist())
    print({"mae": mae(y_true, y_pred), "spearman": spearman(y_true, y_pred)})


if __name__ == "__main__":
    main()
