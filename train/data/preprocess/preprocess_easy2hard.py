"""
Preprocess Easy2Hard-Bench (excluding Lichess) to unified CSV:
id,dataset,split,prompt_text,label_raw

Policy (simplified):
- For each included subset, merge all available raw splits (train + eval/evaluation/test if present).
- Then split the ENTIRE subset into train/dev/test using user-defined ratios.
- Finally, apply a GLOBAL robust min-max (percentiles) to `rating` across *all* included subsets,
  map to [0,1], and write unified CSVs.

Included E2H configs: E2H-AMC, E2H-Codeforces, E2H-GSM8K, E2H-ARC, E2H-Winogrande
Excluded: E2H-Lichess

Usage:
  python preprocess_easy2hard.py --out_dir data/raw/e2h --train_frac 0.70 --dev_frac 0.10 --test_frac 0.20 --seed 1337
"""

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from datasets import load_dataset

HF_ID = "furonghuang-lab/Easy2Hard-Bench"
INCLUDE_CONFIGS = ["E2H-AMC", "E2H-Codeforces", "E2H-GSM8K", "E2H-ARC", "E2H-Winogrande"]
DATASET_NAME = "e2h"

SAFE_QUESTION_KEYS = ("question", "problem", "prompt", "text", "stem")
SAFE_CF_KEYS = ("statement", "problem_statement", "description")


def set_seed(seed: int):
    random.seed(seed)


def extract_prompt(ex: dict) -> str:
    # Prefer explicit question/statement fields; never include answers/solutions.
    for k in SAFE_QUESTION_KEYS:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for k in SAFE_CF_KEYS:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Fallback: first short safe text field
    for k, v in ex.items():
        lk = k.lower()
        if any(bad in lk for bad in ("answer", "solution", "label", "explanation")):
            continue
        if isinstance(v, str) and 0 < len(v) < 4000:
            return v.strip()
    return ""


def write_csv(rows: List[Tuple[str, str, str, str, str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "dataset", "split", "prompt_text", "label_raw"])
        w.writerows(rows)


def robust_minmax(x: np.ndarray, lo_q=1.0, hi_q=99.0) -> Tuple[float, float]:
    lo = float(np.percentile(x, lo_q))
    hi = float(np.percentile(x, hi_q))
    if math.isclose(hi, lo):
        hi = lo + 1e-6
    return lo, hi


def split_by_ratios(items: List, ratios: Tuple[float, float, float], seed: int) -> Tuple[List, List, List]:
    """Split items into train/dev/test with deterministic shuffle."""
    total = len(items)
    idx = list(range(total))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    a, b, c = ratios
    s = a + b + c
    if s <= 0:
        a, b, c, s = 0.7, 0.1, 0.2, 1.0
    a /= s
    b /= s
    c /= s
    n_train = int(round(total * a))
    n_dev = int(round(total * b))
    train_idx = idx[:n_train]
    dev_idx = idx[n_train:n_train + n_dev]
    test_idx = idx[n_train + n_dev:]

    def get(ids):
        return [items[i] for i in ids]
    return get(train_idx), get(dev_idx), get(test_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/raw/e2h")
    ap.add_argument("--seed", type=int, default=1337)

    # Per-subset split ratios
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--dev_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.20)

    # Robust normalization percentiles for rating -> [0,1]
    ap.add_argument("--lo_q", type=float, default=1.0)
    ap.add_argument("--hi_q", type=float, default=99.0)
    args = ap.parse_args()

    set_seed(args.seed)
    ratios = (args.train_frac, args.dev_frac, args.test_frac)

    # Load each subset, MERGE its available splits, collect (subset, prompt, rating, id)
    pooled_by_subset: Dict[str, List[Tuple[str, float, str]]] = {}  # subset -> list of (prompt, rating, raw_id)
    all_ratings: List[float] = []

    for cfg in INCLUDE_CONFIGS:
        ds = load_dataset(HF_ID, cfg)
        raw_keys = list(ds.keys())
        print(f"[e2h] subset {cfg} raw splits: {raw_keys}")

        # Merge all available relevant splits
        merged_examples = []
        for k in raw_keys:
            if k.lower() not in {"train", "eval", "evaluation", "test"}:
                continue
            for i, ex in enumerate(ds[k]):
                r = ex.get("rating", None)
                if r is None:
                    continue
                try:
                    rating = float(r)
                except Exception:
                    continue
                prompt = extract_prompt(ex)
                if not prompt:
                    continue
                exid = str(ex.get("id") or ex.get("uid") or ex.get("problem_id") or f"{cfg}-{k}-{i}")
                merged_examples.append((prompt, rating, exid))

        if not merged_examples:
            print(f"[e2h][warn] subset {cfg} had no usable records; skipping.")
            continue

        pooled_by_subset[cfg] = merged_examples
        for _, rating, _ in merged_examples:
            all_ratings.append(rating)

    if not pooled_by_subset:
        raise SystemExit("[e2h] No records loaded. Check HF access/configs.")

    # Compute GLOBAL robust bounds and define rating -> [0,1]
    lo, hi = robust_minmax(np.array(all_ratings, dtype=float), args.lo_q, args.hi_q)
    print(f"[e2h] robust bounds over all subsets: lo={lo:.6f}, hi={hi:.6f}")

    def to01(r: float) -> float:
        r = min(max(r, lo), hi)
        return (r - lo) / (hi - lo)

    # Per-subset split using the specified ratios, then assemble final rows
    rows_by_split: Dict[str, List[Tuple[str, str, str, str, str]]] = {"train": [], "dev": [], "test": []}

    for subset, items in pooled_by_subset.items():
        # Split this subset
        tr_items, dv_items, te_items = split_by_ratios(items, ratios, seed=args.seed)

        # Normalize ratings and create rows
        for split_name, seq in (("train", tr_items), ("dev", dv_items), ("test", te_items)):
            for prompt, rating, exid in seq:
                rid = f"{DATASET_NAME}:{subset}:{split_name}:{exid}"
                rows_by_split[split_name].append([rid, DATASET_NAME, split_name, prompt, f"{to01(rating):.6f}"])

        print(f"[e2h] subset {subset}: train={len(tr_items)} dev={len(dv_items)} test={len(te_items)}")

    # Write unified CSVs
    out = Path(args.out_dir)
    write_csv(rows_by_split["train"], out / f"{DATASET_NAME}_train.csv")
    write_csv(rows_by_split["dev"], out / f"{DATASET_NAME}_dev.csv")
    write_csv(rows_by_split["test"], out / f"{DATASET_NAME}_test.csv")

    print(f"[e2h] wrote CSVs to {out.resolve()}")
    for s in ("train", "dev", "test"):
        print(f"  {s}: {len(rows_by_split[s])} rows")


if __name__ == "__main__":
    main()
