"""
Preprocess MATH (Hendrycks et al.) to unified CSV:
id,dataset,split,prompt_text,label_raw

- Supports multi-config datasets (subjects). Will enumerate all configs and merge.
- Fields expected: problem (text), level (1..5). Solution is ignored to avoid leakage.
- Splits: train/test per subject; optional global dev split carved from the merged train.

Usage:
  python preprocess_math.py --out_dir data/raw/math --make_dev 1 --dev_frac 0.05
"""

import argparse
import csv
import re
from pathlib import Path
from typing import List, Tuple

from datasets import load_dataset, get_dataset_config_names, Dataset, DatasetDict
from datasets.utils.logging import set_verbosity_error

DATASET_NAME = "math"
DATASET_ID = "EleutherAI/hendrycks_math"

LEVEL_RE = re.compile(r"(\d)")


def parse_level(raw) -> str:
    """
    Accepts integers or strings like 'Level 3', 'level5', 'L 2'.
    Returns a stringified integer in {'1','2','3','4','5'}; defaults to '3' if unknown.
    """
    if raw is None:
        return "3"
    # already an int-like
    try:
        val = int(raw)
        if 1 <= val <= 5:
            return str(val)
    except Exception:
        pass
    # try regex
    m = LEVEL_RE.search(str(raw))
    if m:
        val = int(m.group(1))
        if 1 <= val <= 5:
            return str(val)
    return "3"


def write_csv(rows: List[Tuple[str, str, str, str, str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "dataset", "split", "prompt_text", "label_raw"])
        w.writerows(rows)


def try_list_configs() -> List[str]:
    set_verbosity_error()
    last_err = None
    try:
        configs = get_dataset_config_names(DATASET_ID)
        if configs:
            return configs
    except Exception as e:
        last_err = e
    raise RuntimeError(
        f"Failed to enumerate configs for {DATASET_ID}\n"
        f"Last error: {type(last_err).__name__}: {last_err}"
    )


def load_split(ds_id: str, config: str, split: str) -> Dataset | None:
    """
    Robustly load a split for a given config; returns None if split missing.
    """
    try:
        dd: DatasetDict = load_dataset(ds_id, config, split=None)
    except Exception:
        # Some mirrors allow direct split loading
        try:
            return load_dataset(ds_id, config, split=split)
        except Exception:
            return None
    if split in dd:
        return dd[split]
    # Some mirrors use 'validation' instead of 'dev'
    if split == "dev" and "validation" in dd:
        return dd["validation"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/raw/math")
    ap.add_argument("--make_dev", type=int, default=1, help="carve a global dev split from merged train")
    ap.add_argument("--dev_frac", type=float, default=0.05)
    args = ap.parse_args()

    configs = try_list_configs()
    print(f"[math] found configs: {configs}")

    all_train: List[Tuple[str, str, str, str, str]] = []
    all_test: List[Tuple[str, str, str, str, str]] = []

    # Iterate configs (subjects) and collect rows
    for cfg in configs:
        ds_train = load_split(DATASET_ID, cfg, "train")
        ds_test = load_split(DATASET_ID, cfg, "test")

        if ds_train is None and ds_test is None:
            print(f"[warn] config '{cfg}' has no train/test; skipping.")
            continue

        if ds_train is not None:
            for i, ex in enumerate(ds_train):
                problem = ex.get("problem", "")
                level = parse_level(ex.get("level", ex.get("difficulty", "3")))
                rid = f"{DATASET_NAME}:{cfg}:train:{i}"
                all_train.append([rid, DATASET_NAME, "train", problem, str(level)])

        if ds_test is not None:
            for i, ex in enumerate(ds_test):
                problem = ex.get("problem", "")
                level = parse_level(ex.get("level", ex.get("difficulty", "3")))
                rid = f"{DATASET_NAME}:{cfg}:test:{i}"
                all_test.append([rid, DATASET_NAME, "test", problem, str(level)])

    if not all_train and not all_test:
        raise RuntimeError("No data loaded from any MATH config; check network or HF auth.")

    # Optional: carve a global dev from merged train
    all_dev: List[Tuple[str, str, str, str, str]] = []
    if args.make_dev and len(all_train) > 0:
        n = len(all_train)
        n_dev = max(1, int(n * args.dev_frac))
        all_dev = all_train[:n_dev]
        all_train = all_train[n_dev:]

    out = Path(args.out_dir)
    write_csv(all_train, out / f"{DATASET_NAME}_train.csv")
    write_csv(all_dev, out / f"{DATASET_NAME}_dev.csv")
    write_csv(all_test, out / f"{DATASET_NAME}_test.csv")

    print(f"[math] wrote CSVs to {out.resolve()}")
    print(f"  train: {len(all_train)} rows")
    print(f"  dev:   {len(all_dev)} rows")
    print(f"  test:  {len(all_test)} rows")


if __name__ == "__main__":
    main()
