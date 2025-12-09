"""
Preprocess ANLI (R1/R2/R3 x train/dev/test) to unified CSV:
id,dataset,split,prompt_text,label_raw

- Loads HF dataset 'anli' which commonly has nine splits:
    train_r1, dev_r1, test_r1, train_r2, dev_r2, test_r2, train_r3, dev_r3, test_r3
- Builds a simple NLI prompt without gold labels:
    Premise: ...
    Hypothesis: ...
    Question: Does the premise support the hypothesis?
- label_raw = "R1" / "R2" / "R3" (used by normalize.py to map to high difficulty)

Usage:
  python preprocess_anli.py --out_dir data/raw/anli
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_dataset

DATASET_NAME = "anli"
PROMPT_TMPL = "Premise: {prem}\nHypothesis: {hyp}\nQuestion: Does the premise support the hypothesis?"

# Map HF split name -> (our_split, round_label)
SPLIT_MAP: Dict[str, Tuple[str, str]] = {
    "train_r1": ("train", "R1"),
    "dev_r1": ("dev", "R1"),
    "test_r1": ("test", "R1"),
    "train_r2": ("train", "R2"),
    "dev_r2": ("dev", "R2"),
    "test_r2": ("test", "R2"),
    "train_r3": ("train", "R3"),
    "dev_r3": ("dev", "R3"),
    "test_r3": ("test", "R3"),
}


def write_csv(rows: List[Tuple[str, str, str, str, str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "dataset", "split", "prompt_text", "label_raw"])
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/raw/anli")
    ap.add_argument("--hf_id", type=str, default="anli", help="HF dataset id (default: 'anli')")
    args = ap.parse_args()

    out = Path(args.out_dir)
    ds_all = load_dataset(args.hf_id)  # expect nine splits, but handle partial gracefully

    rows = {"train": [], "dev": [], "test": []}

    for hf_split, (our_split, round_label) in SPLIT_MAP.items():
        if hf_split not in ds_all:
            # some mirrors may miss certain splits
            continue
        dset = ds_all[hf_split]
        for i, ex in enumerate(dset):
            prem = (ex.get("premise") or "").strip()
            hyp = (ex.get("hypothesis") or "").strip()
            if not prem or not hyp:
                continue
            prompt = PROMPT_TMPL.format(prem=prem, hyp=hyp)
            rid = f"{DATASET_NAME}:{round_label}:{our_split}:{i}"
            rows[our_split].append([rid, DATASET_NAME, our_split, prompt, round_label])

    # write merged CSVs
    write_csv(rows["train"], out / f"{DATASET_NAME}_train.csv")
    write_csv(rows["dev"], out / f"{DATASET_NAME}_dev.csv")
    write_csv(rows["test"], out / f"{DATASET_NAME}_test.csv")

    print(f"[anli] wrote CSVs to {out.resolve()}")
    for s in ("train", "dev", "test"):
        print(f"  {s}: {len(rows[s])} rows")


if __name__ == "__main__":
    main()
