"""
Preprocess RACE to unified CSV:
id,dataset,split,prompt_text,label_raw

- Loads from HF dataset: race
- RACE has 'high' and 'middle' subsets. Some releases include question-level 'difficulty';
  if not present, we derive label_raw from subset: middle->Easy, high->Hard.
- Flatten each article into multiple question rows:
  prompt_text = passage + "\nQ: {question}\n" + joined options

Splits: train/validation/test (kept as train/dev/test)

Usage:
  python preprocess_race.py --out_dir data/raw/race
"""

import argparse
import csv
from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "race"


def fmt_prompt(article, question, options):
    opts = "\n".join([f"{chr(65 + i)}) {o}" for i, o in enumerate(options)])
    return f"{article.strip()}\n\nQ: {question.strip()}\n{opts}"


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "dataset", "split", "prompt_text", "label_raw"])
        for r in rows:
            w.writerow(r)


def prepare_subset(level):
    # level: "high" or "middle"
    ds = load_dataset("ehovy/race", level)
    rows = {"train": [], "dev": [], "test": []}
    for split_hf, split_out in [("train", "train"), ("validation", "dev"), ("test", "test")]:
        if split_hf not in ds:
            continue
        for i, ex in enumerate(ds[split_hf]):
            article = ex["article"]
            q = ex["question"]
            opts = ex["options"]

            rid = f"{DATASET_NAME}:{level}:{split_out}:{ex['example_id']}"
            prompt = fmt_prompt(article, q, opts)
            rows[split_out].append([rid, DATASET_NAME, split_out, prompt, level])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/raw/race")
    args = ap.parse_args()
    out = Path(args.out_dir)

    rows_mid = prepare_subset("middle")
    rows_high = prepare_subset("high")

    merged = {"train": [], "dev": [], "test": []}
    for s in ["train", "dev", "test"]:
        merged[s].extend(rows_mid[s])
        merged[s].extend(rows_high[s])
        write_csv(merged[s], out / f"{DATASET_NAME}_{s}.csv")

    print("Wrote RACE CSVs to", out)


if __name__ == "__main__":
    main()
