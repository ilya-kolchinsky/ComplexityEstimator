"""
Preprocess ARC to unified CSV:
id,dataset,split,prompt_text,label_raw

- Loads from HF dataset: ai2_arc with subsets ARC-Easy and ARC-Challenge
- prompt_text = question + formatted choices
- label_raw = "Easy" or "Challenge"
- Splits: train/validation/test

Usage:
  python preprocess_arc.py --out_dir data/raw/arc
"""
import argparse
from pathlib import Path
import csv
from datasets import load_dataset

DATASET_NAME = "arc"


def fmt_prompt(q, choices):
    # choices: dict with 'text' & 'label' arrays
    pairs = [f"{lbl}) {txt}" for lbl, txt in zip(choices["label"], choices["text"])]
    return q.strip() + "\n" + "\n".join(pairs)


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "dataset", "split", "prompt_text", "label_raw"])
        for r in rows:
            w.writerow(r)


def prepare_subset(subset_name):
    ds = load_dataset("allenai/ai2_arc", subset_name)  # 'ARC-Easy' or 'ARC-Challenge'
    rows = {"train": [], "dev": [], "test": []}
    for split_hf, split_out in [("train", "train"), ("validation", "dev"), ("test", "test")]:
        if split_hf not in ds:
            continue
        for i, ex in enumerate(ds[split_hf]):
            rid = f"{DATASET_NAME}:{subset_name}:{split_out}:{i}"
            prompt = fmt_prompt(ex["question"], ex["choices"])
            label_raw = "Easy" if "Easy" in subset_name else "Challenge"
            rows[split_out].append([rid, DATASET_NAME, split_out, prompt, label_raw])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/raw/arc")
    args = ap.parse_args()
    out = Path(args.out_dir)

    rows_easy = prepare_subset("ARC-Easy")
    rows_chal = prepare_subset("ARC-Challenge")

    # Optionally merge Easy+Challenge into unified CSVs
    merged = {"train": [], "dev": [], "test": []}
    for s in ["train", "dev", "test"]:
        merged[s].extend(rows_easy[s])
        merged[s].extend(rows_chal[s])
        write_csv(merged[s], out / f"{DATASET_NAME}_{s}.csv")

    print("Wrote ARC CSVs to", out)


if __name__ == "__main__":
    main()
