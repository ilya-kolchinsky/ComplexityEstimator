"""
Preprocess APPS to unified CSV without using `datasets.load_dataset`.

It supports two modes:

1) Hub snapshot:
   - Download all files from a HF dataset repo (default: codeparrot/apps)
     via huggingface_hub.snapshot_download, then parse JSON/JSONL files.

2) Local directory:
   - Parse an already-downloaded APPS tree (JSON/JSONL).

Output CSV schema (per split):
    id,dataset,split,prompt_text,label_raw

- dataset    = "apps"
- split      = train/dev/test
- prompt_text: problem statement only (no solutions/tests)
- label_raw  = difficulty ("Introductory" / "Interview" / "Competition") when known,
               else "" (empty)

Usage examples:
  python preprocess_apps.py --out_dir data/raw/apps
  python preprocess_apps.py --out_dir data/raw/apps --hf_id myfork/apps
  python preprocess_apps.py --out_dir data/raw/apps --local_dir /path/to/APPS
  python preprocess_apps.py --out_dir data/raw/apps --make_dev 1 --dev_frac 0.05
"""
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

DATASET_NAME = "apps"
JSON_SUFFIXES = {".json", ".jsonl"}


def write_csv(rows: List[Tuple[str, str, str, str, str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "dataset", "split", "prompt_text", "label_raw"])
        w.writerows(rows)


def snapshot_to_dir(hf_id: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id=hf_id, repo_type="dataset", revision=revision)
    return Path(local_dir)


def iter_jsonlike_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in JSON_SUFFIXES:
                yield p


def read_jsonlike(p: Path) -> Iterable[dict]:
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    yield json.loads(s)
    else:  # .json
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, list):
                for r in obj:
                    if isinstance(r, dict):
                        yield r
            elif isinstance(obj, dict):
                # common container key
                if "data" in obj and isinstance(obj["data"], list):
                    for r in obj["data"]:
                        if isinstance(r, dict):
                            yield r
                else:
                    # some mirrors store one item per file
                    yield obj


# ---------- Field extraction ----------

PROMPT_KEYS = ("question", "prompt", "problem", "problem_statement", "input", "text")


def extract_prompt(ex: dict) -> str:
    for k in PROMPT_KEYS:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Some APPS store statement nested
    if "question" in ex and isinstance(ex["question"], dict):
        txt = ex["question"].get("text")
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
    return ""


def extract_diff_from_record(ex: dict) -> str:
    candidates = [
        ex.get("difficulty"), ex.get("difficulty_label"), ex.get("level"), ex.get("category")
    ]
    for c in candidates:
        if not c:
            continue
        s = str(c).lower()
        if "intro" in s:
            return "Introductory"
        if "interview" in s:
            return "Interview"
        if "compet" in s:
            return "Competition"
    return ""


def infer_split_from_path(p: Path) -> str:
    s = str(p.as_posix()).lower()
    if "/test" in s or s.endswith("/test") or "_test" in s:
        return "test"
    if "/valid" in s or "/val" in s or "_val" in s or "_valid" in s:
        return "dev"
    if "/train" in s or s.endswith("/train") or "_train" in s:
        return "train"
    # default to train
    return "train"


def infer_diff_from_path(p: Path) -> str:
    s = str(p.as_posix()).lower()
    if "intro" in s:
        return "Introductory"
    if "interview" in s:
        return "Interview"
    if "compet" in s:
        return "Competition"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/raw/apps")
    ap.add_argument("--hf_id", type=str, default="codeparrot/apps",
                    help="Hugging Face dataset repo id")
    ap.add_argument("--revision", type=str, default=None,
                    help="Optional HF revision (tag/sha)")
    ap.add_argument("--local_dir", type=str, default=None,
                    help="If set, parse this dir instead of downloading from HF")
    ap.add_argument("--make_dev", type=int, default=1,
                    help="carve dev from train if no explicit dev found")
    ap.add_argument("--dev_frac", type=float, default=0.05)
    args = ap.parse_args()

    if args.local_dir:
        root = Path(args.local_dir)
        if not root.exists():
            raise SystemExit(f"[apps] local_dir not found: {root}")
        print(f"[apps] using local dir: {root}")
    else:
        print(f"[apps] downloading snapshot: {args.hf_id} (revision={args.revision})")
        root = snapshot_to_dir(args.hf_id, args.revision)
        print(f"[apps] snapshot at: {root}")

    rows_by_split: Dict[str, List[Tuple[str, str, str, str, str]]] = {"train": [], "dev": [], "test": []}
    seen = 0
    for fp in iter_jsonlike_files(root):
        split = infer_split_from_path(fp)
        diff_from_path = infer_diff_from_path(fp)

        try:
            for i, ex in enumerate(read_jsonlike(fp)):
                prompt = extract_prompt(ex)
                if not prompt:
                    continue
                difficulty = extract_diff_from_record(ex) or diff_from_path
                # prefer stable ids if present
                exid = ex.get("problem_id") or ex.get("id") or f"{fp.stem}-{i}"
                rid = f"{DATASET_NAME}:{split}:{exid}"
                rows_by_split[split].append([rid, DATASET_NAME, split, prompt, difficulty])
                seen += 1
        except Exception as e:
            # be forgiving: skip bad files but report path
            print(f"[apps][warn] failed to parse {fp}: {e}")

    if seen == 0:
        raise SystemExit("[apps] no records found — check repo layout or provide --local_dir")

    # if there is no explicit dev, carve from train
    if args.make_dev and not rows_by_split["dev"] and rows_by_split["train"]:
        n = len(rows_by_split["train"])
        n_dev = max(1, int(n * args.dev_frac))
        rows_by_split["dev"] = rows_by_split["train"][:n_dev]
        rows_by_split["train"] = rows_by_split["train"][n_dev:]

    out = Path(args.out_dir)
    write_csv(rows_by_split["train"], out / f"{DATASET_NAME}_train.csv")
    write_csv(rows_by_split["dev"], out / f"{DATASET_NAME}_dev.csv")
    write_csv(rows_by_split["test"], out / f"{DATASET_NAME}_test.csv")

    print(f"[apps] wrote CSVs to {out.resolve()}")
    for split in ("train", "dev", "test"):
        print(f"  {split}: {len(rows_by_split[split])} rows")


if __name__ == "__main__":
    main()
