"""
Preprocess BatsResearch/Cross-Difficulty to unified CSV:
id,dataset,split,prompt_text,label_raw

What it does
------------
- Authenticates to the gated HF dataset using a token from HF_TOKEN environment variable
- Enumerates ALL available configs/subsets on the repo.
- For each subset, merges all raw splits (train/valid/val/dev/test/eval/evaluation),
  then splits the ENTIRE subset into train/dev/test by user-specified ratios.
- Target = absolute difficulty: `1pl_diff` (NOT quantiles).
- Global robust min-max (1st–99th percentiles) maps `1pl_diff` to [0,1] across the union.
- Prompt building:
    - use `question` or `prompt` (whichever exists and is non-empty)
    - if `options` exists, append formatted options (one per line)

Output
------
Writes three CSVs in the output dir:
  crossdiff_train.csv
  crossdiff_dev.csv
  crossdiff_test.csv

Each with schema:
  id,dataset,split,prompt_text,label_raw

Usage
-----
python preprocess_cross_difficulty.py --out_dir data/raw/crossdiff \
  --train_frac 0.70 --dev_frac 0.10 --test_frac 0.20 --seed 1337
"""

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Sequence

import numpy as np
from datasets import load_dataset, get_dataset_config_names
from dotenv import load_dotenv

load_dotenv()

HF_REPO_ID = "BatsResearch/Cross-Difficulty"
DATASET_NAME = "crossdiff"
RAW_TEST_NAMES = {"test", "eval", "evaluation"}
RAW_DEV_NAMES = {"validation", "valid", "val", "dev"}
RAW_TRAIN_NAME = "train"


# -----------------------------
# Helpers
# -----------------------------


def set_seed(seed: int):
    random.seed(seed)


def robust_minmax(x: np.ndarray, lo_q=1.0, hi_q=99.0) -> Tuple[float, float]:
    lo = float(np.percentile(x, lo_q))
    hi = float(np.percentile(x, hi_q))
    if math.isclose(hi, lo):
        hi = lo + 1e-6
    return lo, hi


def split_by_ratios(items: List, ratios: Tuple[float, float, float], seed: int) -> Tuple[List, List, List]:
    """Deterministically split a list into train/dev/test by given ratios."""
    idx = list(range(len(items)))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    a, b, c = ratios
    s = a + b + c
    if s <= 0:
        a, b, c, s = 0.7, 0.1, 0.2, 1.0
    a /= s
    b /= s
    c /= s
    n_train = int(round(len(items) * a))
    n_dev = int(round(len(items) * b))
    train_idx = idx[:n_train]
    dev_idx = idx[n_train:n_train + n_dev]
    test_idx = idx[n_train + n_dev:]

    def get(arr, ids):
        return [arr[i] for i in ids]
    return get(items, train_idx), get(items, dev_idx), get(items, test_idx)


def write_csv(rows: List[Tuple[str, str, str, str, str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "dataset", "split", "prompt_text", "label_raw"])
        w.writerows(rows)


def deterministic_rng(seed: int, exid: str) -> random.Random:
    h = hashlib.md5(f"{seed}:{exid}".encode("utf-8")).hexdigest()
    # Use int from hex to seed a local RNG
    return random.Random(int(h[:16], 16))


# -----------------------------
# Option-dependence core
# Signals: regex cues, stem length, cue words, low stem↔options overlap (Jaccard on content words)
# -----------------------------

# Lightweight stopword list (en)
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "from", "is", "are", "was", "were", "be",
    "been", "being",
    "with", "without", "by", "as", "that", "this", "these", "those", "it", "its", "if", "then", "than", "which", "what",
    "who",
    "whom", "whose", "there", "their", "them", "they", "you", "your", "yours", "we", "our", "ours", "i", "me", "my",
    "mine",
    "do", "does", "did", "done", "doing", "can", "could", "should", "would", "may", "might", "must", "will", "shall",
    "not",
    "no", "yes", "all", "any", "each", "every", "some", "such", "one", "two", "three", "about", "into", "over", "under",
    "out",
    "up", "down", "more", "most", "less", "least", "few", "fewer"
}

# Phrases that strongly indicate dependence on options
_OD_PATTERNS = [
    r"\bwhich of the following\b",
    r"\bselect (?:one|two|the|all)\b",
    r"\ball that apply\b",
    r"\bfollowing (?:statements|options|answers|definitions|claims)\b",
    r"\b(?:is|are|was|were|would be) (?:true|false|correct|incorrect|not true|not correct)\b",
    r"\bexcept\b",
    r"\bnone of the above\b",
    r"\ball of the above\b",
]
OD_REGEX = re.compile("|".join(_OD_PATTERNS), re.IGNORECASE)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _content_words(text: str) -> List[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(text)]
    return [t for t in toks if t not in _STOP and len(t) > 2]


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def assess_option_dependence(stem: str, options: Sequence[str]) -> Tuple[bool, float, Dict[str, float]]:
    """
    Returns:
      od_bool: option-dependent?
      confidence: 0..1
      features: dict with individual signal scores
    """
    stem = stem or ""
    opts = [o for o in (options or []) if isinstance(o, str) and o.strip()]
    has_opts = len(opts) > 0

    # Signals
    s_regex = 1.0 if OD_REGEX.search(stem) else 0.0
    words = _content_words(stem)
    stem_len = len(words)

    s_short_stub = 1.0 if (stem_len < 8 and ("which" in stem.lower() or "select" in stem.lower())) else 0.0

    # Cue words near "following"/"except"/"incorrect" (soft signal)
    cue_hits = sum(int(bool(re.search(p, stem, re.IGNORECASE))) for p in [
        r"\bfollowing\b", r"\bexcept\b", r"\bincorrect\b", r"\bcorrect\b", r"\btrue\b", r"\bfalse\b"
    ])
    s_cues = min(1.0, cue_hits / 2.0)  # 0, 0.5, or 1.0

    # Overlap: stem vs all options (content words); low overlap + cues => dependent
    all_opt_words: List[str] = []
    for o in opts:
        all_opt_words.extend(_content_words(o))
    jac = _jaccard(words, all_opt_words)
    s_low_overlap = 1.0 if (jac < 0.08 and (s_regex > 0 or s_cues > 0 or s_short_stub > 0)) else 0.0

    # Combine (weighted)
    # Strong regex dominates; short-stub + cues + low-overlap backs it up.
    conf = (
            0.60 * s_regex +
            0.20 * s_cues +
            0.15 * s_short_stub +
            0.15 * s_low_overlap
    )
    od = (conf >= 0.50) and has_opts  # must have options to be option-dependent

    feats = {
        "regex": s_regex,
        "cues": s_cues,
        "short_stub": s_short_stub,
        "low_overlap": s_low_overlap,
        "jaccard": jac,
        "stem_len": float(stem_len)
    }
    return od, float(conf), feats


# -----------------------------
# MC formatting
# -----------------------------

LETTER_LABELS = [chr(ord('A') + i) for i in range(26)]
DIGIT_LABELS = [str(i + 1) for i in range(26)]


def _pair_labels(options: Sequence[str], labels: Sequence[str]) -> List[Tuple[str, str]]:
    k = min(len(options), len(labels))
    return [(labels[i], options[i].strip()) for i in range(k)]


def _opts_inline_labeled(pairs: Sequence[Tuple[str, str]], sep=", ") -> str:
    return sep.join([f"{lab}. {txt}" for lab, txt in pairs])


def _opts_lines_labeled(pairs: Sequence[Tuple[str, str]], bullet: str = "") -> str:
    if bullet:
        return "\n".join([f"{bullet} {lab}) {txt}" for lab, txt in pairs])
    return "\n".join([f"{lab}) {txt}" for lab, txt in pairs])


def _opts_yaml(pairs: Sequence[Tuple[str, str]]) -> str:
    return "Options:\n" + "\n".join([f"  - {lab}: {txt}" for lab, txt in pairs])


def _opts_json(pairs: Sequence[Tuple[str, str]]) -> str:
    # Build a proper JSON object and dump it (handles all escaping safely).
    return json.dumps({"options": {lab: txt for lab, txt in pairs}}, ensure_ascii=False)


def _opts_csv(pairs: Sequence[Tuple[str, str]]) -> str:
    return "Options (label,text):\n" + "\n".join([f"{lab}, {txt}" for lab, txt in pairs])


def _opts_table(pairs: Sequence[Tuple[str, str]]) -> str:
    lines = ["Options", "-------", "Label | Text", "----- | ----"]
    lines += [f"{lab}    | {txt}" for lab, txt in pairs]
    return "\n".join(lines)


def render_mc_question(base: str, options: Sequence[str], rng: random.Random,
                       include_mc_options: bool, allow_shuffle: bool = True) -> str:
    """Build a multiple-choice prompt with randomized, deterministic formatting."""
    opts = [o for o in options if isinstance(o, str) and o.strip()]
    if not opts:
        return base

    is_multi_choice_question, _, _ = assess_option_dependence(base, opts)
    if not is_multi_choice_question and not include_mc_options:
        return base

    # Choose lettered or numbered labels
    use_letters = rng.random() < 0.8
    labels = LETTER_LABELS if use_letters else DIGIT_LABELS

    # Optionally shuffle (deterministic via rng)
    if allow_shuffle:
        idx = list(range(len(opts)))
        rng.shuffle(idx)
        opts = [opts[i] for i in idx]

    pairs = _pair_labels(opts, labels)

    # Choose a template id
    template_id = rng.randrange(0, 14)

    # Sometimes add brief instruction
    instruct_variants = [
        "", "Choose the best answer.", "Select one option.", "Pick the correct choice.",
        "Answer with the letter only.", "Multiple-choice question."
    ]
    instruction = rng.choice(instruct_variants)

    # Build by template
    if template_id == 0:
        # Classic lines + 'Options:'
        body = f"{base}\nOptions:\n{_opts_lines_labeled(pairs)}"
    elif template_id == 1:
        # Inline comma-separated
        body = f"{base}\nOptions: {_opts_inline_labeled(pairs)}"
    elif template_id == 2:
        # Markdown bullets
        body = f"{base}\nOptions:\n{_opts_lines_labeled(pairs, bullet='-')}"
    elif template_id == 3:
        # Numbered labels (1)… if chosen above; else letters; inline semicolons
        body = f"{base}\n{_opts_inline_labeled(pairs, sep='; ')}"
    elif template_id == 4:
        # YAML-ish
        body = f"{base}\n{_opts_yaml(pairs)}"
    elif template_id == 5:
        # JSON-ish
        body = f"{base}\n{_opts_json(pairs)}"
    elif template_id == 6:
        # CSV-ish
        body = f"{base}\n{_opts_csv(pairs)}"
    elif template_id == 7:
        # Table-ish
        body = f"{base}\n{_opts_table(pairs)}"
    elif template_id == 8:
        # Bracketed labels
        body = f"{base}\nOptions:\n" + "\n".join([f"[{lab}] {txt}" for lab, txt in pairs])
    elif template_id == 9:
        # “Answer with the letter only.” prominent
        body = f"{base}\nAnswer with the letter only.\n{_opts_lines_labeled(pairs)}"
    elif template_id == 10:
        # Instruction before
        body = f"{instruction}\n{base}\n{_opts_lines_labeled(pairs)}"
    elif template_id == 11:
        # Instruction after
        body = f"{base}\n{_opts_lines_labeled(pairs)}\n{instruction}"
    elif template_id == 12:
        # Inline pipe-separated
        body = f"{base}\nOptions: {_opts_inline_labeled(pairs, sep=' | ')}"
    else:
        # Minimal: one per line, no explicit 'Options:'
        body = f"{base}\n{_opts_lines_labeled(pairs)}"

    # Occasionally add a short “Question:” label for variety
    if rng.random() < 0.25:
        body = "Question: " + body

    return body.strip()


# -----------------------------
# Prompt builder
# -----------------------------

def build_prompt_with_templates(ex: dict, seed: int, exid: str, include_mc_options: bool, shuffle_options: bool) -> str:
    prompt = ""
    for k in ("question", "prompt"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            prompt = v.strip()
            break

    narrative = ex.get("narrative")
    if narrative:
        return f"{narrative}\n{prompt}"

    opts = ex.get("options", ex.get("choices"))
    rng = deterministic_rng(seed, exid)

    if isinstance(opts, dict):
        opts = opts.get("text")
    if isinstance(opts, str):
        opts = ast.literal_eval(opts)
    if isinstance(opts, (list, tuple)) and len(opts) > 0:
        prompt = render_mc_question(prompt, opts, rng, include_mc_options=include_mc_options,
                                    allow_shuffle=shuffle_options)

    return prompt


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/raw/crossdiff")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--dev_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.20)
    ap.add_argument("--lo_q", type=float, default=1.0, help="robust low percentile for 1pl_diff")
    ap.add_argument("--hi_q", type=float, default=99.0, help="robust high percentile for 1pl_diff")
    ap.add_argument("--include_mc_options", type=int, default=0,
                    help="1 to include MC options in the example, 0 to omit unless absolutely necessary")
    ap.add_argument("--shuffle_options", type=int, default=1,
                    help="1 to shuffle MC options deterministically")
    args = ap.parse_args()

    set_seed(args.seed)
    ratios = (args.train_frac, args.dev_frac, args.test_frac)
    include_mc_options = bool(args.include_mc_options)
    shuffle_options = bool(args.shuffle_options)

    # List configs / subsets
    token = os.getenv("HF_TOKEN")
    try:
        configs = get_dataset_config_names(HF_REPO_ID, token=token)
    except TypeError:
        # older datasets version
        configs = get_dataset_config_names(HF_REPO_ID, use_auth_token=token)

    if not configs:
        raise SystemExit(f"[{DATASET_NAME}] No configs found on {HF_REPO_ID}. Check access/token.")

    # Load each subset, merge all raw splits, collect (subset, prompt, 1pl_diff, id)
    pooled_by_subset: Dict[str, List[Tuple[str, float, str]]] = {}
    all_scores: List[float] = []

    for cfg in configs:
        # Load with auth token (handle API arg name changes)
        try:
            ds = load_dataset(HF_REPO_ID, cfg, token=token)
        except TypeError:
            ds = load_dataset(HF_REPO_ID, cfg, use_auth_token=token)

        raw_keys = list(ds.keys())
        print(f"[{DATASET_NAME}] subset {cfg} raw splits: {raw_keys}")

        merged: List[Tuple[str, float, str]] = []  # (prompt, score, exid)

        for split_name, d in ds.items():
            if split_name.lower() not in {RAW_TRAIN_NAME, *RAW_DEV_NAMES, *RAW_TEST_NAMES}:
                continue
            for i, ex in enumerate(d):
                score = ex.get("1pl_diff", None)
                if score is None:
                    continue
                try:
                    s = float(score)
                except Exception:
                    continue

                exid = str(
                    ex.get("id")
                    or ex.get("uid")
                    or ex.get("example_id")
                    or f"{cfg}-{split_name}-{i}"
                )
                prompt = build_prompt_with_templates(
                    ex, seed=args.seed, exid=exid,
                    include_mc_options=include_mc_options, shuffle_options=shuffle_options
                )
                if not prompt:
                    continue

                merged.append((prompt, s, exid))

        if not merged:
            print(f"[{DATASET_NAME}][warn] subset {cfg} had no usable records; skipping.")
            continue

        pooled_by_subset[cfg] = merged
        all_scores.extend(s for _, s, _ in merged)

    if not pooled_by_subset:
        raise SystemExit(f"[{DATASET_NAME}] No items loaded from any subset. Check token/permissions.")

    # Global robust normalization of 1pl_diff -> [0,1]
    lo, hi = robust_minmax(np.array(all_scores, dtype=float), args.lo_q, args.hi_q)
    print(f"[{DATASET_NAME}] robust bounds over all subsets: lo={lo:.6f}, hi={hi:.6f}")

    def to01(v: float) -> float:
        v = min(max(v, lo), hi)
        return (v - lo) / (hi - lo)

    # Per-subset deterministic split by ratios; assemble output rows
    rows_by_split: Dict[str, List[Tuple[str, str, str, str, str]]] = {"train": [], "dev": [], "test": []}

    for subset, items in pooled_by_subset.items():
        tr, dv, te = split_by_ratios(items, ratios, seed=args.seed)
        for split_name, seq in (("train", tr), ("dev", dv), ("test", te)):
            for prompt, score, exid in seq:
                dataset_subset_id = f"{DATASET_NAME}:{subset}"
                rid = f"{dataset_subset_id}:{split_name}:{exid}"
                rows_by_split[split_name].append([rid, dataset_subset_id, split_name, prompt, f"{to01(score):.6f}"])
        print(f"[{DATASET_NAME}] subset {subset}: train={len(tr)} dev={len(dv)} test={len(te)}")

    # Write unified CSVs
    out = Path(args.out_dir)
    write_csv(rows_by_split["train"], out / f"{DATASET_NAME}_train.csv")
    write_csv(rows_by_split["dev"], out / f"{DATASET_NAME}_dev.csv")
    write_csv(rows_by_split["test"], out / f"{DATASET_NAME}_test.csv")

    print(f"[{DATASET_NAME}] wrote CSVs to {out.resolve()}")
    for s in ("train", "dev", "test"):
        print(f"  {s}: {len(rows_by_split[s])} rows")


if __name__ == "__main__":
    main()
