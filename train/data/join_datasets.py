"""
Unify multiple datasets into a single mixed dataset, preserving:
  - user-specified dataset mix ratios, and
  - user-specified train/dev/test split ratios.

Behavior
--------
1) For each dataset, read <prefix>_train.csv, _dev.csv, _test.csv and JOIN them
   into a single POOL (optionally deduped by prompt_text).
2) Decide TOTAL SIZE:
   - If --total_size > 0, use it.
   - Else AUTO: compute the maximum feasible total so that dataset mix ratios hold
     given per-dataset availability: total <= min_d floor( avail_d / ratio_d ).
3) Compute split sizes (train/dev/test) from total and split ratios.
4) Allocate per-SPLIT counts per DATASET using Hamilton rounding with capacity checks.
   We iterate splits in order (train, dev, test), consuming each dataset’s remaining
   capacity. This preserves both dataset ratios and split ratios simultaneously.
5) Sample without replacement from each dataset pool for each split, shuffle, normalize
   labels via normalize.norm_from_source, and write:
       unified_train.csv, unified_dev.csv, unified_test.csv
   with columns: id,dataset,split,prompt_text,label_raw,label
"""

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml

from train.data.normalize import norm_from_source

COLS = ["id", "dataset", "split", "prompt_text", "label_raw"]
OUT_COLS = ["id", "dataset", "split", "prompt_text", "label_raw", "label"]


@dataclass
class DSDef:
    name: str
    dir: Path
    prefix: str
    weight: float  # desired dataset-mix ratio (will be normalized)
    enabled: bool = True


# ----------------- IO -----------------

def parse_kv(s: str) -> Dict[str, str]:
    out = {}
    for part in s.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Bad dataset arg segment (expected key=value): {part}")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_mix_from_yaml(p: Path) -> List[DSDef]:
    if yaml is None:
        raise RuntimeError("PyYAML not available. Install with: pip install pyyaml, or use --dataset flags.")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    ds = []
    for row in data.get("datasets", []):
        if not row.get("enabled", True):
            continue
        ds.append(DSDef(
            name=str(row["name"]),
            dir=Path(row["dir"]),
            prefix=str(row["prefix"]),
            weight=float(row.get("weight", 1.0)),
            enabled=True
        ))
    return ds


def read_csv_rows(path: Path) -> List[List[str]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if header is None:
            return rows
        idx = {h: i for i, h in enumerate(header)}
        missing = [c for c in COLS if c not in idx]
        if missing:
            raise RuntimeError(f"{path} is missing columns: {missing}")
        for r in rdr:
            rows.append([r[idx[c]] for c in COLS])
    return rows


def dedup_rows(rows: List[List[str]]) -> List[List[str]]:
    seen = set()
    out = []
    for r in rows:
        text_norm = " ".join(r[3].split()).casefold()
        if text_norm in seen:
            continue
        seen.add(text_norm)
        out.append(r)
    return out


# -------------- math utils ------------

def normalize_ratios(ratios: Dict[str, float]) -> Dict[str, float]:
    positive = {k: v for k, v in ratios.items() if v > 0}
    if not positive:
        n = len(ratios)
        return {k: (1.0 / n if n else 0.0) for k in ratios}
    s = sum(positive.values())
    return {k: (positive.get(k, 0.0) / s) for k in ratios}


def hamilton_round(target: int, ratios: Dict[str, float]) -> Dict[str, int]:
    """Largest remainder allocation to hit exact target given ratios."""
    floats = {k: ratios[k] * target for k in ratios}
    floors = {k: int(math.floor(floats[k])) for k in floats}
    assigned = sum(floors.values())
    remainder = target - assigned
    if remainder <= 0:
        return floors
    fracs = sorted(
        [(k, floats[k] - floors[k]) for k in floats],
        key=lambda x: (x[1], x[0]),
        reverse=True
    )
    out = floors.copy()
    i = 0
    while remainder > 0 and i < len(fracs):
        k, _ = fracs[i]
        out[k] += 1
        remainder -= 1
        i += 1
    return out


def compute_split_sizes(total: int, train_ratio: float, dev_ratio: float, test_ratio: float) -> Dict[str, int]:
    ratios = [max(0.0, train_ratio), max(0.0, dev_ratio), max(0.0, test_ratio)]
    s = sum(ratios)
    if s <= 0:
        raise SystemExit("[prepare_data] At least one split ratio must be > 0.")
    ratios = [r / s for r in ratios]
    t_train = int(math.floor(total * ratios[0]))
    t_dev = int(math.floor(total * ratios[1]))
    t_test = total - t_train - t_dev
    if t_test < 0:
        t_test = 0
        if t_train >= t_dev:
            t_train = max(0, total - t_dev)
        else:
            t_dev = max(0, total - t_train)
    return {"train": t_train, "dev": t_dev, "test": t_test}


# ----- AUTO total -----

def compute_auto_total_size(avail_by_ds: Dict[str, int], desired_ratios: Dict[str, float]) -> int:
    """Max total so that per-dataset ratios can hold: total <= min_d floor(avail_d / ratio_d)."""
    ratios = normalize_ratios({k: desired_ratios.get(k, 0.0) for k in avail_by_ds.keys()})
    if not ratios:
        return 0
    caps = []
    for name, ratio in ratios.items():
        if ratio <= 0:
            continue
        caps.append(int(math.floor(avail_by_ds[name] / ratio)))
    return max(0, min(caps) if caps else 0)


# ----- allocation with capacities per split -----

def allocate_per_split_with_capacity(
        split_target: int,
        ds_ratios: Dict[str, float],
        remaining_cap: Dict[str, int],
) -> Dict[str, int]:
    """
    Allocate 'split_target' items across datasets per ratios, while respecting remaining_cap.
    Steps: Hamilton rounding -> clip to capacity -> redistribute deficits among datasets with capacity.
    """
    # Filter to datasets with capacity
    candidates = {k: ds_ratios.get(k, 0.0) for k, v in remaining_cap.items() if v > 0}
    if not candidates:
        return {k: 0 for k in remaining_cap}
    ratios = normalize_ratios(candidates)
    alloc = hamilton_round(split_target, ratios)

    # Clip to capacity
    clipped = {k: min(alloc.get(k, 0), remaining_cap.get(k, 0)) for k in candidates}
    assigned = sum(clipped.values())
    deficit = split_target - assigned
    if deficit <= 0:
        return clipped

    # Redistribute deficit among those with remaining capacity
    caps = {k: remaining_cap[k] - clipped.get(k, 0) for k in candidates}
    can_take = {k: ratios[k] for k, cap in caps.items() if cap > 0}
    if not can_take:
        return clipped  # can't fill further
    add = hamilton_round(deficit, normalize_ratios(can_take))
    for k, v in add.items():
        clipped[k] += min(v, caps[k])
    return clipped


# -------------- sampling / writing --------------

def sample_rows(rows: List[List[str]], k: int, rng: random.Random) -> List[List[str]]:
    if k >= len(rows):
        return rows[:]
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    pick = set(idx[:k])
    return [rows[i] for i in range(len(rows)) if i in pick]


def normalize_and_write(rows: List[List[str]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(OUT_COLS)
        for r in rows:
            ds = r[1]
            raw = r[4]
            y = norm_from_source(ds, raw)
            w.writerow(r + [f"{float(y):.6f}"])


# ------------------- main -------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", type=str, default="config/config.yaml", help="YAML dataset mix ratios.")
    ap.add_argument("--dataset", action="append", default=[],
                    help="Inline: name=...,dir=...,prefix=...,weight=...  (repeatable)")
    ap.add_argument("--out_dir", type=str, default="data/processed")
    ap.add_argument("--total_size", type=int, default=0, help="If <=0, auto-compute maximum feasible total.")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--dev_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dedup", type=int, default=0, help="1=dedup by prompt_text when pooling per dataset.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)

    # Load dataset definitions
    ds_defs: List[DSDef] = []
    if args.mix:
        ds_defs.extend(load_mix_from_yaml(Path(args.mix)))
    for ds_arg in args.dataset:
        kv = parse_kv(ds_arg)
        ds_defs.append(DSDef(
            name=kv["name"],
            dir=Path(kv["dir"]),
            prefix=kv["prefix"],
            weight=float(kv.get("weight", 1.0)),
            enabled=True
        ))
    if not ds_defs:
        raise SystemExit("[prepare_data] No datasets specified. Use --mix or --dataset flags.")
    if len(set(d.name for d in ds_defs)) != len(ds_defs):
        raise SystemExit("[prepare_data] Duplicate dataset names in config/flags.")

    # 1) POOL rows per dataset (join train/dev/test)
    pool_by_ds: Dict[str, List[List[str]]] = {}
    avail_by_ds: Dict[str, int] = {}
    for d in ds_defs:
        rows = []
        for split in ("train", "dev", "test"):
            p = d.dir / f"{d.prefix}_{split}.csv"
            new_rows = read_csv_rows(p)
            if d.name != d.prefix:
                # this dataset shares the data file with other datasets, filtering is needed
                new_rows = [r for r in new_rows if r[1] == d.name]
            rows.extend(new_rows)
        if args.dedup:
            rows = dedup_rows(rows)
        if rows:
            pool_by_ds[d.name] = rows
            avail_by_ds[d.name] = len(rows)
        else:
            print(f"[prepare_data] note: dataset '{d.name}' has no rows; excluded from mix.")

    if not pool_by_ds:
        # Write empty files with headers for reproducibility
        for split in ("train", "dev", "test"):
            normalize_and_write([], out_dir / f"unified_{split}.csv")
        print("[prepare_data] No data found in any dataset. Wrote empty files.")
        return

    # 2) Dataset ratios (normalized over datasets that actually have data)
    desired_ratios = {d.name: max(0.0, d.weight) for d in ds_defs if d.name in pool_by_ds}
    ds_ratios = normalize_ratios(desired_ratios)
    print(f"[prepare_data] dataset mix ratios (normalized over available): "
          f"{ {k: round(v, 4) for k, v in ds_ratios.items()} }")

    # 3) Decide TOTAL SIZE
    if args.total_size and args.total_size > 0:
        total = int(args.total_size)
        auto = False
    else:
        total = compute_auto_total_size(avail_by_ds, ds_ratios)
        auto = True
    print(f"[prepare_data] total_size = {total} ({'auto' if auto else 'user-specified'})")

    # 4) Split sizes
    split_sizes = compute_split_sizes(total, args.train_ratio, args.dev_ratio, args.test_ratio)
    print(f"[prepare_data] split sizes -> {split_sizes}")

    # 5) Allocate per-split with capacities (consume each dataset's pool progressively)
    remaining_cap = avail_by_ds.copy()
    per_split_alloc: Dict[str, Dict[str, int]] = {"train": {}, "dev": {}, "test": {}}
    for split in ("train", "dev", "test"):
        target = split_sizes[split]
        alloc = allocate_per_split_with_capacity(target, ds_ratios, remaining_cap)
        per_split_alloc[split] = alloc
        # consume capacity
        for ds_name, cnt in alloc.items():
            remaining_cap[ds_name] = max(0, remaining_cap[ds_name] - cnt)

    # Sanity: totals per dataset shouldn’t exceed availability
    for ds_name, cap in avail_by_ds.items():
        used = sum(per_split_alloc[s].get(ds_name, 0) for s in ("train", "dev", "test"))
        if used > cap:
            raise SystemExit(f"[prepare_data] internal error: overused dataset {ds_name}: used={used} cap={cap}")

    # 6) Sample, shuffle, normalize, write
    # Build per-dataset shuffled indices once (so sampling across splits is consistent)
    shuffled_pools: Dict[str, List[List[str]]] = {}
    for ds_name, rows in pool_by_ds.items():
        rows_copy = rows[:]
        rng.shuffle(rows_copy)
        shuffled_pools[ds_name] = rows_copy

    # Cursor per dataset to take disjoint slices for train/dev/test
    cursors = {ds_name: 0 for ds_name in pool_by_ds}

    def take(ds_name: str, k: int) -> List[List[str]]:
        if k <= 0: return []
        rows = shuffled_pools[ds_name]
        cur = cursors[ds_name]
        end = min(cur + k, len(rows))
        chunk = rows[cur:end]
        cursors[ds_name] = end
        return chunk

    out_rows: Dict[str, List[List[str]]] = {"train": [], "dev": [], "test": []}
    for split in ("train", "dev", "test"):
        for ds_name, k in per_split_alloc[split].items():
            out_rows[split].extend(take(ds_name, k))
        rng.shuffle(out_rows[split])  # final randomized order per split
        normalize_and_write(out_rows[split], out_dir / f"unified_{split}.csv")
        got = len(out_rows[split])
        print(f"[prepare_data] {split}: wrote {got} rows -> {out_dir / ('unified_' + split + '.csv')}")

    # Final mix report
    for split in ("train", "dev", "test"):
        counts = {}
        for r in out_rows[split]:
            counts[r[1]] = counts.get(r[1], 0) + 1
        tot = sum(counts.values()) or 1
        eff = {k: round(v / tot, 4) for k, v in counts.items()}
        print(f"[prepare_data] effective mix in {split}: {eff}")


if __name__ == "__main__":
    main()
