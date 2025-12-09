FLOOR_MAX = 0.15  # reserved for floorpack
NONFLOOR_MIN = 0.20  # start other datasets above the buffer
NONFLOOR_MAX = 1.00


def _affine_to_range(x01: float, lo: float, hi: float) -> float:
    # map [0,1] -> [lo,hi]
    x01 = max(0.0, min(1.0, float(x01)))
    return lo + x01 * (hi - lo)


def norm_from_source(dataset: str, raw_label):
    ds = str(dataset).lower()

    # 1) Floorpack: pass-through (already in [0,1], but conceptually [0, 0.15])
    if ds == "floorpack":
        return float(raw_label)

    # 2) Dataset-specific normalization to [0,1]
    if ds == "math":
        lvl = int(raw_label)  # 1..5
        local = (lvl - 1) / 4.0  # -> [0,1]
    elif ds == "arc":
        m = str(raw_label).lower()
        local = 0.0 if "easy" in m else 1.0
    elif ds == "apps":
        m = str(raw_label).lower()
        local = 0.0 if "intro" in m else (0.5 if "interview" in m else 1.0)
    elif ds == "race":
        m = str(raw_label).lower()
        local = 0.5 if "middle" in m else 1.0
    elif ds == "anli":  # all are hard; preserve R1<R2<R3
        r = str(raw_label).lower()
        if "r1" in r:
            local = 0.70
        elif "r2" in r:
            local = 0.85
        else:
            local = 1.00
    elif ds.startswith("crossdiff"):
        local = float(raw_label)
    else:
        raise ValueError(f"Unsupported dataset: {ds}")

    # 3) Global remap for all non-floor datasets to [NONFLOOR_MIN, NONFLOOR_MAX]
    return _affine_to_range(local, NONFLOOR_MIN, NONFLOOR_MAX)
