import os
import random

import numpy as np
import torch
from scipy.stats import spearmanr


def get_device(cfg):
    """
    Returns one of: 'cuda', 'mps', or 'cpu'.
    Order in auto mode: CUDA > MPS > CPU.
    """
    pref = (cfg.device or "auto").lower()
    if pref in {"cuda", "cpu", "mps"}:
        if pref == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        if pref == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return pref

    # auto
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def spearman(y_true, y_pred):
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho)
