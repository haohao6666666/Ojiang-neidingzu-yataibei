"""Shared paths, data loading, helpers and plotting style for APMCM 2026 Problem B."""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data_raw")
CLEAN = os.path.join(ROOT, "data_clean")
FIG = os.path.join(ROOT, "figures")
TAB = os.path.join(ROOT, "tables")
SEED = 42

HEATS = list(range(1, 11))
HEAT_ID = {1: "6B03205", 2: "6B03206", 3: "6B03270", 4: "6B03271", 5: "6C02057",
           6: "6C02058", 7: "6C02059", 8: "6C02060", 9: "6C02107", 10: "6C02108"}
GROUP = {k: ("6B" if k <= 4 else "6C") for k in HEATS}
DT = {k: (1.0 if k == 5 else 0.5) for k in HEATS}   # heat 5 judged to be 1 s sampling
SAT = 65535.0
WL_GRID = np.arange(341.0, 1023.0 + 1e-9, 1.0)       # unified 1 nm grid inside common range
GROUP_COLOR = {"6B": "#1f77b4", "6C": "#d62728"}


def load_raw(k):
    """Return dict with wl, spec (n x 2047), o2, co2, co, t for heat k (from cached npz)."""
    d = np.load(os.path.join(RAW, f"heat{k:02d}.npz"))
    out = {n: d[n] for n in d.files}
    out["t"] = np.arange(len(out["o2"])) * DT[k]
    return out


def load_clean(k):
    d = np.load(os.path.join(CLEAN, f"heat{k:02d}_clean.npz"))
    return {n: d[n] for n in d.files}


def hampel_mask(x, win, nsig):
    """Boolean mask of Hampel outliers along a 1-D series."""
    med = median_filter(x, size=win, mode="nearest")
    mad = median_filter(np.abs(x - med), size=win, mode="nearest")
    scale = 1.4826 * np.maximum(mad, 1e-9)
    return np.abs(x - med) > nsig * scale


def update_points(x):
    """Indices where a sample-and-hold signal changes value (plus first index)."""
    idx = np.r_[0, np.where(np.diff(x) != 0)[0] + 1]
    return idx


def pchip_fill(t, y, valid):
    f = PchipInterpolator(t[valid], y[valid], extrapolate=True)
    return f(t)


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "legend.fontsize": 8, "figure.dpi": 110,
        "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
