"""Shared figure style for the Problem B paper.

Every figure uses the same palette (steel), the APMCM text width (5.76 in) and the same
semantic colours, so a reader can recognise a quantity by its colour across the paper:

    CO   -> brick red      CO2  -> steel blue     O2 / lance flow / model -> navy
    6B   -> light steel    6C   -> navy           highlight (ours, key value) -> sand gold
    Stages I-IV -> four pale tints shared by Figures 10 and 11
"""
from __future__ import annotations

import os
import sys

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
import mcm_style as ms  # noqa: E402

DATA = os.path.join(HERE, "data")
OUT = os.path.join(REPO, "figures")
os.makedirs(OUT, exist_ok=True)

ms.use_style(palette="euler", profile="apmcm", base_pt=9.0)
C = ms.C

# --- Euler-Award look (2519935): grey-lavender panel, white grid, no spines, sans font, navy axis labels ---
import matplotlib as mpl  # noqa: E402

LABEL = "#1F2A78"
PANEL = "#EAEAF2"
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.facecolor": PANEL,
    "axes.edgecolor": PANEL,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "both",
    "grid.color": "white", "grid.linestyle": "-", "grid.linewidth": 0.8,
    "axes.labelcolor": LABEL,
    "xtick.color": "#333333", "ytick.color": "#333333",
    "xtick.major.size": 0, "ytick.major.size": 0,
    "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
    "legend.frameon": True, "legend.facecolor": "#E3E3EE", "legend.edgecolor": "none", "legend.framealpha": 0.9,
    "font.size": 8.5, "axes.labelsize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
})

INDIGO = "#4F6096"     # primary (Euler dark bar)
LAVENDER = "#B3A6CF"   # secondary (Euler light bar)
PINK = "#DDB6C6"
PEACH = "#F4C38F"
SALMON = "#E0876A"
BRICK = "#B4605A"
NAVY = "#2E3A6B"

CO = SALMON
CO2 = INDIGO
O2 = NAVY
MODEL = NAVY
GOLD = PEACH
GREY = "#9A9AA8"
G6B = LAVENDER
G6C = INDIGO
GROUP_COLOR = {"6B": G6B, "6C": G6C}
STAGE_TINT = ["#DCDCEC", "#F3E3D6", "#E3E3EC", "#EBDDE4"]
STAGE_NAME = ["I", "II", "III", "IV"]

# Euler heat-map ramp: indigo -> lavender -> pink -> peach -> cream (monotone in lightness)
FLAME = LinearSegmentedColormap.from_list(
    "flame_euler", ["#26305E", "#4F6096", "#8F8BC2", "#D7AFC6", "#F4C38F", "#FCEBD2"])
DIV = LinearSegmentedColormap.from_list("div_euler", ["#3E4F86", "#9FA6D0", "#F5F5F8", "#EDB49C", "#B4605A"])
TEXTW = ms.textwidth("apmcm")


def load(name: str):
    return np.load(os.path.join(DATA, name), allow_pickle=True)


def save(fig, name: str):
    paths = ms.save(fig, os.path.join(OUT, name), formats=("pdf",))
    print("saved", paths[0])
    return paths


def stage_shading(ax, edges, y_text=None, labels=True, fontsize=7):
    """Shade the four blowing stages between consecutive x edges [x0, b1, b2, b3, x_end]."""
    for i in range(4):
        ax.axvspan(edges[i], edges[i + 1], color=STAGE_TINT[i], lw=0, zorder=0)
        if labels:
            yt = ax.get_ylim()[1] if y_text is None else y_text
            ax.text((edges[i] + edges[i + 1]) / 2, yt, STAGE_NAME[i], ha="center", va="top",
                    fontsize=fontsize, color="#4A4A4A", style="italic")


def corner(ax, text, loc="upper left", fontsize=7.5, color="#222222", **kw):
    x, ha = (0.03, "left") if "left" in loc else (0.97, "right")
    y, va = (0.95, "top") if "upper" in loc else (0.05, "bottom")
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize, color=color, **kw)
