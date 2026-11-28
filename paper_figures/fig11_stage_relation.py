"""Figure 11 - CO and CO2 versus cumulative oxygen for the ten heats, with the four blowing stages. Width 1.0."""
import numpy as np
import pandas as pd

from style import CO, CO2, G6C, MODEL, REPO, STAGE_NAME, STAGE_TINT, load, ms, save
import os

d = load("fig10_11_process.npz")
st = pd.read_csv(os.path.join(REPO, "..", "B_results", "tables", "q2_stages.csv")).set_index("heat")

fig, axs = ms.subplots(2, 5, frac=1.0, aspect=0.50, sharex=True, sharey=True, gridspec_kw={"hspace": 0.02, "wspace": 0.01})
for k, ax in zip(range(1, 11), axs.flat):
    V = d[f"V{k}"] / 1e3
    co, co2, b = d[f"co{k}"], d[f"co2{k}"], d[f"bps{k}"] / 1e3
    edges = [0, *b, V[-1]]
    for i in range(4):
        ax.axvspan(edges[i], edges[i + 1], color=STAGE_TINT[i], lw=0, zorder=0)
    B = np.column_stack([np.ones_like(V), V] + [np.clip(V - bi, 0, None) for bi in b])
    fit = B @ np.linalg.lstsq(B, co, rcond=None)[0]
    ax.plot(V, co2, color=CO2, lw=1.1)
    ax.plot(V, co, color=CO, lw=1.1)
    ax.plot(V, fit, color=MODEL, lw=0.8, ls="--")
    ax.plot(b, B[np.searchsorted(V, b * 1e3 / 1e3)] @ np.linalg.lstsq(B, co, rcond=None)[0], "o", ms=2.5, color=MODEL)
    ax.set_xlim(0, 3.7)
    ax.set_ylim(-3, 78)
    grp = "6B" if k <= 4 else "6C"
    ax.text(0.04, 0.96, f"Heat {k}", transform=ax.transAxes, ha="left", va="top", fontsize=7.5, 
            color="#1F2A78")
for ax in axs[1]:
    ax.set_xlabel(r"$V$ (10³ m³)")
for ax in axs[:, 0]:
    ax.set_ylabel("Reading (vol.%)")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
handles = [Line2D([], [], color=CO, lw=1.2, label="CO"), Line2D([], [], color=CO2, lw=1.2, label=r"CO$_2$"),
           Line2D([], [], color=MODEL, lw=0.9, ls="--", marker="o", ms=2.5, label="Piecewise fit (CO)")]
names = ["I Early blow", "II Ramp-up", "III Steady", "IV Decline"]
handles += [Patch(facecolor=STAGE_TINT[i], edgecolor="#BBBBBB", lw=0.4, label=names[i]) for i in range(4)]
fig.legend(handles=handles, loc="outside upper center", ncol=7, fontsize=7, handlelength=1.4, columnspacing=1.0)
save(fig, "fig11_four_stage_relation")
