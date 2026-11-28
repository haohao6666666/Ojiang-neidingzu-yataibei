"""Figure 14 - leave-one-heat-out zero-probe DSS predictions for the ten heats, with 90% conformal intervals. Width 1.0."""
import os

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from style import CO, CO2, GOLD, REPO, load, ms, save

d = load("fig14_loho_series.npz")
met = pd.read_csv(os.path.join(REPO, "solution_v4_verified", "solution_outputs", "tables", "q3_loho_per_heat_metrics.csv"))
met = met[met.Mode == "Combined_SoftGated_Hybrid"].set_index("Heat")
ID = ["6B03205", "6B03206", "6B03270", "6B03271", "6C02057", "6C02058", "6C02059", "6C02060", "6C02107", "6C02108"]

fig, axs = ms.subplots(5, 2, frac=1.0, aspect=1.02, sharex=True, sharey=True, gridspec_kw={"hspace": 0.02, "wspace": 0.02})
for k, ax in zip(range(1, 11), axs.flat):
    t = d[f"t_{k}"] / 60
    for key, col in (("co", CO), ("co2", CO2)):
        z, h, y = d[f"z_{key}_{k}"], d[f"zhw_{key}_{k}"], d[f"y_{key}_{k}"]
        ax.fill_between(t, np.clip(z - h, 0, None), z + h, color=col, alpha=0.16, lw=0)
        ax.plot(t, y, color=col, lw=1.1)
        ax.plot(t, z, color=col, lw=0.9, ls=(0, (3, 1.6)), alpha=0.95)
    m = met.loc[f"H{k:02d}"]
    ax.text(0.02, 0.93, f"Heat {k}", transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color="#1F2A78")
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 95)
    ax.set_yticks([0, 25, 50, 75])
    if k == 1:
        ax.set_facecolor("#F3E6DC")
for ax in axs[-1]:
    ax.set_xlabel("Blowing time (min)")
for ax in axs[:, 0]:
    ax.set_ylabel("vol.%")
handles = [Line2D([], [], color="#444444", lw=1.1, label="Measured (delay-aligned)"),
           Line2D([], [], color="#444444", lw=0.9, ls=(0, (3, 1.6)), label="Zero-probe DSS prediction"),
           Patch(facecolor="#888888", alpha=0.25, label="90% conformal interval"),
           Line2D([], [], color=CO, lw=2, label="CO"), Line2D([], [], color=CO2, lw=2, label=r"CO$_2$")]
fig.legend(handles=handles, loc="outside upper center", ncol=5, fontsize=7, handlelength=1.8, columnspacing=1.0)
save(fig, "fig14_loho_predictions")
