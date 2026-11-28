"""Figure 13 - windowed cross-correlation delays versus lance flow, with the PFD curve. Width 0.7."""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from style import C, G6B, G6C, MODEL, REPO, SALMON, load, ms, save

w = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "fig13_windows.csv"))
bins = pd.read_csv(os.path.join(REPO, "..", "B_results", "tables", "q2_delay_window_bins.csv"))
d = load("fig12_pfd.npz")
tmix, vp, (lo, hi) = float(d["tmix"]), float(d["vpipe"]), d["ci"]
rho, p = spearmanr(w.Q_mean, w.lag_xcorr)

fig, ax = ms.subplots(frac=0.7, aspect=0.66)
q = np.linspace(2.8, 6.2, 200)
ax.fill_between(q, tmix + lo / q, tmix + hi / q, color="#C9C6E3", lw=0)
ax.plot(q, tmix + vp / q, color=MODEL, lw=1.6, label=fr"PFD $\tau = {tmix:.0f} + {vp:.1f}/Q$")
for g, col in (("6B", G6B), ("6C", G6C)):
    gg = w[w.group == g]
    ax.scatter(gg.Q_mean, gg.lag_xcorr, s=11, color=col, alpha=0.75, edgecolor="none", label=f"Windows ({g})")
ax.errorbar(bins.Q_center, bins.delay_median, yerr=[bins.delay_median - bins.q25, bins.q75 - bins.delay_median],
            fmt="s", color=SALMON, mec="#8E3F38", mew=0.6, ms=6, capsize=3, elinewidth=1.1, ecolor="#8E3F38",
            label="Bin median (IQR)", zorder=5)
mant, expo = f"{p:.1e}".split("e")
ax.text(0.97, 0.97, fr"Spearman $\rho$ = $-${abs(rho):.2f}" "\n" fr"$p$ = {mant}$\times10^{{{int(expo)}}}$", transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
        color=MODEL, bbox=dict(boxstyle="round,pad=0.3", fc="#E3E3EE", ec="none"))
ax.set(xlim=(2.8, 6.2), ylim=(0, 100), xlabel=r"Window-mean lance flow $Q$ (m³/s)", ylabel="Delay (s)")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=6.5, handlelength=1.4, labelspacing=0.3)
save(fig, "fig13_windowed_delay_vs_flow")
