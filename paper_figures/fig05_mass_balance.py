"""Figure 5 - carbon-oxygen balance under the flow-rate reading. Width 0.7."""
import os

import numpy as np
import pandas as pd

from style import G6B, G6C, MODEL, REPO, SALMON, ms, save

df = pd.read_csv(os.path.join(REPO, "..", "B_results", "tables", "q2_mass_balance.csv"))
r = df["ratio_needed_to_supplied"].to_numpy()
mu, sd = r.mean(), r.std(ddof=1)

fig, ax = ms.subplots(frac=0.7, aspect=0.6)
x = np.arange(1, 11)
cols = [G6B if k <= 4 else G6C for k in x]
bars = ax.bar(x, r, color=cols, width=0.66, zorder=2)
for xi, v in zip(x, r):
    ax.text(xi, v - 0.15, f"{v:.2f}", ha="center", va="top", fontsize=7, color="#2E3A6B" if xi <= 4 else "white", zorder=4)
ax.axhline(mu, color=MODEL, ls="--", lw=0.9, zorder=1, label=f"Mean {mu:.2f}")
ax.axhline(1.0, color=SALMON, lw=1.4, zorder=3, label="Physical limit (ratio = 1)")
ax.set(xticks=x, xticklabels=[f"H{k}" for k in x], xlim=(0.4, 10.6), ylim=(0, 8.6),
       ylabel=r"O$_2$ required / O$_2$ supplied")
ax.bar([0], [0], color=G6B, label="6B heats")
ax.bar([0], [0], color=G6C, label="6C heats")
ax.legend(loc="upper left", ncol=2, fontsize=7, columnspacing=1.0)
save(fig, "fig05_carbon_oxygen_balance")
