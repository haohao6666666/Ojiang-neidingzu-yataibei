"""Figure 4 - beat artifact of the oxygen counter (Heat 3). Width 1.0."""
import os

import numpy as np
import pandas as pd

from style import GREY, INDIGO, LAVENDER, MODEL, O2, REPO, SALMON, load, ms, save

beat = pd.read_csv(os.path.join(REPO, "..", "B_results", "tables", "q2_o2_counter_beat.csv")).set_index("heat").loc[3]
d = load("fig04_heat03_beat.npz")
t, dV = d["t"], d["dV"]

fig, (a, b) = ms.subplots(1, 2, frac=1.0, aspect=0.40, gridspec_kw={"width_ratios": [1, 1.45], "wspace": 0.06})

# (a) raw increments in a 60-s window of the main blow
w0 = 300.0
m = (t >= w0) & (t <= w0 + 60)
tt, dd = t[m] - w0, dV[m]
single = (dd > 0) & (dd < 1.6 * beat.single_increment_median)
double = dd >= 1.6 * beat.single_increment_median
a.vlines(tt[single], 0, dd[single], color=LAVENDER, lw=1.0)
a.plot(tt[single], dd[single], "o", ms=2.4, color=INDIGO, label=f"Single step ({beat.single_increment_median:.2f} m³)")
a.vlines(tt[double], 0, dd[double], color=SALMON, lw=1.6)
a.plot(tt[double], dd[double], "o", ms=4.2, color=SALMON, mec="white", mew=0.5, label="Double step")
td = tt[double]
i = int(np.argmin(np.abs(np.diff(td) - beat.double_spacing_s)))
x0, x1 = td[i], td[i + 1]
a.annotate("", xy=(x1, 11.0), xytext=(x0, 11.0), arrowprops=dict(arrowstyle="<->", color=MODEL, lw=0.8, shrinkA=0, shrinkB=0))
a.text((x0 + x1) / 2, 11.4, f"{beat.double_spacing_s:.1f} s", ha="center", va="bottom", fontsize=8, color=MODEL)
a.set(xlim=(-1, 61), ylim=(0, 15.5), xlabel="Time in a main-blow window (s)", ylabel=r"Counter increment $\Delta V$ (m³)")
a.set_yticks([0, 4.72, 9.44])
a.set_yticklabels(["0", "4.72", "9.44"])
a.legend(loc="upper center", ncol=2, columnspacing=0.8, handletextpad=0.3, fontsize=7)
ms.panel_label(a, "(a)", loc="below")

# (b) flow estimated from regression slopes over three windows
z = (t >= 180) & (t <= 390)
tm = t[z] / 60
b.plot(tm, d["q5"][z], color=GREY, lw=0.6, alpha=0.9, label="5 s")
b.plot(tm, d["q12"][z], color=SALMON, lw=1.0, label=f"12 s (SD {beat.Q_detrended_std_win12s:.2f})")
b.plot(tm, d["q24"][z], color=INDIGO, lw=1.9, label=f"24 s (SD {beat.Q_detrended_std_win24s:.2f})")
b.set(ylim=(3.7, 7.3), xlim=(tm[0], tm[-1]), xlabel="Time (min)", ylabel=r"Lance O$_2$ flow $Q$ (m³/s)")
b.legend(loc="upper center", ncol=3, fontsize=7, columnspacing=0.8, handlelength=1.4, title="Regression window", title_fontsize=7)
ms.panel_label(b, "(b)", loc="below")
save(fig, "fig04_o2_counter_beat")
