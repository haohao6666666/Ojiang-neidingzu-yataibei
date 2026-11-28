"""Figure: beat artefact of the cumulative-O2 counter and its removal (heat 3).
(a) per-sample increments of the raw counter in a 60-s main-blow window;
(b) lance flow from sliding-regression slopes with 5-s, 12-s and 24-s windows.
Numbers in the annotations are read from tables/q2_o2_counter_beat.csv."""
import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.isotonic import IsotonicRegression

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (os.path.join(_HERE, "..", "..", "tools"),
           os.path.join(_HERE, "..", "..", ".claude", "skills", "mcm-figure-style", "scripts")):
    sys.path.insert(0, _d)
import mcm_style as ms
from common import load_raw, DT, FIG, TAB

ms.use_style(palette="steel", profile="apmcm")
C = ms.C
K = 3
r = load_raw(K)
t, o2, dt = r["t"], r["o2"], DT[K]
beat = pd.read_csv(os.path.join(TAB, "q2_o2_counter_beat.csv")).set_index("heat").loc[K]


def flow(win_s):
    iso = IsotonicRegression().fit_transform(t, o2)
    w = int(round(win_s / dt)) | 1
    return np.clip(savgol_filter(iso, w, 1, deriv=1, delta=dt, mode="interp"), 0, None)


dV = np.r_[0.0, np.diff(o2)]
fig, (a, b) = ms.subplots(1, 2, frac=1.0, aspect=0.36, gridspec_kw={"width_ratios": [1, 1.35]})

# (a) raw increments, 60-s window in the main blow
w0, w1 = 300.0, 360.0
m = (t >= w0) & (t <= w1)
tt, dd = t[m] - w0, dV[m]
single = (dd > 0) & (dd < 1.6 * beat.single_increment_median)
double = dd >= 1.6 * beat.single_increment_median
a.vlines(tt[single], 0, dd[single], color=C.primary, lw=1.1)
a.plot(tt[single], dd[single], "o", ms=2.6, color=C.primary,
       label=f"Single tick ({beat.single_increment_median:.2f} m$^3$)")
a.vlines(tt[double], 0, dd[double], color=C.accent, lw=1.4)
a.plot(tt[double], dd[double], "o", ms=3.6, color=C.accent, label="Double tick")
a.set_xlim(-1, 61)
a.set_ylim(0, 14.5)
a.set_xlabel("Time in window (s)")
a.set_ylabel(r"Counter increment $\Delta V$ (m$^3$)")
dt_ = np.diff(tt[double])
if len(dt_):
    i = len(dt_) - 1 - int(np.argmin(np.abs(dt_[::-1] - beat.double_spacing_s)))   # right-most matching pair
    x0, x1 = tt[double][i], tt[double][i + 1]
    a.annotate("", xy=(x1, 10.5), xytext=(x0, 10.5), arrowprops=dict(arrowstyle="<->", color=C.dark, lw=0.8))
    a.text((x0 + x1) / 2, 10.7, f"{x1 - x0:.1f} s", ha="center", va="bottom", color=C.dark)
a.legend(loc="upper left", frameon=False, handletextpad=0.3, ncol=2, columnspacing=0.8, bbox_to_anchor=(0.0, 1.02))
ms.panel_label(a, "(a)")

# (b) flow estimates in a 4-min main-blow window (raw increments are shown in panel a)
q5, q12, q24 = flow(5.0), flow(12.0), flow(24.0)
z = (t >= 180) & (t <= 420)
tm = t[z] / 60
b.plot(tm, q5[z], color=C.baseline, lw=0.7, label="5-s slope")
b.plot(tm, q12[z], color=C.accent, lw=1.0, label=f"12-s slope (SD {beat.Q_detrended_std_win12s:.2f})")
b.plot(tm, q24[z], color=C.primary, lw=1.9, label=f"24-s slope (SD {beat.Q_detrended_std_win24s:.2f})")
b.set_ylim(3.6, 7.0)
b.set_xlim(tm[0], tm[-1])
b.set_xlabel("Time (min)")
b.set_ylabel(r"Lance O$_2$ flow $Q$ (m$^3$/s)")
b.legend(loc="upper left", frameon=False, ncol=2, columnspacing=0.8, handlelength=1.4, bbox_to_anchor=(0.0, 1.03))
ms.panel_label(b, "(b)")
fig.tight_layout()
out = ms.save(fig, os.path.join(FIG, "fig_q2_o2_beat_artifact"))
print(out)
