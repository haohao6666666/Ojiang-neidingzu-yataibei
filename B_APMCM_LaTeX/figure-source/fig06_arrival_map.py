"""Figure 6 - arrival-time map (schematic built from the fitted PFD numbers). Width 0.8."""
import numpy as np
from matplotlib.patches import Patch

from style import CO, GOLD, GREY, MODEL, load, ms, save

d = load("fig12_pfd.npz")
tmix, dv, (lo, hi) = float(d["tmix"]), float(d["vpipe"]), d["ci"]
LOW, HIGH = "#9C93C9", MODEL
flows = [(2.35, LOW, "2.35 m³/s (ignition)"), (5.0, HIGH, "5.0 m³/s (main blow)")]

fig, (ax, bx) = ms.subplots(2, 1, frac=0.8, aspect=0.74, sharex=True, gridspec_kw={"height_ratios": [3.1, 1], "hspace": 0.02})
t = np.linspace(0, 75, 400)

# (top) oxygen blown since the bath event; the analyzer sees the gas once the duct volume has been refilled
ax.axhspan(lo, hi, color=GOLD, alpha=0.25, lw=0)
ax.axhline(dv, color="#9C7A2B", lw=1.0)
ax.text(1.5, dv + 9, r"$\Delta V_{\mathrm{pipe}}$ = " f"{dv:.1f} m³", fontsize=7.2,
        color="#7A5A14", va="bottom", linespacing=1.1)
qv = 2.35 + (5.0 - 2.35) / (1 + np.exp(-(t - 22) / 5))
Vv = np.cumsum(qv) * (t[1] - t[0])
ax.plot(t, Vv, color=GREY, lw=1.0, ls="--", label=r"Ramping $Q(t)$")
tv = float(np.interp(dv, Vv, t))
ax.plot(tv, dv, "o", ms=4, color=GREY, mec="white", mew=0.6, zorder=5)
for q, col, lab in flows:
    ax.plot(t, q * t, color=col, lw=1.8, label=lab)
    ax.plot([dv / q] * 2, [0, dv], color=col, lw=0.8, ls=":")
    ax.plot(dv / q, dv, "o", ms=5.5, color=col, mec="white", mew=0.8, zorder=6)
ax.plot(0, 0, marker="*", ms=12, color=CO, mec="white", mew=0.5, zorder=7, clip_on=False)
ax.annotate("bath event (flame sees it)", xy=(0.6, 0.6), xytext=(28.5, 4), fontsize=7, color=CO,
            va="bottom", arrowprops=dict(arrowstyle="-|>", color=CO, lw=0.6, mutation_scale=6))
ax.annotate(r"analyzer sees it at $t_{\mathrm{arr}}$", xy=(dv / 5.0, dv), xytext=(1.5, 188), fontsize=7,
            color=MODEL, va="center", arrowprops=dict(arrowstyle="-|>", color=MODEL, lw=0.6, mutation_scale=6))
ax.set(ylim=(0, 205), ylabel=r"$V(s) - V(t)$ (m³)")
ax.set_yticks([0, 50, 100, 150, 200])
handles, labels = ax.get_legend_handles_labels()
mixing_key = Patch(facecolor=LOW, alpha=0.45, hatch="////", edgecolor="white", linewidth=0)
ax.legend([*handles, mixing_key], [*labels, fr"$T_{{\mathrm{{mix}}}} = {tmix:g}$ s"],
          loc="lower right", fontsize=6.8, handlelength=1.6)

# (bottom) delay decomposition = transport DeltaV/Q + mixing lag T_mix (values of Table 6)
for y, (q, col, lab) in zip((1, 0), flows):
    ta = dv / q
    bx.barh(y, ta, color=col, height=0.5)
    bx.barh(y, tmix, left=ta, color=col, alpha=0.45, height=0.5, hatch="////", edgecolor="white", lw=0)
    bx.text(ta + tmix + 1.0, y, fr"$\tau$ = {ta:.1f} + {tmix:.0f} = {ta + tmix:.1f} s", fontsize=7.3, color=col,
            va="center", weight="bold")
bx.set_yticks([1, 0], [f"{q} m³/s" for q, _, _ in flows], fontsize=7)
bx.set(ylim=(-0.6, 1.6), xlim=(0, 75), xlabel=r"Time since the bath event, $s - t$ (s)")
bx.grid(False)
save(fig, "fig06_arrival_time_map")
