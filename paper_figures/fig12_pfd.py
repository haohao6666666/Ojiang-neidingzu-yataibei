"""Figure 12 - Plug-Flow Delay model: delay curve, oxygen-coordinate collapse, leave-one-heat-out stability. Width 1.0."""
import os

import numpy as np
import pandas as pd

from style import C, G6B, G6C, GOLD, GREY, MODEL, REPO, load, ms, save

d = load("fig12_pfd.npz")
ev = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "fig12_events.csv"))
fold = pd.read_csv(os.path.join(REPO, "solution_v4_verified", "solution_outputs", "tables", "q3_per_fold_delay_params.csv"))
tmix, vp, (lo, hi), r2 = float(d["tmix"]), float(d["vpipe"]), d["ci"], float(d["r2"])

fig, (a, b, c) = ms.subplots(1, 3, frac=1.0, aspect=0.36, gridspec_kw={"width_ratios": [1.15, 1.15, 0.8], "wspace": 0.04})

# (a) delay versus lance flow
q = np.linspace(2.0, 7.2, 300)
a.fill_between(q, tmix + lo / q, tmix + hi / q, color="#C9C6E3", lw=0)
a.plot(q, vp / q, color=GREY, lw=0.9, ls="--", label=r"$\Delta V/Q$ only")
a.plot(q, tmix + vp / q, color=MODEL, lw=1.8, label=fr"PFD $\tau = {tmix:.0f} + {vp:.1f}/Q$")
ig = ev[ev.Event == "Ignition"]
mr = ev[ev.Event == "MainRamp"]
a.plot(ig.Q_m3_s, ig.Tau_obs_s, "o", ms=3.6, mfc="white", mec=G6C, mew=0.8, label="Ignition")
a.plot(mr.Q_m3_s, mr.Tau_obs_s, "D", ms=3.4, color=GOLD, mec="white", mew=0.4, label="Main-blow onset")
for qv in (3.0, 5.0):
    tv = tmix + vp / qv
    a.plot(qv, tv, "o", ms=4.5, color=MODEL, zorder=5)
    a.annotate(f"{tv:.1f} s", xy=(qv, tv), xytext=(qv + 0.5, tv + (5 if qv < 4 else 9)), fontsize=7.5, color=MODEL, weight="bold",
               arrowprops=dict(arrowstyle="-", color=MODEL, lw=0.5))
a.set(xlim=(2.0, 7.2), ylim=(10, 100), xlabel=r"Lance O$_2$ flow $Q$ (m³/s)", ylabel=r"Flue-gas delay $\tau$ (s)")
a.legend(loc="upper right", fontsize=6.5, handlelength=1.4, labelspacing=0.3)
ms.panel_label(a, "(a)", loc="below")

# (b) collapse of the oxygen-equivalent signal in the oxygen coordinate
for k in range(1, 11):
    col = G6B if k <= 4 else G6C
    b.plot(d[f"o2_{k}"], d[f"co2_{k}"] + 0.5 * d[f"co_{k}"], color=col, lw=0.8, alpha=0.85)
b.axvspan(0, vp, color=GOLD, alpha=0.18, lw=0)
b.axvline(vp, color=GOLD, lw=1.0)
b.annotate(fr"$\Delta V_{{\mathrm{{pipe}}}}$ = {vp:.1f} m³", xy=(vp, 6), xytext=(520, 6), fontsize=7.5, color="#9A6A2E",
           va="center", arrowprops=dict(arrowstyle="-|>", color="#9A6A2E", lw=0.6, mutation_scale=6))
b.plot([], [], color=G6B, label="6B heats (1–4)")
b.plot([], [], color=G6C, label="6C heats (5–10)")
b.legend(loc="upper left", fontsize=6.5, handlelength=1.3, bbox_to_anchor=(0.0, 1.0))
b.set(xlim=(0, 3700), ylim=(-2, 50), xlabel=r"Cumulative O$_2$ $V$ (m³)",
      ylabel=r"$y_{\mathrm{CO_2}} + 0.5\,y_{\mathrm{CO}}$ (vol.%)")
ms.panel_label(b, "(b)", loc="below")

# (c) leave-one-heat-out recalibration of the duct volume
dv = fold["Train_dV_pipe_m3"].to_numpy()
y = np.arange(10)[::-1]
c.axvspan(lo, hi, color="#D6D3EA", lw=0)
c.axvline(vp, color=MODEL, lw=1.0)
cols = [G6B if i < 4 else G6C for i in range(10)]
c.hlines(y, vp, dv, color=cols, lw=1.0)
c.scatter(dv, y, s=16, color=cols, zorder=3, edgecolor="white", linewidth=0.4)
c.set_yticks(y, [f"w/o H{i + 1}" for i in range(10)], fontsize=7)
c.set(xlim=(121, 135), ylim=(-1.4, 9.6), xlabel=r"$\Delta V_{\mathrm{pipe}}$ (m³)")
c.grid(axis="y", visible=False)
c.grid(axis="x", visible=True)
c.text(0.5, 1.01, f"{dv.mean():.1f} ± {dv.std(ddof=1):.1f} m³", transform=c.transAxes, ha="center", va="bottom",
       fontsize=7.5, color=MODEL, weight="bold")
ms.panel_label(c, "(c)", loc="below")
save(fig, "fig12_plug_flow_delay")
