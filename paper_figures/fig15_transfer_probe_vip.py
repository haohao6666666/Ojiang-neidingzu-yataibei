"""Figure 15 - cross-heat transfer, probe-saving trade-off and wavelength importance. Width 1.0."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from style import CO, CO2, DIV, GOLD, GREY, MODEL, REPO, ms, save

V4 = os.path.join(REPO, "solution_v4_verified", "solution_outputs", "tables")
mat = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "fig15_transfer.csv"), index_col=0).to_numpy()
summ = pd.read_csv(os.path.join(V4, "q3_loho_ablation_summary.csv")).set_index("Mode")
vip = pd.read_csv(os.path.join(V4, "q3_pls_vip_wavelength.csv"))

fig = plt.figure(figsize=ms.figsize(1.0, 0.40))
gs = fig.add_gridspec(1, 6, width_ratios=[0.95, 0.045, 0.62, 1.15, 0.27, 1.05], wspace=0.0)
a, cax, b, c = (fig.add_subplot(gs[0, i]) for i in (0, 1, 3, 5))
fig.subplots_adjust(left=0.065, right=0.995, bottom=0.25, top=0.9)

# (a) single-heat transfer matrix (diagonal = in-sample, greyed out)
off = ~np.eye(10, dtype=bool)
show = np.where(off, np.clip(mat, -1.0, 1.0), np.nan)
im = a.imshow(show, cmap=DIV.reversed(), norm=TwoSlopeNorm(0, -1, 1), interpolation="nearest")
for i in range(10):
    a.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fc="#E3E3E3", ec="white", lw=0.5))
a.set_xticks(range(10), [str(i) for i in range(1, 11)], fontsize=7)
a.set_yticks(range(10), [str(i) for i in range(1, 11)], fontsize=7)
a.set(xlabel="Test heat", ylabel="Training heat")
a.grid(False)
a.set_facecolor("white")
for s in a.spines.values():
    s.set_visible(False)
a.tick_params(length=0)
o = mat[off]
cb = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1])
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=6.5, length=2)
cb.set_label(r"Transfer R²", fontsize=7, labelpad=1)
cax.grid(False)
pos = cax.get_position()
cax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.84])
ms.panel_label(a, "(a)", loc="below")

# (b) accuracy versus probe interval
modes = ["Combined_SoftGated_Hybrid", "Hybrid_Smith_Pulse_120s_99.6pctSave", "Hybrid_Smith_Pulse_60s_99.2pctSave",
         "Hybrid_Smith_Pulse_30s_98.3pctSave", "Hybrid_Smith_Pulse_15s_96.7pctSave", "Hybrid_Smith_Continuous_DualModal"]
naive = [None, "Naive_Lagged_Sensor_Pulse_120s_99.6pctSave", "Naive_Lagged_Sensor_Pulse_60s_99.2pctSave",
         "Naive_Lagged_Sensor_Pulse_30s_98.3pctSave", "Naive_Lagged_Sensor_Pulse_15s_96.7pctSave",
         "Naive_Lagged_Sensor_Continuous_DualModal"]
xl = ["None", "120 s", "60 s", "30 s", "15 s", "Cont."]
sv = ["100%", "99.6%", "99.2%", "98.3%", "96.7%", "0%"]
x = np.arange(6)
FLOOR = -0.25
b.axvspan(0.6, 1.4, color=GOLD, alpha=0.35, lw=0)
for col, key, mk in ((CO, "CO_Mean_R2", "o"), (CO2, "CO2_Mean_R2", "s")):
    ours = summ.loc[modes, key].to_numpy()
    lag = np.array([np.nan] + [summ.loc[m, key] for m in naive[1:]])
    b.plot(x, ours, mk + "-", color=col, lw=1.6, ms=4, mec="white", mew=0.5)
    b.plot(x, np.clip(lag, FLOOR, None), mk + "--", color=col, lw=0.9, ms=3.4, mfc="white", mec=col, alpha=0.85)
    for xi, v in zip(x, lag):
        if v < FLOOR:
            b.annotate(f"$-${abs(v):.2f}", xy=(xi, FLOOR), xytext=(xi - 0.75, FLOOR + 0.12), fontsize=6.5, color=col,
                       arrowprops=dict(arrowstyle="-|>", color=col, lw=0.6, mutation_scale=6))
            b.plot(xi, FLOOR, "v", color=col, ms=4)
co120, lag120 = summ.loc[modes[1], "CO_Mean_R2"], summ.loc[naive[1], "CO_Mean_R2"]
b.annotate(f"{co120:.3f}", xy=(1, co120), xytext=(1.25, 1.03), fontsize=7, color=CO, weight="bold",
           arrowprops=dict(arrowstyle="-", color=CO, lw=0.5))
b.annotate(f"{lag120:.2f}", xy=(1, lag120), xytext=(1.35, 0.50), fontsize=7, color=CO,
           arrowprops=dict(arrowstyle="-", color=CO, lw=0.5))
b.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0])
b.set_xticks(x, ["None", "120", "60", "30", "15", "Cont."], fontsize=7.2)
b.set(ylim=(-0.62, 1.1), ylabel=r"Mean LOHO R²", xlim=(-0.35, 5.35))
b.set_xlabel("Probe interval (s)")
b.plot([], [], "-", color="#444444", lw=1.6, label="DSS + Smith correction")
b.plot([], [], "--", color="#444444", lw=0.9, label="Hold last lagged reading")
b.legend(loc="lower right", fontsize=6.5, handlelength=1.8, bbox_to_anchor=(1.0, 0.0))
ms.panel_label(b, "(b)", loc="below")

# (c) PLS VIP by wavelength
wl = vip["Wavelength_nm"].to_numpy()
c.axvspan(730, 825, color=GOLD, alpha=0.30, lw=0)
c.axhline(1.0, color=GREY, ls="--", lw=0.8)
c.plot(wl, vip["CO2_VIP"], color=CO2, lw=1.1, label=r"CO$_2$")
c.plot(wl, vip["CO_VIP"], color=CO, lw=1.1, label="CO")
for wv, name, xt in ((589.3, "Na 589 nm", 430), (766.5, "K 767 nm", 830)):
    c.annotate(name, xy=(wv, 2.42 if wv > 700 else 2.30), xytext=(xt, 2.78), fontsize=6.8, color="#8E3F38",
               ha="left", arrowprops=dict(arrowstyle="-", color="#8E3F38", lw=0.5))
c.set(xlim=(380, 1010), ylim=(0, 3.0), xlabel="Wavelength (nm)", ylabel="PLS VIP score")
c.legend(loc="center right", fontsize=6.8, handlelength=1.4, bbox_to_anchor=(1.0, 0.62))
ms.panel_label(c, "(c)", loc="below")
save(fig, "fig15_transfer_probe_vip")
