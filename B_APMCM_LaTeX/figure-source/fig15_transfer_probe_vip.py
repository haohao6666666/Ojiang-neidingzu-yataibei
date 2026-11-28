"""Figure 15 - cross-heat transfer, probe-saving trade-off and wavelength importance. Width 1.0."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from style import CO, CO2, DATA, DIV, GOLD, GREY, LABEL, ms, save

mat = pd.read_csv(os.path.join(DATA, "fig15_transfer.csv"), index_col=0).to_numpy()
summ = pd.read_csv(os.path.join(DATA, "q3_loho_ablation_summary.csv")).set_index("Mode")
vip = pd.read_csv(os.path.join(DATA, "q3_pls_vip_wavelength.csv"))

fig = plt.figure(figsize=ms.figsize(1.0, 1.37))


def panel(bottom, top, title, note):
    fig.text(0.5, top - 0.020, title, ha="center", va="center",
             fontsize=10, weight="bold", color=LABEL)
    fig.text(0.5, bottom + 0.012, note, ha="center", va="bottom",
             fontsize=8, linespacing=1.25, color="#303040")


panel(0.692, 0.990, "(a) Cross-heat transfer",
      "The grey diagonal marks in-sample fits. Off-diagonal scores show poor single-heat transfer.")
panel(0.333, 0.673, "(b) Probe savings and prediction accuracy",
      "Intervals of 120, 60, 30 and 15 s save 99.6%, 99.2%, 98.3% and 96.7% of probe readings.\n"
      r"At 120 s, Smith correction keeps CO $R^2$ above 0.9. Holding the lagged reading gives 0.59.")
panel(0.014, 0.313, "(c) Wavelength importance",
      "The potassium and sodium lines have the highest PLS VIP scores.")

a = fig.add_axes([0.33, 0.760, 0.34, 0.180])
cax = fig.add_axes([0.652, 0.768, 0.016, 0.164])
b = fig.add_axes([0.150, 0.466, 0.75, 0.157])
c = fig.add_axes([0.150, 0.079, 0.75, 0.185])

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
cb = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1])
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=6.5, length=2)
cb.set_label(r"Transfer R²", fontsize=7, labelpad=1)
cax.grid(False)

# (b) accuracy versus probe interval
modes = ["Combined_SoftGated_Hybrid", "Hybrid_Smith_Pulse_120s_99.6pctSave", "Hybrid_Smith_Pulse_60s_99.2pctSave",
         "Hybrid_Smith_Pulse_30s_98.3pctSave", "Hybrid_Smith_Pulse_15s_96.7pctSave", "Hybrid_Smith_Continuous_DualModal"]
naive = [None, "Naive_Lagged_Sensor_Pulse_120s_99.6pctSave", "Naive_Lagged_Sensor_Pulse_60s_99.2pctSave",
         "Naive_Lagged_Sensor_Pulse_30s_98.3pctSave", "Naive_Lagged_Sensor_Pulse_15s_96.7pctSave",
         "Naive_Lagged_Sensor_Continuous_DualModal"]
x = np.arange(6)
b.axvspan(0.6, 1.4, color=GOLD, alpha=0.35, lw=0)
for col, key, mk in ((CO, "CO_Mean_R2", "o"), (CO2, "CO2_Mean_R2", "s")):
    ours = summ.loc[modes, key].to_numpy()
    lag = np.array([np.nan] + [summ.loc[m, key] for m in naive[1:]])
    b.plot(x, ours, mk + "-", color=col, lw=1.6, ms=4, mec="white", mew=0.5)
    b.plot(x, lag, mk + "--", color=col, lw=0.9, ms=3.4, mfc="white", mec=col, alpha=0.85)
co2_lag120 = summ.loc[naive[1], "CO2_Mean_R2"]
axis_floor = -0.25
offscale_label = f"−{abs(co2_lag120):.2f} " + r"(CO$_2$, off-scale)"
b.annotate("", xy=(1, axis_floor), xytext=(1, axis_floor + 0.18),
           arrowprops=dict(arrowstyle="-|>", color=CO2, linewidth=1, mutation_scale=9,
                           shrinkA=0, shrinkB=0), annotation_clip=False)
co120, lag120 = summ.loc[modes[1], "CO_Mean_R2"], summ.loc[naive[1], "CO_Mean_R2"]
b.annotate(f"{co120:.3f}", xy=(1, co120), xytext=(1.25, 1.03), fontsize=7, color=CO, weight="bold",
           arrowprops=dict(arrowstyle="-", color=CO, lw=0.5))
b.annotate(f"{lag120:.2f}", xy=(1, lag120), xytext=(1.35, 0.50), fontsize=7, color=CO,
           arrowprops=dict(arrowstyle="-", color=CO, lw=0.5))
b.set_yticks([axis_floor, 0, 0.25, 0.5, 0.75, 1.0])
b.set_xticks(x, ["None", "120", "60", "30", "15", "Cont."], fontsize=7.2)
b.set(ylim=(axis_floor, 1.1), ylabel=r"Mean LOHO R²", xlim=(-0.35, 5.35))
b.set_xlabel("Probe interval (s)")
b.plot([], [], "-", color="#444444", lw=1.6, label="DSS + Smith correction")
b.plot([], [], "--", color="#444444", lw=0.9, label="Hold last lagged reading")
method_keys, _ = b.get_legend_handles_labels()
gas_keys = [
    Line2D([], [], color=CO, marker="o", markersize=3.5, label="CO"),
    Line2D([], [], color=CO2, marker="s", markersize=3.5, label=r"CO$_2$"),
]
offscale_key = Line2D([], [], marker=r"$\downarrow$", linestyle="none", markersize=7,
                      color=CO2, label=offscale_label)
fig.legend(handles=[*method_keys, *gas_keys, offscale_key], loc="upper center",
           bbox_to_anchor=(0.5, 0.413), ncol=5, fontsize=7, handlelength=1.55,
           columnspacing=1.1)

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
save(fig, "fig15_transfer_probe_vip")
