"""Figure 8 - outlier detection and reconstruction (Heat 9, the heat with most flagged pixels). Width 1.0."""
import numpy as np

from style import CO, GREY, INDIGO, LAVENDER, MODEL, PEACH, load, ms, save
GOLD = PEACH

d = load("fig08_heat09_outliers.npz")
t = d["t"] / 60
wl = d["wl"]
k = int(np.where(d["cand"] == 1097)[0][0])          # bright burst frame at 9.14 min (64% of pixels flagged)
fi = int(d["cand"][k])

fig, (a, b) = ms.subplots(1, 2, frac=1.0, aspect=0.40, gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.06})

# (a) 600-nm trace
o = d["out"].astype(bool)
a.plot(t, d["x"], ".", ms=1.6, color=GREY, alpha=0.55, label="Observed", rasterized=True)
a.plot(t[o], d["x"][o], "o", ms=3.2, mfc="none", mec=CO, mew=0.7, label="Outlier")
a.plot(t, d["rec"], color=LAVENDER, lw=0.45, alpha=0.9, label="Reconstruction")
a.plot(t, d["trend"], color=INDIGO, lw=1.8, label="Trend")
a.axvline(t[fi], color=CO, ls=":", lw=0.9)
a.set(xlabel="Time (min)", ylabel=r"log$_{10}$ intensity at 600 nm", ylim=(2.45, 5.35), xlim=(0, t[-1]))
a.legend(loc="upper left", ncol=4, fontsize=7, columnspacing=0.8, handletextpad=0.3, markerscale=1.5, bbox_to_anchor=(0, 1.04))
ms.panel_label(a, "(a)", loc="below")

# (b) the anomalous frame
x = d["cand_X"][k]
ok = d["cand_out"][k].astype(bool)
sat = x >= np.log10(65535) - 0.005          # 16-bit detector limit
b.plot(wl, x, color=GREY, lw=0.8, label="Observed")
b.plot(wl[ok & ~sat], x[ok & ~sat], "o", ms=1.6, color=CO, mec="none", label="Outlier")
b.plot(np.where(sat, wl, np.nan), np.where(sat, x, np.nan), color=GOLD, lw=3.0, solid_capstyle="butt")
b.text(wl[sat].mean(), x[sat].max() + 0.07, "saturated", ha="center", va="bottom", fontsize=7, color="#9A6A2E")
b.plot(wl, d["cand_rec"][k], color=MODEL, lw=1.5, label="RLR")
b.plot(wl, d["cand_trend"][k], color=GOLD, lw=1.1, ls="--", label="Trend")
b.set(xlabel="Wavelength (nm)", ylabel=r"log$_{10}$ intensity", xlim=(340, 1025), ylim=(1.95, 5.15), yticks=[3.0, 3.5, 4.0, 4.5, 5.0])
b.legend(loc="lower right", fontsize=7, ncol=2, handlelength=1.2, columnspacing=0.7, markerscale=2, bbox_to_anchor=(1.03, -0.03))
ms.panel_label(b, "(b)", loc="below")
save(fig, "fig08_outlier_reconstruction")
