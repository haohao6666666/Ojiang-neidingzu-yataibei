"""Figure 7 - low-rank plus sparse decomposition of Heat 3 (X = L + S). Width 1.0."""
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from style import DIV, FLAME, load, ms, save

d = load("fig07_heat03_decomposition.npz")
t, wl = d["t"] / 60, d["wl"].astype(float)
X, L, S = (d[k].astype(float) for k in ("X", "L", "S"))
ext = [0, t[-1], wl[0], wl[-1]]
vmin, vmax = 2.6, 4.45
sv = float(np.nanpercentile(np.abs(S), 99.9))

import matplotlib.pyplot as plt
fig = plt.figure(figsize=ms.figsize(1.0, 0.36))
gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 0.045, 0.30, 1, 0.045], wspace=0.10)
axs = [fig.add_subplot(gs[0, 0])]
axs += [fig.add_subplot(gs[0, 1], sharey=axs[0]), fig.add_subplot(gs[0, 4], sharey=axs[0])]
caxes = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 5])]
plt.setp(axs[1].get_yticklabels(), visible=False)
plt.setp(axs[2].get_yticklabels(), visible=False)
tags = [r"Observed $\mathbf{X}$", r"Low-rank $\mathbf{L}$ (rank 3)", r"Sparse $\mathbf{S}$"]
for ax, M, tag, lab in zip(axs, (X, L, S), tags, ("(a)", "(b)", "(c)")):
    if M is S:
        im_s = ax.imshow(M.T, aspect="auto", origin="lower", extent=ext, cmap=DIV, vmin=-sv, vmax=sv,
                         interpolation="nearest", rasterized=True)
    else:
        im = ax.imshow(M.T, aspect="auto", origin="lower", extent=ext, cmap=FLAME, vmin=vmin, vmax=vmax,
                       interpolation="nearest", rasterized=True)
    ax.grid(False)
    ax.set_xlabel("Time (min)")
    ax.set_xticks([0, 4, 8])
    ms.panel_label(ax, lab, loc="below")
axs[0].set_ylabel("Wavelength (nm)")

for wv, name in ((767, "K 767 nm"), (589, "Na 589 nm")):
    axs[2].annotate(name, xy=(t[-1] * 0.98, wv), xytext=(t[-1] * 0.42, wv + 75), fontsize=7, color="#8E3F38",
                    ha="center", arrowprops=dict(arrowstyle="-|>", color="#8E3F38", lw=0.6, mutation_scale=6),
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))

for cax, mappable, label in ((caxes[0], im, r"log$_{10}$ intensity"), (caxes[1], im_s, r"Sparse residual (log$_{10}$)")):
    cb = fig.colorbar(mappable, cax=cax, ticks=[3.0, 3.5, 4.0] if mappable is im else [-0.08, -0.04, 0, 0.04, 0.08])
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_visible(False)
    cb.set_label(label, fontsize=7.5)
    cax.grid(False)
save(fig, "fig07_rlr_decomposition")
