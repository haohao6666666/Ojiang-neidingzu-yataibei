"""Figure 9 - continuous spectral trend surfaces of the ten heats (log10 intensity). Width 1.0."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from style import FLAME, G6B, G6C, load, ms, save

d = load("fig09_surfaces.npz")
wl = d["wl"]
ZMIN, ZMAX = 2.3, 4.6
norm = Normalize(ZMIN, ZMAX)
ID = ["6B03205", "6B03206", "6B03270", "6B03271", "6C02057", "6C02058", "6C02059", "6C02060", "6C02107", "6C02108"]

W = ms.textwidth("apmcm")
fig = plt.figure(figsize=(W, W * 0.46))
left, right, top, bottom = 0.01, 0.895, 0.975, 0.075
cw, ch = (right - left) / 5, (top - bottom) / 2
for i in range(10):
    k = i + 1
    r, c = divmod(i, 5)
    ax = fig.add_axes([left + c * cw, top - (r + 1) * ch, cw * 1.04, ch * 1.06], projection="3d")
    t = d[f"t{k}"] / 60
    Z = d[f"z{k}"]
    TT, WW = np.meshgrid(t, wl, indexing="ij")
    ax.plot_surface(TT, WW, Z, facecolors=FLAME(norm(Z)), rstride=2, cstride=2, linewidth=0, antialiased=False,
                    shade=False, rasterized=True)
    ax.set_facecolor((1, 1, 1, 0))
    ax.view_init(elev=26, azim=-58)
    ax.set_box_aspect((1.35, 1.1, 0.72))
    ax.set_zlim(ZMIN, ZMAX)
    ax.set_xlim(0, 13)
    ax.set_ylim(340, 1025)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"].update(color=(0.85, 0.85, 0.85, 1), linewidth=0.3)
        axis.line.set_linewidth(0.5)
    ax.set_xticks([0, 6, 12])
    ax.set_yticks([400, 700, 1000])
    if r == 0:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        ax.set_yticklabels(["400", "700", ""])
        ax.set_xticklabels(["", "6", "12"])
    ax.set_zticks([2.5, 3.5, 4.5])
    ax.set_zticklabels([])
    ax.tick_params(labelsize=5.5, pad=-2.5, length=0)
    grp = ID[i][:2]
    ax.text2D(0.5, 0.90, f"Heat {k}", transform=ax.transAxes, ha="center", fontsize=7.5, color="#1F2A78")

fig.text(0.45, 0.012, "Axes: time (min) × wavelength (nm) × log$_{10}$ intensity", ha="center", va="bottom", fontsize=7.5, color="#1F2A78")
cax = fig.add_axes([0.925, 0.22, 0.013, 0.56])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=FLAME), cax=cax, ticks=[2.5, 3.0, 3.5, 4.0, 4.5])
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=7, length=2)
cb.set_label(r"log$_{10}$ intensity", fontsize=7.5)
cax.grid(False)
save(fig, "fig09_spectral_surfaces")
