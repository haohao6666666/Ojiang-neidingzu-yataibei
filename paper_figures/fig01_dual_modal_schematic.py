"""Figure 1 - dual-modal measurement at the converter.

Base image: base/fig01_base_render.png (AI-generated text-free illustration, 1448 x 1086 px).
All labels are added here so fonts and spelling stay under our control. Width 0.75.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from style import HERE, TEXTW, save

img = Image.open(os.path.join(HERE, "base", "fig01_base_render.png"))
W_PX, H_PX = img.size
w_in = TEXTW * 0.75
fig = plt.figure(figsize=(w_in, w_in * H_PX / W_PX))
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(img, extent=[0, W_PX, H_PX, 0], interpolation="none")
ax.set_xlim(0, W_PX)
ax.set_ylim(H_PX, 0)
ax.axis("off")

NAVY, TEAL, SALMON = "#1F2A78", "#1E8C7E", "#D4624A"


def label(x, y, title, sub=None, sub_color=NAVY, target=None, ha="left"):
    """White rounded label with an optional second line and a leader line to `target`."""
    t = ax.text(x, y, title, fontsize=7.6, weight="bold", color=NAVY, ha=ha, va="center", zorder=5,
                family="sans-serif",
                bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.5", fc="white", ec="none", alpha=0.92))
    if sub:
        ax.text(x, y + 40, sub, fontsize=6.8, color=sub_color, ha=ha, va="center", zorder=6, style="italic",
                weight="bold")
    if target is not None:
        ax.annotate("", xy=target, xytext=(x, y), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=0.9, mutation_scale=7, shrinkA=14, shrinkB=2))
    return t


label(70, 250, "Spectrometer")
ax.text(70, 296, "real time", fontsize=7, color="white", ha="left", va="center", weight="bold", zorder=6,
        bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.5", fc=TEAL, ec="none"))
label(615, 375, "Mouth flame", target=(470, 390))
label(615, 585, "O$_2$ lance", target=(450, 590))
label(615, 790, "Molten bath", target=(470, 820))
label(820, 230, "Flue-gas duct  ($\\Delta V_{\\mathrm{pipe}}$)", target=(880, 150))
label(880, 640, "Gas analyzer", target=(1085, 660))
ax.text(880, 690, "delayed", fontsize=7, color="white", ha="left", va="center", weight="bold", zorder=6,
        bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.5", fc=SALMON, ec="none"))
paths = save(fig, "fig01_dual_modal_schematic")
fig.savefig(paths[1], dpi=400, bbox_inches="tight", pad_inches=0.01)
