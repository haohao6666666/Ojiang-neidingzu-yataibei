"""Figure 1 - (a) a real converter shop, (b) the dual-modal measurement at the converter. Width 1.0.

(a) base/fig01a_photo_qarmet_cc0.jpg: "Конвертерный цех Qarmet.jpg" by DDEKK, Wikimedia Commons, CC0
    (cropped to 4:3, https://commons.wikimedia.org/wiki/File:Конвертерный_цех_Qarmet.jpg).
(b) base/fig01_base_render.png: AI-generated text-free illustration; every label is added here.
"""
import os

import matplotlib.pyplot as plt
from PIL import Image

from style import HERE, TEXTW, save

photo = Image.open(os.path.join(HERE, "base", "fig01a_photo_qarmet_cc0.jpg")).convert("RGB")
render = Image.open(os.path.join(HERE, "base", "fig01_base_render.png")).convert("RGB")

GAP = 0.025                      # gap between panels, fraction of figure width
PW = (1 - GAP) / 2               # panel width (fraction)
ASPECT = 3 / 4                   # both panels are 4:3
CAP = 0.075                      # room below the panels for "(a)" / "(b)", fraction of figure height
W = TEXTW
H = W * PW * ASPECT / (1 - CAP)
fig = plt.figure(figsize=(W, H))

NAVY, TEAL, SALMON = "#1F2A78", "#1E8C7E", "#D4624A"
FS = 6.0


def panel(x0, img):
    ax = fig.add_axes([x0, CAP, PW, 1 - CAP])
    w, h = img.size
    ax.imshow(img, extent=[0, w, h, 0], interpolation="none")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    return ax


axa = panel(0, photo)
axb = panel(PW + GAP, render)
for x0, tag in ((0, "(a)"), (PW + GAP, "(b)")):
    fig.text(x0 + PW / 2, CAP / 2, tag, ha="center", va="center", fontsize=8.5, weight="bold", color="#222222")

# (b) labels on the render (coordinates in render pixels, 1448 x 1086); arrows start at the label edge
import matplotlib.patheffects as pe

HALO = [pe.Stroke(linewidth=2.4, foreground="white"), pe.Normal()]


def label(x, y, text, target=None, ha="left"):
    kw = dict(fontsize=FS, weight="bold", color=NAVY, ha=ha, va="center", zorder=5,
              bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.5", fc="white", ec="none", alpha=0.95))
    if target is None:
        axb.text(x, y, text, **kw)
        return
    a = axb.annotate(text, xy=target, xytext=(x, y), **kw,
                     arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=0.9, mutation_scale=7, shrinkA=0, shrinkB=1))
    a.arrow_patch.set_path_effects(HALO)


def tag(x, y, text, color):
    axb.text(x, y, text, fontsize=FS - 0.3, color="white", weight="bold", ha="left", va="center", zorder=6,
             bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.5", fc=color, ec="none"))


label(40, 500, "Spectrometer", target=(140, 425))
tag(40, 555, "real time", TEAL)
label(640, 330, "Mouth flame", target=(470, 380))
label(640, 560, "O$_2$ lance", target=(450, 560))
label(640, 800, "Molten bath", target=(480, 830))
label(760, 250, "Flue-gas duct ($\\Delta V_{\\mathrm{pipe}}$)", target=(900, 155))
label(1010, 470, "Gas analyzer", target=(1150, 560))
tag(1010, 525, "delayed", SALMON)

paths = save(fig, "fig01_dual_modal_schematic")
fig.savefig(paths[1], dpi=450, bbox_inches="tight", pad_inches=0.01)
