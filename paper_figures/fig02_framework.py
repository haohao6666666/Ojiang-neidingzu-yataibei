"""Figure 2 - framework of our work (methods only, no results).

Base image: base/fig02_base_bands.png (AI-generated text-free three-band layout, 1672 x 941 px).
Card icons are faded into watermarks so text can sit over them (Euler-Award style). Width 1.0.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

from style import HERE, TEXTW, save

img = Image.open(os.path.join(HERE, "base", "fig02_base_bands.png")).convert("RGB")
# move the duct-and-clock icon of the Foundation panel to the left so the formula fits on one line
_icon = img.crop((1178, 85, 1376, 236))
img.paste((254, 254, 254), (1178, 85, 1376, 236))
img.paste(_icon, (975, 100))
W_PX, H_PX = img.size
fig = plt.figure(figsize=(TEXTW, TEXTW * H_PX / W_PX))
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(img, extent=[0, W_PX, H_PX, 0], interpolation="none")
ax.set_xlim(0, W_PX)
ax.set_ylim(H_PX, 0)
ax.axis("off")

INK = "#1F2A44"
TEAL, AMBER, BLUE, CORAL, VIOLET = "#167A6E", "#C8742E", "#2F6A92", "#C2513A", "#5A4FB0"


def veil(x0, y0, x1, y1, color, alpha=0.62):
    """Fade the watermark icons inside a card body."""
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=color, ec="none", alpha=alpha, zorder=2))


def t(x, y, s, size=8.0, color=INK, weight=None, ha="left", va="center", rot=0, style=None, family="Arial Narrow"):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va, rotation=rot, style=style,
            family=family, zorder=5)


def bullets(x, y0, items, dy, size=7.6, color=INK, family="Arial Narrow"):
    for i, s in enumerate(items):
        t(x, y0 + i * dy, "•  " + s, size=size, color=color, family=family)


# band titles (vertical, in the left margin of each band)
t(30, 152, "Data & Foundation", size=7.2, color=TEAL, weight="bold", ha="center", rot=90)
t(30, 503, "Models", size=7.6, color="#4A4A4A", weight="bold", ha="center", rot=90)
t(30, 823, "Validation", size=7.4, color=VIOLET, weight="bold", ha="center", rot=90)

# top-left: problems to address (one line per icon)
probs = [(68, "Irregular time axis"), (127, "Saturated, flickering spectra"), (180, "Stepwise oxygen counter"),
         (236, "Ambiguous gas units")]
for y, s in probs:
    t(170, y, s, size=8.6)

# top-right: foundation
t(965, 64, "Foundation", size=8.4, color=TEAL, weight="bold")
t(1128, 64, "Oxygen-clock arrival map", size=8.6, weight="bold")
t(1205, 172, r"$V(t_{\mathrm{arr}}) - V(t) = \Delta V_{\mathrm{pipe}}$", size=9.2)

# middle: three task cards (header y 360-410, body 410-610)
cards = [(50, 547, "#FDF1E6", "Task 1  ·  Spectral Cleaning",
          ["Robust low-rank (RLR) model", "Pixel & frame outlier rules", "Continuous 3-D spectral surface"]),
         (597, 1053, "#E2EEF8", "Task 2  ·  Oxygen, Gas & Delay",
          ["Six continuous process curves", "Four-stage oxygen–gas relation", "Plug-flow delay  τ(Q)"]),
         (1135, 1622, "#FBE7E2", "Task 3  ·  Real-Time Soft Sensor",
          ["Causal spectral features", "Soft-gated heat combination", "Smith correction & conformal band"])]
for x0, x1, body, head, items in cards:
    veil(x0 + 4, 412, x1 - 4, 606, body)
    t((x0 + x1) / 2, 385, head, size=8.6, color="white", weight="bold", ha="center")
    bullets(x0 + 26, 452, items, 46, size=8.0)

# dependency labels
t(1093, 462, "delay", size=7.2, color=INK, ha="center", style="italic")
t(820, 645, "spectral basis", size=7.6, color=INK, ha="center", style="italic")

# bottom: validation and sensitivity
t(225, 772, "Model Validation", size=8.4, color=VIOLET, weight="bold")
bullets(225, 812, ["Outlier injection test", "Leave-one-heat-out evaluation", "Leakage audit (future frames, test heat)"], 34, size=7.6)
t(880, 772, "Sensitivity Analysis", size=8.4, color=VIOLET, weight="bold")
bullets(880, 812, ["Outlier threshold & rank", "Duct volume ±10%", "Probe sampling interval"], 34, size=7.6)

paths = save(fig, "fig02_framework")
fig.savefig(paths[1], dpi=400, bbox_inches="tight", pad_inches=0.01)
