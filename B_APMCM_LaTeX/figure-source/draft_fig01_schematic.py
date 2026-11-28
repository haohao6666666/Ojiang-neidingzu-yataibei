"""Figure 1 - dual-modal measurement at the converter (schematic). Width 0.7."""
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

from diagram import arrow, box, canvas, text
from style import INDIGO, LABEL, NAVY, PEACH, SALMON, TEXTW, save

W, H = TEXTW * 0.7, 2.55
fig, ax = canvas(W, H)
cx = 1.05

# converter vessel: conical mouth, straight barrel, rounded bottom; refractory lining; molten bath
def profile(r_mouth, r_body, y_mouth, y_cone, y_bot, cx_):
    th = np.linspace(np.pi, 2 * np.pi, 40)
    bottom = np.c_[cx_ + r_body * np.cos(th), y_bot + 0.55 * r_body * np.sin(th) + 0.0]
    right = [[cx_ + r_body, y_cone], [cx_ + r_mouth, y_mouth]]
    left = [[cx_ - r_mouth, y_mouth], [cx_ - r_body, y_cone]]
    return np.vstack([left, bottom, right])
ax.add_patch(Polygon(profile(0.21, 0.50, 1.22, 0.86, 0.42, cx), closed=True, fc=NAVY, ec="none", zorder=2))
ax.add_patch(Polygon(profile(0.15, 0.43, 1.22, 0.84, 0.44, cx), closed=True, fc="#C9C4E0", ec="none", zorder=3))
th = np.linspace(np.pi, 2 * np.pi, 40)
bath = np.vstack([[[cx - 0.43, 0.56]], np.c_[cx + 0.43 * np.cos(th), 0.44 + 0.55 * 0.43 * np.sin(th)], [[cx + 0.43, 0.56]]])
ax.add_patch(Polygon(bath, closed=True, fc=SALMON, ec="none", zorder=4))
ax.add_patch(Rectangle((cx - 0.43, 0.53), 0.86, 0.05, fc=PEACH, ec="none", zorder=4))

# oxygen lance
ax.add_patch(Rectangle((cx - 0.025, 0.90), 0.05, 1.46, fc="#8C8CA3", ec="none", zorder=6))
text(ax, cx - 0.07, 2.30, "O$_2$ lance", size=7, ha="right", color="#555566")

# mouth flame
ft = np.linspace(0, np.pi, 40)
for scale, col, a in ((1.0, PEACH, 0.85), (0.6, "#FCEBD2", 0.95)):
    fx = cx + 0.18 * scale * np.cos(ft) * (1 + 0.15 * np.sin(5 * ft))
    fy = 1.20 + 0.42 * scale * np.sin(ft) ** 1.4
    ax.add_patch(Polygon(np.c_[fx, fy], closed=True, fc=col, ec="none", alpha=a, zorder=5))

# hood + duct (as a thick stroke) with gas parcels
duct = np.array([[cx + 0.30, 1.80], [cx + 0.30, 2.18], [3.42, 2.18], [3.42, 1.05]])
ax.add_patch(Polygon([[cx - 0.40, 1.55], [cx + 0.40, 1.55], [cx + 0.42, 1.88], [cx + 0.18, 1.88]],
                     closed=True, fc="#D4D2E6", ec="none", zorder=1))
ax.plot(duct[:, 0], duct[:, 1], color="#D4D2E6", lw=15, solid_capstyle="butt", solid_joinstyle="round", zorder=1)
for (x0, y0), (x1, y1) in [((cx + 0.30, 1.95), (cx + 0.30, 2.14)),
                           ((3.42, 1.95), (3.42, 1.62))]:
    arrow(ax, (x0, y0), (x1, y1), color=NAVY, lw=0.8, ms_=6, z=4)
for start, end in ((1.73, 2.37), (2.66, 3.33)):
    arrow(ax, (start, 2.18), (end, 2.18), color=NAVY, lw=1.15, ms_=8, z=4)
for x, y in [(1.62, 2.18), (2.55, 2.18), (cx + 0.30, 2.05), (3.42, 1.30)]:
    ax.add_patch(Circle((x, y), 0.045, fc=SALMON, ec="white", lw=0.5, zorder=5))

# duct volume bracket
yb = 2.40
ax.plot([cx + 0.30, 3.42], [yb, yb], color=LABEL, lw=0.8)
for xb in (cx + 0.30, 3.42):
    ax.plot([xb, xb], [yb - 0.06, yb + 0.06], color=LABEL, lw=0.8)
text(ax, (cx + 0.30 + 3.42) / 2, yb + 0.12, r"$\Delta V_{\mathrm{pipe}}$", size=8.5, color=LABEL)

# analyzer at the bend
box(ax, 3.58, 1.80, 0.40, 0.38, fc=INDIGO, ec="none", r=0.05, z=6)
text(ax, 3.78, 1.99, "Raman\nanalyzer", size=6.4, color="white", weight="bold", linespacing=1.1, z=7)
ax.plot([3.53, 3.58], [1.99, 1.99], color=INDIGO, lw=1.2, zorder=6)
box(ax, 3.52, 1.46, 0.52, 0.24, fc="#F3D9CF", ec="none", r=0.05, z=6)
text(ax, 3.78, 1.58, "≈30 s later", size=7, color="#9E4A3A", z=7)

# spectrometer looking at the flame
box(ax, 0.04, 1.30, 0.77, 0.34, fc=INDIGO, ec="none", r=0.05, z=6)
text(ax, 0.425, 1.47, "Spectrometer", size=6.4, color="white", weight="bold", z=7)
ax.plot([0.81, cx - 0.12], [1.47, 1.38], color=INDIGO, lw=0.8, ls=(0, (2.5, 1.5)), zorder=5)
box(ax, 0.155, 0.98, 0.54, 0.24, fc="#E1DEF0", ec="none", r=0.05, z=6)
text(ax, 0.425, 1.10, "real time", size=7, color=LABEL, z=7)
save(fig, "fig01_dual_modal_schematic")
