"""Figure 3 - five-step data preprocessing workflow. Width 1.0.

Uniform geometry: equal card widths and equal arrow lengths; every text box is checked against its card.
"""
import matplotlib.pyplot as plt

from diagram import arrow, badge, box, canvas, text
from style import INDIGO, LABEL, NAVY, TEXTW, save

W, H = TEXTW, 1.12
fig, ax = canvas(W, H)

steps = [  # (group, title, key fact, fill)
    ("Time axis", "Attachment 1", "corrected file", "#DEDBF0"),
    ("Time axis", "Heat 5 step", r"$\Delta t$ = 1 s", "#DEDBF0"),
    ("Spectra", "Common grid", "341–1023 nm", "#F1DCE6"),
    ("Lance flow", "O$_2$ counter", "24-s slope", "#D7DCEF"),
    ("Gas readings", "Gas units", "vol.%", "#FBE3CC"),
]

M = 0.03                 # outer margin
EW = 0.40                # width of the "Raw data" / "Clean data" end labels
AW = 0.17                # every arrow gap has the same length
n = len(steps)
CW = (W - 2 * M - 2 * EW - (n + 1) * AW) / n
y, h = 0.08, 0.72
yc = y + h / 2
checks = []


def gap_arrow(x):
    arrow(ax, (x + 0.025, yc), (x + AW - 0.025, yc), color=NAVY, lw=1.0, ms_=7)


x = M
checks.append((text(ax, x + EW / 2, yc, "Raw\ndata", size=8, color=NAVY, weight="bold", linespacing=1.15), x, x + EW))
x += EW
spans = {}
for i, (grp, title, fact, fill) in enumerate(steps):
    gap_arrow(x)
    x += AW
    box(ax, x, y, CW, h, fc=fill, ec="none", r=0.07)
    badge(ax, x + CW / 2, y + h - 0.14, str(i + 1), fc=INDIGO, size=6.6, r=0.07)
    checks.append((text(ax, x + CW / 2, y + h - 0.33, title, size=7.4, weight="bold", color="#222222"), x, x + CW))
    checks.append((text(ax, x + CW / 2, y + 0.15, fact, size=7.8, color=LABEL), x, x + CW))
    spans.setdefault(grp, [x, x + CW])[1] = x + CW
    x += CW
gap_arrow(x)
x += AW
checks.append((text(ax, x + EW / 2, yc, "Clean\ndata", size=8, color=NAVY, weight="bold", linespacing=1.15), x, x + EW))

for grp, (x0, x1) in spans.items():
    ax.plot([x0 + 0.03, x1 - 0.03], [y + h + 0.07] * 2, color="#9A9AB8", lw=0.8)
    text(ax, (x0 + x1) / 2, y + h + 0.15, grp, size=7.0, color="#555577", style="italic")

# make sure nothing spills out of its card (pad 0.03 in on each side)
fig.canvas.draw()
inv = ax.transData.inverted()
r = fig.canvas.get_renderer()
for t, x0, x1 in checks:
    bb = t.get_window_extent(r)
    (bx0, _), (bx1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    assert bx0 >= x0 + 0.03 and bx1 <= x1 - 0.03, f"text overflows: {t.get_text()!r} {bx0:.3f}-{bx1:.3f} vs {x0:.3f}-{x1:.3f}"
print(f"card width {CW:.3f} in, arrow length {AW - 0.05:.3f} in, all {len(checks)} labels inside their boxes")
save(fig, "fig03_preprocessing_workflow")
