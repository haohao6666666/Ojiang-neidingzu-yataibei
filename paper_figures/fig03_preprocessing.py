"""Figure 3 - data preprocessing workflow (schematic, Euler-Award style: few words, one key fact per step). Width 1.0."""
from diagram import arrow, badge, box, canvas, text
from style import INDIGO, LABEL, NAVY, TEXTW, save

W, H = TEXTW, 1.05
fig, ax = canvas(W, H)

steps = [  # (group, title, key fact, fill)
    ("Time axis", "Attachment 1", "corrected file", "#DEDBF0"),
    ("Time axis", "Heat 5 step", r"$\Delta t$ = 1 s", "#DEDBF0"),
    ("Spectra", "Common grid", "341–1023 nm", "#F1DCE6"),
    ("Lance flow", "O$_2$ counter", "24-s slope", "#D7DCEF"),
    ("Gas readings", "Gas units", "vol.%", "#FBE3CC"),
]
ew, cw, gap = 0.50, 0.86, 0.11
y, h = 0.10, 0.62
x = 0.02
text(ax, x + ew / 2, y + h / 2, "Raw\ndata", size=8, color=NAVY, weight="bold", linespacing=1.2)
x += ew
arrow(ax, (x - 0.04, y + h / 2), (x + gap - 0.01, y + h / 2), color=NAVY, lw=1.0, ms_=7)
x += gap
spans = {}
for i, (grp, title, fact, fill) in enumerate(steps):
    box(ax, x, y, cw, h, fc=fill, ec="none", r=0.08)
    badge(ax, x + 0.13, y + h - 0.16, str(i + 1), fc=INDIGO, size=6.8, r=0.075)
    text(ax, x + 0.25, y + h - 0.16, title, size=7.3, ha="left", color="#222222", weight="bold")
    text(ax, x + cw / 2, y + 0.19, fact, size=8.2, color=LABEL)
    spans.setdefault(grp, [x, x + cw])[1] = x + cw
    x += cw
    if i < len(steps) - 1:
        arrow(ax, (x + 0.015, y + h / 2), (x + gap - 0.015, y + h / 2), color=NAVY, lw=0.9, ms_=6)
    x += gap
arrow(ax, (x - gap + 0.015, y + h / 2), (x + 0.07, y + h / 2), color=NAVY, lw=1.0, ms_=7)
text(ax, x + 0.08 + ew / 2 - 0.02, y + h / 2, "Clean\ndata", size=8, color=NAVY, weight="bold", linespacing=1.2)
for grp, (x0, x1) in spans.items():
    ax.plot([x0 + 0.03, x1 - 0.03], [y + h + 0.07] * 2, color="#9A9AB8", lw=0.8)
    text(ax, (x0 + x1) / 2, y + h + 0.15, grp, size=7.2, color="#555577", style="italic")
save(fig, "fig03_preprocessing_workflow")
