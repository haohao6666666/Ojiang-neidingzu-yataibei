"""Figure 2 - framework of our work (Euler-Award three-band layout, short phrases only). Width 1.0."""
from diagram import arrow, box, canvas, text
from style import INDIGO, LABEL, NAVY, TEXTW, save

W, H = TEXTW, 2.75
fig, ax = canvas(W, H)
bands = [  # (label, y0, height, background)
    ("Data", 2.02, 0.66, "#ECEBF5"),
    ("Models", 0.98, 0.92, "#F6ECE4"),
    ("Results", 0.08, 0.78, "#E8EAF3"),
]
lx = 0.36
for name, y0, h, bg in bands:
    box(ax, 0.02, y0, W - 0.04, h, fc=bg, ec="none", r=0.08, z=0)
    text(ax, 0.19, y0 + h / 2, name, size=8.5, color=LABEL, weight="bold", rotation=90)

cols = [lx + 0.05, lx + 1.85, lx + 3.65]
cw = 1.50

# Data band: input -> checks -> arrival-time map
yD, hD = 2.12, 0.46
box(ax, cols[0], yD, cw, hD, fc="white", ec="none")
text(ax, cols[0] + cw / 2, yD + hD / 2, "10 heats\nspectra + O$_2$ + CO/CO$_2$", size=7.3, linespacing=1.3)
box(ax, cols[1], yD, cw, hD, fc="white", ec="none")
text(ax, cols[1] + cw / 2, yD + hD / 2, "4 consistency checks\n(Section 3)", size=7.3, linespacing=1.3)
box(ax, cols[2], yD, cw, hD, fc="#FBE3CC", ec="none")
text(ax, cols[2] + cw / 2, yD + hD / 2, "Arrival-time map\n$V(t_{arr}) - V(t) = \\Delta V_{pipe}$", size=7.3, linespacing=1.4)
for i in range(2):
    arrow(ax, (cols[i] + cw + 0.03, yD + hD / 2), (cols[i + 1] - 0.03, yD + hD / 2), color=NAVY, lw=1.0, ms_=7)

# Models band: three tasks
yM, hM = 1.10, 0.64
tasks = [("Task 1", "RLR", "$X = L + S$"), ("Task 2", "4 stages + PFD", r"$\tau(Q)$"), ("Task 3", "DSS + Smith", "real-time CO, CO$_2$")]
for c, (tag, name, sub) in zip(cols, tasks):
    box(ax, c, yM, cw, hM, fc="white", ec="none")
    box(ax, c, yM + hM - 0.2, cw, 0.2, fc=INDIGO, ec="none")
    ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle((c, yM + hM - 0.2), cw, 0.1, fc=INDIGO, ec="none", zorder=2))
    text(ax, c + cw / 2, yM + hM - 0.1, tag, size=7.5, color="white", weight="bold")
    text(ax, c + cw / 2, yM + 0.27, name, size=8, color="#222222", weight="bold")
    text(ax, c + cw / 2, yM + 0.10, sub, size=7.2, color="#444444")
# data feeds every task; Task 1 and Task 2 feed Task 3
for c in cols:
    arrow(ax, (c + cw / 2, yD - 0.03), (c + cw / 2, yM + hM + 0.03), color=NAVY, lw=0.9, ms_=6)
arrow(ax, (cols[1] + cw + 0.03, yM + 0.32), (cols[2] - 0.03, yM + 0.32), color="#D07A5E", lw=1.3, ms_=8)
text(ax, (cols[1] + cw + cols[2]) / 2, yM + 0.17, "delay", size=6.8, color="#B4605A")
arrow(ax, (cols[0] + cw * 0.70, yM - 0.03), (cols[2] + cw * 0.30, yM - 0.03), color="#D07A5E", lw=1.3, ms_=8, rad=0.18)
text(ax, cols[1] + cw / 2, yM - 0.23, "spectral basis", size=6.6, color="#B4605A")

# Results band: one key number per task
yR, hR = 0.18, 0.46
res = ["Spike F1 0.89\nhold-out error 2.0%", r"$\tau$ = 5 + 127.1/$Q$ s" "\nstages at 23/43/90% O$_2$",
       "CO $R^2$ 0.873 → 0.908\n99.6% fewer probe readings"]
for c, r in zip(cols, res):
    box(ax, c, yR, cw, hR, fc="white", ec="none")
    text(ax, c + cw / 2, yR + hR / 2, r, size=7.2, color=LABEL, linespacing=1.35)
save(fig, "fig02_framework")
