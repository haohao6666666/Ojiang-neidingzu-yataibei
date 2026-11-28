"""Figure 10 - six continuous process curves of Heat 3 with the four blowing stages. Width 1.0."""
import numpy as np

from style import CO, CO2, GREY, O2, STAGE_NAME, load, ms, save, stage_shading

K = 3
d = load("fig10_11_process.npz")
t = d[f"t{K}"] / 60
tb = d[f"tbs{K}"] / 60
edges = [0, *tb, t[-1]]
dt = 0.5

fig, axs = ms.subplots(3, 2, frac=1.0, aspect=0.72, sharex=True, gridspec_kw={"hspace": 0.03, "wspace": 0.05})
(aQ, aV), (a2, a1), (e2, e1) = axs

# (a) lance flow
aQ.plot(t, d[f"Q{K}"], color=O2, lw=1.6)
aQ.set(ylim=(0, 7.5), ylabel=r"$Q$ (m³/s)")

# (b) cumulative oxygen
V = d[f"V{K}"]
aV.plot(t, V, color=O2, lw=1.6)
aV.set(ylabel=r"$V$ (m³)", ylim=(0, 3900))

# (c) CO2 and (d) CO readings
for ax, raw_k, key, col, name in ((a2, "co2_raw", "co2", CO2, r"CO$_2$"), (a1, "co_raw", "co", CO, "CO")):
    y = d[f"{key}{K}"]
    ax.plot(t, d[f"{raw_k}{K}"], ".", ms=1.0, color=GREY, alpha=0.6, rasterized=True)
    ax.plot(t, y, color=col, lw=1.5)
    ax.set(ylabel=f"{name} (vol.%)", ylim=(0, y.max() * 1.3))

# (e, f) cumulative emissions
for ax, key, col, name in ((e2, "Eco2", CO2, r"$E_{\mathrm{CO_2}}$"), (e1, "Eco", CO, r"$E_{\mathrm{CO}}$")):
    E = d[f"{key}{K}"] / 1e3
    ax.plot(t, E, color=col, lw=1.5)
    ax.set(ylabel=name + r" (10³ vol.%$\cdot$s)", ylim=(0, E.max() * 1.18), xlabel="Time (min)")

labels = ["(a) Lance flow", "(b) Cumulative oxygen", r"(c) CO$_2$ reading", "(d) CO reading",
          r"(e) Cumulative CO$_2$", "(f) Cumulative CO"]
for ax, lab in zip(axs.flat, labels):
    stage_shading(ax, edges, labels=False)
    ax.set_xlim(0, t[-1])
    ax.text(0.02, 0.95, lab.split(" ")[0], transform=ax.transAxes, ha="left", va="top", fontsize=8, weight="bold", color="#333333")
for ax in axs[0]:
    for i in range(4):
        ax.text((edges[i] + edges[i + 1]) / 2, 1.02, STAGE_NAME[i], transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7.5, style="italic", color="#4A4A4A")
save(fig, "fig10_process_curves_heat03")
