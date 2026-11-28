"""mcm_style: O-award figure style for MCM/ICM/APMCM papers.

Usage (copy this file next to your plotting code, or add its folder to sys.path):

    import mcm_style as ms
    ms.use_style(palette="euler", profile="apmcm")      # once, before plotting
    fig, ax = ms.subplots(frac=1.0, aspect=0.45)        # sized to the real text width
    ax.plot(x, y, color=ms.C.primary)
    ms.save(fig, "figures/fig_q1_rank_selection")        # -> .pdf (vector) + .png (300 dpi)

Design rules baked in (see SKILL.md for the reasoning):
* figures are drawn at their final printed width, so 8-9 pt text stays 8-9 pt in the PDF;
* serif text (Times) + STIX math to match a LaTeX body set in Times (newtx);
* no top/right spines, light dashed y-grid, no in-figure titles (the caption is the title);
* one palette per paper; `primary` = our model, `baseline` = grey, `good`/`bad` only for signed change.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

# --------------------------------------------------------------------------------------
# Text widths (inches) of the two templates this skill targets.
#   mcmthesis   : A4, margin = 1in            -> 210mm - 50.8mm  = 159.2mm = 6.27in
#   apmcmthesis : A4, left/right = 31.8mm     -> 210mm - 63.6mm  = 146.4mm = 5.76in
# --------------------------------------------------------------------------------------
TEXTWIDTH = {"mcm": 6.27, "apmcm": 5.76}

# --------------------------------------------------------------------------------------
# Palettes. Each was distilled from k-means clustering of the coloured pixels in O-award
# papers (see references/o-award-figure-analysis.md). Common structure: a cool dominant hue
# family (60-75 % of coloured area) + ONE warm accent (10-25 %) for the thing to look at.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Palette:
    name: str
    primary: str      # our model / main series / hero bars
    secondary: str    # second series of the same family (variant, "after", fit)
    tertiary: str     # third series
    accent: str       # warm highlight: the one thing the reader must see
    accent2: str      # second warm tone (rarely needed)
    baseline: str     # baseline / reference / "others" - always neutral
    dark: str         # text-like dark tone for annotations / outlines
    light: str        # pale fill of primary family (bands, shaded regions, table header)
    light_accent: str  # pale fill of accent family (highlight regions / rows)
    good: str         # signed change: improvement (use ONLY for deltas)
    bad: str          # signed change: deterioration (use ONLY for deltas)
    series: tuple = field(default_factory=tuple)  # categorical order, lightness-spread & CVD-checked

    def cycle(self) -> list[str]:
        """Order for multi-series plots. First 3 pass CVD (dE>18) and grayscale (dL>12) checks;
        beyond 3 series ALSO vary marker / linestyle (see SKILL.md)."""
        return list(self.series) if self.series else [self.primary, self.accent, self.secondary, self.tertiary]


PALETTES: dict[str, Palette] = {
    # 2519935 (2025 D, Euler award): indigo/lavender family + peach/terracotta accent.
    "euler": Palette("euler", primary="#4C5D91", secondary="#A99BCF", tertiary="#8FA7D6",
                     accent="#E8956A", accent2="#C46A4A", baseline="#9A9A9A", dark="#2F3A5C",
                     light="#E4E1F1", light_accent="#FBE3D0", good="#3E7CB1", bad="#C8553D",
                     series=("#3B4A7E", "#E8956A", "#8E80C2", "#A9B8DE", "#C46A4A")),
    # 2611034 (2026 C): teal family + salmon accent.
    "teal": Palette("teal", primary="#2F7F91", secondary="#7DB3C0", tertiary="#AFD5DC",
                    accent="#E2876A", accent2="#B85C62", baseline="#9AA3A8", dark="#244A55",
                    light="#DCEEF1", light_accent="#F8DDD2", good="#2F7F91", bad="#C0504D",
                    series=("#2F7F91", "#E2876A", "#1F4F5B", "#8CC3CE", "#B85C62")),
    # 2604153 (2026 E) / 2500759 (2025 C): steel-blue family + sand-gold accent.
    "steel": Palette("steel", primary="#206FAA", secondary="#6C92AB", tertiary="#97BED5",
                     accent="#D9A441", accent2="#B5524A", baseline="#9C9C9C", dark="#24435C",
                     light="#DDEAF4", light_accent="#F6EACB", good="#206FAA", bad="#B5524A",
                     series=("#206FAA", "#D9A441", "#1D3A52", "#B3D0E5", "#B5524A")),
    # 2625185 / 2627711 / 2608346: slate greys + single orange accent (most "serious" look).
    "slate": Palette("slate", primary="#3E5C76", secondary="#748CA3", tertiary="#A9BCCD",
                     accent="#E07B39", accent2="#5E9E6E", baseline="#A0A0A0", dark="#263645",
                     light="#E3EAF0", light_accent="#FBE2D0", good="#3E7CB1", bad="#C8553D",
                     series=("#3E5C76", "#E07B39", "#1F2D3A", "#9FB4C7", "#5E9E6E")),
}


class _ColorNS:
    """Attribute access to the active palette: ms.C.primary, ms.C.accent, ..."""

    def __init__(self) -> None:
        self._p = PALETTES["euler"]

    def __getattr__(self, item):  # noqa: D401
        return getattr(self._p, item)

    @property
    def palette(self) -> Palette:
        return self._p


C = _ColorNS()
_PROFILE = {"profile": "mcm"}


def use_style(palette: str = "euler", profile: str = "mcm", serif: bool = True, base_pt: float = 9.0) -> Palette:
    """Activate a palette + rcParams. profile: 'mcm' (mcmthesis) or 'apmcm' (apmcmthesis)."""
    if palette not in PALETTES:
        raise ValueError(f"unknown palette {palette!r}; choose from {sorted(PALETTES)}")
    if profile not in TEXTWIDTH:
        raise ValueError("profile must be 'mcm' or 'apmcm'")
    C._p = PALETTES[palette]
    _PROFILE["profile"] = profile
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.expanduser("~"), ".cache", "mpl-mcm"))
    rc = {
        "font.family": "serif" if serif else "sans-serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stix" if serif else "dejavusans",
        "font.size": base_pt,
        "axes.labelsize": base_pt,
        "axes.titlesize": base_pt,          # titles discouraged; if used keep them small
        "xtick.labelsize": base_pt - 1,
        "ytick.labelsize": base_pt - 1,
        "legend.fontsize": base_pt - 1,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#222222",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#D9D9D9",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=C.palette.cycle()),
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "patch.linewidth": 0.6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,                 # embed TrueType -> selectable text, no Type-3
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "image.cmap": "viridis",
    }
    mpl.rcParams.update(rc)
    mpl.colormaps.register(cmap("seq"), name=f"mcm_{palette}_seq", force=True)
    mpl.colormaps.register(cmap("div"), name=f"mcm_{palette}_div", force=True)
    return C.palette


def textwidth(profile: str | None = None) -> float:
    return TEXTWIDTH[profile or _PROFILE["profile"]]


def figsize(frac: float = 1.0, aspect: float = 0.62, profile: str | None = None) -> tuple[float, float]:
    """(width, height) in inches. frac = share of \\textwidth the figure will occupy in LaTeX.
    Use the SAME frac in \\includegraphics[width=<frac>\\textwidth]{...} so nothing is rescaled."""
    w = textwidth(profile) * frac
    return (w, w * aspect)


def subplots(nrows: int = 1, ncols: int = 1, frac: float = 1.0, aspect: float | None = None, **kw):
    if aspect is None:
        aspect = 0.62 * nrows / max(ncols, 1) if ncols > 1 else 0.62 * nrows
    kw.setdefault("constrained_layout", True)
    return plt.subplots(nrows, ncols, figsize=figsize(frac, aspect), **kw)


def cmap(kind: str = "seq", palette: str | None = None) -> LinearSegmentedColormap:
    """'seq': white -> light -> primary -> dark (magnitudes, densities, heatmaps).
    'div': accent <- white -> primary (signed values: residuals, correlations, deltas)."""
    p = PALETTES[palette] if palette else C.palette
    if kind == "seq":
        return LinearSegmentedColormap.from_list(f"{p.name}_seq", ["#FFFFFF", p.light, p.tertiary, p.primary, p.dark])
    if kind == "div":
        return LinearSegmentedColormap.from_list(f"{p.name}_div", [p.accent2, p.accent, "#F7F7F7", p.tertiary, p.primary])
    raise ValueError("kind must be 'seq' or 'div'")


def save(fig, path_no_ext: str, formats: Sequence[str] = ("pdf", "png"), dpi: int = 300, close: bool = True) -> list[str]:
    """Save vector PDF (for LaTeX) + 300-dpi PNG (for quick viewing / Word). Returns paths."""
    os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
    out = []
    for f in formats:
        p = f"{path_no_ext}.{f}"
        fig.savefig(p, dpi=dpi)
        out.append(p)
    if close:
        plt.close(fig)
    return out


# --------------------------------------------------------------------------------------
# Annotation helpers - the "O-award touches": direct labels, signed deltas, named regions,
# reference lines, uncertainty bands, inset zoom.
# --------------------------------------------------------------------------------------

def label_bars(ax, bars, fmt: str = "{:.1f}", pad: float = 2, color: str | None = None, fontsize: float | None = None):
    """Write the value on top of each bar (replaces a separate table for small comparisons)."""
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, pad),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=fontsize or mpl.rcParams["xtick.labelsize"], color=color or C.dark)


def delta_label(ax, x: float, y: float, delta: float, fmt: str = "{:+.1f}%", dy: float = 4, fontsize=None):
    """Signed change label: good colour for +, bad colour for -. Use only for real deltas."""
    col = C.good if delta >= 0 else C.bad
    ax.annotate(fmt.format(delta), (x, y), xytext=(0, dy), textcoords="offset points",
                ha="center", va="bottom", color=col, fontweight="bold",
                fontsize=fontsize or mpl.rcParams["xtick.labelsize"])


def shade_regions(ax, spans: Iterable[tuple], alpha: float = 0.55, text_y: float = 0.97, fontsize=None):
    """spans = [(x0, x1, 'Regime name', color_or_None), ...]. Names regimes directly on the plot
    (cf. 'Unachievable / Collaborative / Elevator-only' in an O-award B paper)."""
    fills = [C.light, C.light_accent, "#EEF3E6", "#F2ECF7"]
    for i, sp in enumerate(spans):
        x0, x1, name = sp[:3]
        col = sp[3] if len(sp) > 3 and sp[3] else fills[i % len(fills)]
        ax.axvspan(x0, x1, color=col, alpha=alpha, lw=0, zorder=0)
        if name:
            ax.text((x0 + x1) / 2, text_y, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=fontsize or mpl.rcParams["xtick.labelsize"] - 1, color=C.dark, style="italic")


def ref_line(ax, value: float, axis: str = "y", label: str | None = None, color: str | None = None, ls: str = "--",
             label_pos: float = 1.0):
    """Threshold / target / cutoff line with an inline label (no legend entry needed)."""
    col = color or C.accent2
    if axis == "y":
        ax.axhline(value, color=col, ls=ls, lw=0.9, zorder=1)
        if label:
            ax.annotate(label, (label_pos, value), xycoords=("axes fraction", "data"), xytext=(-2, 2),
                        textcoords="offset points", ha="right" if label_pos > 0.5 else "left", va="bottom", color=col,
                        fontsize=mpl.rcParams["xtick.labelsize"])
    else:
        ax.axvline(value, color=col, ls=ls, lw=0.9, zorder=1)
        if label:
            ax.annotate(label, (value, 1.0), xycoords=("data", "axes fraction"), xytext=(2, -2),
                        textcoords="offset points", ha="left", va="top", color=col,
                        fontsize=mpl.rcParams["xtick.labelsize"])


def band(ax, x, lo, hi, color: str | None = None, alpha: float = 0.25, label: str | None = None):
    """Uncertainty / prediction-interval band in the pale tone of the series colour."""
    return ax.fill_between(x, lo, hi, color=color or C.primary, alpha=alpha, lw=0, label=label)


def inset_zoom(ax, bounds=(0.55, 0.5, 0.4, 0.42), xlim=None, ylim=None):
    """Inset axes that magnifies a region; draws connector lines. Re-plot the data into it."""
    axins = ax.inset_axes(bounds)
    if xlim:
        axins.set_xlim(*xlim)
    if ylim:
        axins.set_ylim(*ylim)
    axins.tick_params(labelsize=mpl.rcParams["xtick.labelsize"] - 1)
    axins.grid(False)
    ax.indicate_inset_zoom(axins, edgecolor=C.dark, alpha=0.6)
    return axins


def panel_label(ax, text: str = "(a)", loc: str = "top-left", fontsize=None):
    """Panel tag. loc='top-left' (default, safest): bold '(a)' just outside the top-left corner.
    loc='below': centred under the x-axis label (mimics LaTeX sub-captions when several panels must
    live in ONE image file). Anchored to the x-label artist, so it never collides with it."""
    fs = fontsize or mpl.rcParams["font.size"]
    if loc == "below":
        ax.annotate(text, xy=(0.5, 0), xycoords=ax.xaxis.label if ax.get_xlabel() else ax,
                    xytext=(0, -3), textcoords="offset points", ha="center", va="top",
                    fontsize=fs, fontweight="bold", annotation_clip=False)
    else:
        ax.text(-0.02, 1.03, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=fs, fontweight="bold")


def emphasize(colors_n: int, hero: Sequence[int] | int, base: str | None = None, hero_color: str | None = None) -> list[str]:
    """Colour list for bars: everything neutral except the hero index(es)."""
    hero = [hero] if isinstance(hero, int) else list(hero)
    return [(hero_color or C.primary) if i in hero else (base or C.tertiary) for i in range(colors_n)]


def tornado(ax, names: Sequence[str], low: Sequence[float], high: Sequence[float], base: float,
            low_label: str = "-20 %", high_label: str = "+20 %"):
    """One-at-a-time sensitivity tornado sorted by swing. low/high = output at low/high input."""
    low, high = np.asarray(low, float), np.asarray(high, float)
    order = np.argsort(np.abs(high - low))
    y = np.arange(len(names))
    ax.barh(y, low[order] - base, left=base, color=C.accent, height=0.6, label=low_label)
    ax.barh(y, high[order] - base, left=base, color=C.primary, height=0.6, label=high_label)
    ax.axvline(base, color=C.dark, lw=0.8)
    ax.set_yticks(y, [names[i] for i in order])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", ncols=2, bbox_to_anchor=(0, 1.0), borderaxespad=0.2)


# --------------------------------------------------------------------------------------
# Accessibility check: colour-vision-deficiency simulation (Machado et al. 2009, severity 1)
# and grayscale lightness separation. Run `python mcm_style.py` to print a report.
# --------------------------------------------------------------------------------------
_CVD = {
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
    "protanopia": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
    "tritanopia": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]),
}


def _lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lab(rgb):
    r = _lin(rgb)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = M @ r / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _simulate(hexcol, kind):
    lin = _lin(to_rgb(hexcol))
    sim = np.clip(_CVD[kind] @ lin, 0, 1)
    return np.where(sim <= 0.0031308, 12.92 * sim, 1.055 * sim ** (1 / 2.4) - 0.055)


def check_palette(colors: Sequence[str]) -> dict:
    """Minimum pairwise CIELAB distance (normal + CVD) and min grayscale lightness gap."""
    out = {}
    for kind in ["normal", *_CVD]:
        labs = [_lab(to_rgb(c)) if kind == "normal" else _lab(_simulate(c, kind)) for c in colors]
        d = min(np.linalg.norm(labs[i] - labs[j]) for i in range(len(labs)) for j in range(i + 1, len(labs)))
        out[kind] = round(float(d), 1)
    L = sorted(_lab(to_rgb(c))[0] for c in colors)
    out["gray_L_min_gap"] = round(float(min(np.diff(L))), 1)
    return out


if __name__ == "__main__":
    for name, p in PALETTES.items():
        cyc = p.cycle()
        print(f"{name:6s} first-3 {check_palette(cyc[:3])} | first-4 {check_palette(cyc[:4])}")

