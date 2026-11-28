"""Small helpers for schematic figures drawn in inch coordinates (matplotlib only, vector output)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from style import ms


def canvas(width_in, height_in):
    fig = plt.figure(figsize=(width_in, height_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_in)
    ax.set_ylim(0, height_in)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, fc="white", ec="#24435C", lw=0.8, r=0.06, z=2, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z, ls=ls)
    ax.add_patch(p)
    return p


def arrow(ax, p0, p1, color="#24435C", lw=1.0, style="-|>", ms_=8, rad=0.0, z=3, ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms_, color=color, lw=lw, zorder=z,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, linestyle=ls)
    ax.add_patch(a)
    return a


def text(ax, x, y, s, size=7.5, color="#222222", ha="center", va="center", weight=None, style=None, z=4, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, weight=weight, style=style, zorder=z, **kw)


def badge(ax, x, y, s, fc="#24435C", size=7, r=0.085):
    c = plt.Circle((x, y), r, fc=fc, ec="white", lw=0.6, zorder=5)
    ax.add_patch(c)
    text(ax, x, y - 0.004, s, size=size, color="white", weight="bold", z=6)
