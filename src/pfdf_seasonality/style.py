"""Shared figure style: colorblind-safe palette, season colors, map framing.

The Okabe-Ito palette and the matplotlib rcParams block were identical in all
eight plotting scripts; the season colors and the western-US map extent were
repeated alongside them. Import ``apply()`` at the top of a plotting script
instead of pasting the block.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from cycler import cycler

__all__ = ["OKABE_ITO", "SEASON_COLOR", "C_INTENSE", "C_WET", "C_SAME", "C_DIFF",
           "WEST_EXTENT", "MONTH_INITIALS", "apply", "frame_map"]

# Okabe & Ito (2008), the standard colorblind-safe categorical set.
OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']

SEASON_COLOR = {"DJF": "#56B4E9", "MAM": "#009E73",
                "JJA": "#E69F00", "SON": "#8B4513"}

C_INTENSE, C_WET = "#009E73", "#CC79A7"     # intensity model vs amount model
C_SAME, C_DIFF = "#DDD8CE", "#D55E00"       # agreement maps

WEST_EXTENT = (-125.5, -101.5, 30.5, 49.6)  # lon0, lon1, lat0, lat1
MONTH_INITIALS = list("JFMAMJJASOND")


def apply(dpi: int = 130, savefig_dpi: int = 200, font_size: int = 9) -> None:
    """Set the project's matplotlib defaults. Call once, at import time."""
    plt.rcParams['axes.prop_cycle'] = cycler(color=OKABE_ITO)
    plt.rcParams['image.cmap'] = 'viridis'
    plt.rcParams.update({
        'figure.dpi': dpi,
        'savefig.dpi': savefig_dpi,
        'font.size': font_size,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def frame_map(ax, states, title, extent=WEST_EXTENT, aspect=1 / 0.75):
    """Draw state boundaries, set the western-US extent, and title the panel."""
    states.boundary.plot(ax=ax, linewidth=0.4, color="0.25", zorder=5)
    ax.set_xlim(*extent[:2])
    ax.set_ylim(*extent[2:])
    ax.set_aspect(aspect)
    ax.set_title(title, fontsize=9, loc="left", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
    return ax
