"""
style.py — shared matplotlib style and palette for all preprint figures.

USAGE
=====

At the top of each figure script::

    from style import apply_style, SPECIES_COLORS, SVP_COLORS, ...
    apply_style()

GLOBAL TWEAKS YOU MIGHT WANT
============================

- Change a species color across every figure at once: edit SPECIES_COLORS below.
- Make all text larger: change FONT_SIZE.
- Switch to a different colormap for svp gradients: change SVP_CMAP_NAME and
  re-import the script. The constants regenerate at module load.
- Override anything for a single figure: in that figure's CONFIG block, set
  e.g. ``CONFIG.palette = {"Poplar": "#000000", "Sorghum": "#888888"}`` and
  pass CONFIG.palette into the plotting calls there.
"""

from __future__ import annotations

import os
import matplotlib as mpl
import matplotlib.pyplot as plt


# ==== TUNABLE GLOBALS (edit here to affect ALL figures) ===========

#: Font size in points used everywhere unless overridden in a figure's CONFIG.
FONT_SIZE = 9

#: Default DPI for ``plt.savefig`` previews. Override per-run with the
#: ``FIG_DPI`` environment variable, e.g. ``FIG_DPI=300 python make_all.py``
#: for publication-quality output.
PREVIEW_DPI = 150

#: Matplotlib colormap NAME used to sample the four svp values to colors.
#: Try "viridis", "plasma", "cividis", or "magma" for alternative looks.
SVP_CMAP_NAME = "viridis"

#: Display order for svp values (used by SVP_COLORS and figure axes).
SVP_ORDER = ["0.1", "0.5", "1.0", "2.0"]

#: Display order for timepoints across the timecourse (1h appears in Root only).
TP_ORDER = ["1h", "2d", "4d", "7d", "14d", "21d"]

#: Per-species line / bar color. Two-tone blue/orange is colorblind-safe.
#: Change either value to a different hex code to recolor every figure.
SPECIES_COLORS = {
    "Poplar":  "#2c7fb8",   # blue
    "Sorghum": "#d95f0e",   # orange
}

#: Treatment linestyle. Solid for Control, dashed for FeLim.
TREATMENT_LS = {
    "Control": "-",
    "FeLim":   "--",
}

#: Per-tier colors used by Figures 5 and 11. ROBUST in dark green, the
#: PEAK_ROBUST / PLAUSIBLE intermediate tiers in lighter shades, DROP modes
#: in warm/gray tones.
TIER_COLORS = {
    "ROBUST":                       "#1b7837",  # dark green
    "PLAUSIBLE_svp_dependent":      "#5aae61",  # mid green
    "PEAK_ROBUST":                  "#a6dba0",  # light green
    "DROP_sign_flip":               "#d6604d",  # red
    "DROP_imbalanced":              "#f4a582",  # peach
    "DROP_imbalanced_neighborhood": "#f4a582",  # alias
    "DROP_low_flux":                "#bdbdbd",  # gray
    "DROP_other":                   "#969696",  # darker gray
    "no_meaningful_flux":           "#bdbdbd",
    "low_flux_uninterpretable":     "#bdbdbd",
}

# ==================================================================
# (Below this line: code that derives convenience constants from the
# tunables above. You normally don't edit this part — change the
# constants above instead.)
# ==================================================================


def _make_svp_colors(svp_order: list[str], cmap_name: str) -> dict[str, tuple]:
    """Sample one RGBA color per svp value from the named colormap."""
    cmap = plt.get_cmap(cmap_name)
    n = len(svp_order)
    # Sample evenly across the [0.15, 0.85] interior of the colormap to avoid
    # the extreme-light and extreme-dark ends, which are harder to read.
    return {s: cmap(0.15 + 0.70 * i / max(n - 1, 1)) for i, s in enumerate(svp_order)}


#: Per-svp color, sampled once at module load from SVP_CMAP_NAME.
SVP_COLORS = _make_svp_colors(SVP_ORDER, SVP_CMAP_NAME)


def apply_style(context: str = "preview") -> None:
    """Install global matplotlib rcParams for consistent styling.

    Parameters
    ----------
    context : {"preview", "publication"}
        ``"preview"`` uses ``PREVIEW_DPI`` (default 150) for fast iteration.
        ``"publication"`` forces 300 dpi regardless of the ``FIG_DPI`` env var.
        Either can be overridden by setting ``FIG_DPI`` in the environment.
    """
    if context == "publication":
        dpi = 300
    else:
        dpi = int(os.environ.get("FIG_DPI", PREVIEW_DPI))
    mpl.rcParams.update({
        "font.family":             "DejaVu Sans",
        # Force math text ($...$) into the same sans-serif font instead
        # of matplotlib's default serif STIX. Keeps all rendered text
        # visually consistent across plain labels and math labels.
        "mathtext.fontset":        "dejavusans",
        "mathtext.default":        "regular",
        "font.size":               FONT_SIZE,
        "axes.titlesize":          FONT_SIZE + 1,
        "axes.labelsize":          FONT_SIZE,
        "axes.spines.top":         False,
        "axes.spines.right":       False,
        "xtick.labelsize":         FONT_SIZE - 1,
        "ytick.labelsize":         FONT_SIZE - 1,
        "legend.fontsize":         FONT_SIZE - 1,
        "legend.frameon":          False,
        "figure.dpi":              dpi,
        "savefig.dpi":             dpi,
        "savefig.bbox":            "tight",
        "savefig.pad_inches":      0.1,
        "lines.linewidth":         1.4,
        "lines.markersize":        5,
    })


def get_dpi() -> int:
    """Return the active DPI, honoring the ``FIG_DPI`` env var."""
    return int(os.environ.get("FIG_DPI", PREVIEW_DPI))
