"""
figure_7_leaf.py — Leaf-only variant of Figure 7.

Same 6-panel layout as preprint_figures/figure_7.py, but Panels D
(Pareto) and F (cross-svp correlation heatmap) are recomputed on
Leaf-only trained conditions instead of the full Leaf + Root set.

Panels A/B/C/E already filter to Leaf in the preprint version, so
this variant only overrides the two panels whose upstream loaders
aggregate across all trained conditions:

  Panel D: `load_loss_summary_all()` parses `run_output.txt` for the
           training log's final loss values, which are means across all
           24 trained conditions per species. We re-derive the Leaf-only
           means by re-reading the last checkpoint's `Losses_step_*.tsv`
           and averaging only over the Leaf row indices.

  Panel F: `_build_corr_matrix()` correlates flux vectors from
           `curated_flux_both_species.tsv` which contains both Leaf and
           Root conditions. We filter to Leaf before building the pivot.

Output: preprint_figures/figure_7_leaf.{png,svg}
"""
from __future__ import annotations

import os
import sys
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "figures_src"))

from style import apply_style, SPECIES_COLORS, SVP_ORDER, get_dpi
from io_utils import (load_freeze_events, load_tsv,
                       fresh_checkpoints_dir, fresh_training_npz,
                       leaf_condition_indices)
import fig_ml_rslt_base as F7


# ---------- Leaf-only Panel D loader ----------

def load_loss_summary_leaf() -> pd.DataFrame:
    """Return (species, svp, Loss_C, Loss_SV) averaged over Leaf conditions only.

    Reads the last checkpoint of `Losses_step_*.tsv` per (species, svp) —
    each row is one condition (Total_Loss, Data_Loss, Mass_Loss) — and
    averages only the rows whose npz treatment string starts with
    `Leaf_Control_` or `Leaf_FeLim_`.

    Returns the columns **raw**, matching `io_utils.load_loss_summary_all()`.
    `Mass_Loss` therefore still carries the mass-penalty multiplier applied at
    `Library/Build_Model.py:774`; `F7._pareto_undo_penalty()` divides it back
    out, and `F7._panel4_pareto()` calls that itself. Do not pre-divide here or
    the panel will divide by p twice.
    """
    rows = []
    for sp in ("Poplar", "Sorghum"):
        leaf_idx = leaf_condition_indices(sp)
        if not leaf_idx:
            continue
        for svp in ("0.1", "0.5", "1.0", "2.0"):
            ck_dir = fresh_checkpoints_dir(sp, svp)
            if ck_dir is None:
                continue
            files = glob.glob(os.path.join(ck_dir, "Losses_step_*.tsv"))
            if not files:
                continue
            last = max(files,
                       key=lambda p: int(re.search(r"Losses_step_(\d+)", p).group(1)))
            df = pd.read_csv(last, sep="\t")
            leaf = df.iloc[leaf_idx]
            rows.append({
                "species": sp,
                "svp":     svp,
                "Loss_C":  float(leaf["Data_Loss"].mean()),
                "Loss_SV": float(leaf["Mass_Loss"].mean()),
            })
    return pd.DataFrame(rows)


# ---------- Leaf-only Panel F builder ----------

def _panel6_triangle_heatmap_leaf(ax, fig):
    """Fork of F7._panel6_triangle_heatmap that filters curated_flux to Leaf
    conditions before computing the cross-svp Pearson correlations."""
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable

    # CLAUDE 2026-08-14: whole-network correlation, not the 29 curated
    # reactions in curated_flux_both_species.tsv. See F7._corr_matrix_network.
    corr_pop = F7._corr_matrix_network("Poplar")
    corr_sor = F7._corr_matrix_network("Sorghum")

    n = len(SVP_ORDER)
    mat_pop  = np.full((n, n), np.nan)
    mat_sor  = np.full((n, n), np.nan)
    mat_diag = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                mat_diag[i, j] = 1.0
            elif i < j:
                mat_pop[i, j] = corr_pop[i, j]
            else:
                mat_sor[i, j] = corr_sor[i, j]

    norm = Normalize(vmin=F7.CONFIG.corr_vmin, vmax=F7.CONFIG.corr_vmax)

    def _capped(cmap_name, top=0.70):
        base = plt.get_cmap(cmap_name)
        return LinearSegmentedColormap.from_list(
            f"{cmap_name}_capped", base(np.linspace(0.0, top, 256)))
    cmap_pop  = _capped("Blues")
    cmap_sor  = _capped("Oranges")
    cmap_diag = _capped("Greys")
    cmap_cbar = cmap_diag

    ax.imshow(mat_pop,  cmap=cmap_pop,  norm=norm, interpolation="nearest")
    ax.imshow(mat_sor,  cmap=cmap_sor,  norm=norm, interpolation="nearest")
    ax.imshow(mat_diag, cmap=cmap_diag, norm=norm, interpolation="nearest")

    ax.set_xticks(range(n)); ax.set_xticklabels(SVP_ORDER)
    ax.set_yticks(range(n)); ax.set_yticklabels(SVP_ORDER)
    ax.set_xlabel("Mass penalty (p)")
    ax.set_ylabel("Mass penalty (p)")

    for (i, j) in np.ndindex(n, n):
        if i == j:
            val = mat_diag[i, j]
        elif i < j:
            val = mat_pop[i, j]
        else:
            val = mat_sor[i, j]
        if np.isnan(val):
            continue
        # Text colour switches at the midpoint of the colour scale, not at a
        # fixed correlation, so it tracks CONFIG.corr_vmin if that is retuned.
        _mid = (F7.CONFIG.corr_vmin + F7.CONFIG.corr_vmax) / 2.0
        ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                fontsize=8,
                color="white" if val > _mid else "black")

    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5],
            color="black", lw=1.2, alpha=0.7, linestyle=":")

    sm = ScalarMappable(norm=norm, cmap=cmap_cbar)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Coefficient (Leaf only)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


# ---------- Top-level builder ----------

def build_figure():
    apply_style()

    print("Loading panel 1 trajectories (already Leaf-filtered)...")
    trajectories = F7._load_panel1_trajectories()

    print("Loading panel 2 freeze events (already Leaf-filtered)...")
    freeze = load_freeze_events()
    freeze = freeze[freeze["tissue"] == "Leaf"]

    print("Loading panel 3 imbalance (already Leaf-filtered)...")
    df_imb = F7._load_panel3_imbalance()

    print("Loading panel 4 final-loss summary — LEAF ONLY override...")
    loss_df = load_loss_summary_leaf()

    print("Building 2x3 figure...")
    fig, axes = plt.subplots(*F7.CONFIG.layout, figsize=F7.CONFIG.figsize)
    axes = np.atleast_2d(axes)

    F7._panel1_trajectory(axes[0, 0], trajectories)
    F7._panel2_freeze(axes[0, 1], freeze)
    F7._panel3_imbalance(axes[0, 2], df_imb)
    F7._panel4_pareto(axes[1, 0], loss_df)          # Leaf-only Pareto
    F7._panel5_cdf(axes[1, 1])                       # already Leaf-only
    _panel6_triangle_heatmap_leaf(axes[1, 2], fig)   # Leaf-only heatmap

    flat_axes = [axes[0, 0], axes[0, 1], axes[0, 2],
                 axes[1, 0], axes[1, 1], axes[1, 2]]
    for ax, letter in zip(flat_axes, F7.CONFIG.panel_letters):
        kwargs = dict(F7.CONFIG.panel_letter_kwargs)
        # Panel D needs no override: with the penalty multiplier divided out
        # the curve slopes down to the right (p = 0.1 top-left, p = 2.0
        # bottom-right), which leaves the default bottom-left corner clear.
        if letter == "E":
            kwargs["xy"] = (0.10, 0.03)
        if letter == "F":
            kwargs["bbox"] = None
        ax.annotate(letter, xytext=(0, 0), textcoords="offset points",
                    **kwargs)

    # No figure-level title: the manuscript caption names the panels.
    fig.tight_layout()
    return fig


def main(out_dir: str = None) -> None:
    if out_dir is None:
        out_dir = _HERE
    os.makedirs(out_dir, exist_ok=True)
    fig = build_figure()
    dpi = get_dpi()
    png = os.path.join(out_dir, "fig_ml_rslt.png")
    svg = os.path.join(out_dir, "figure_7_leaf.svg")
    fig.savefig(png, dpi=dpi)
    fig.savefig(svg)
    plt.close(fig)
    print(f"[OK] {png} ({os.path.getsize(png)//1024} KB)")


if __name__ == "__main__":
    main()
