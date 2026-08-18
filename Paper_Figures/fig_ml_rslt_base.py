"""
Figure 7: Per-term convergence, early-stop dynamics, residual structure,
          and svp operating-point diagnostics
==========================================================================

Six panels in a 2x3 grid combining the prior figure_1.py (4 panels) and
figure_2.py (2 panels) into a single technical-overview figure for the
preprint Results section.

Layout (per the chosen grouping):

    +--------------------+--------------------+--------------------+
    | Per-term loss      | Freeze step        | Imbalance violins  |  <- training /
    | trajectory         | distribution / svp |                    |     optimization
    +--------------------+--------------------+--------------------+
    | Pareto:            | CDF of |SV| /      | Triangle-split     |  <- svp operating-
    | L_data vs L_mass   | throughput per rxn | correlation heat.  |     point validation
    +--------------------+--------------------+--------------------+

The bottom-right correlation heatmap uses a sequential Blues palette
(rather than the original RdYlBu_r) to emphasize that all cell values
sit in the high-correlation regime (~0.99 - 1.00).

Data sources
------------
- Top-left   : Biochem_*/projects/<spc>/ml/svp_1.0/checkpoints/Losses_step_*.tsv
- Top-mid    : fresh per-svp run_output.txt logs (parsed via load_freeze_events)
- Top-right  : Biochem_*/projects/<spc>/ml/svp_1.0/results/*_V_headers.tsv
                + training.npz's S matrix
- Bot-left   : fresh per-svp run_output.txt logs (load_loss_summary_all)
- Bot-mid    : network_resid_both_species.tsv (adj_max_resid column, all reactions)
- Bot-right  : curated_flux_both_species.tsv (per-(species,svp,condition,base_rxn) V_net)

To tweak this figure
--------------------
- Title:                              CONFIG.title
- Panel titles:                       CONFIG.panel_titles
- Trajectory svp / exit_steps:        CONFIG.trajectory_runs
- Pareto operating point:             CONFIG.pareto_highlight
- Correlation heatmap colormap:       CONFIG.corr_cmap
- Cell-text color rule:               _panel_b_triangle_heatmap source
- Figure / panel sizing:              CONFIG.panel_size / CONFIG.figsize
- Output filename:                    CONFIG.filename
"""

from __future__ import annotations

import os
import re
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Helper modules are vendored under figures_src/; prepend it so the
# imports below resolve. _HERE is this directory; the helper modules are
# vendored under figures_src/ so this deck runs from a bare checkout.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "figures_src"))

from style import (apply_style, SPECIES_COLORS, SVP_COLORS, SVP_ORDER, get_dpi)
from io_utils import (
    load_freeze_events, load_loss_trajectory, load_loss_summary_all,
    fresh_checkpoints_dir, fresh_training_npz, leaf_condition_indices,
    fresh_arm_dir, load_tsv,
)


# ==== CONFIG (edit me) ============================================
class CONFIG:
    title         = ""    # no figure-level title; refer to panels A-F in caption
    panel_letters = ["A", "B", "C", "D", "E", "F"]
    panel_letter_kwargs = dict(
        xy=(0.02, 0.03), xycoords="axes fraction",
        ha="left", va="bottom",
        fontsize=14, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                  edgecolor="none", alpha=0.85),
    )

    # ----- top-left panel: line patterns per loss component -----
    # Color is per species (matches species_palette); pattern distinguishes
    # the three components within each species. Total stays solid so it
    # reads as the "sum"; Mass/Data get distinct dash patterns.
    component_linestyles = {
        "Total_median": "-",
        "Mass_median":  "--",
        "Data_median":  ":",
    }
    # Order in which the per-species lines stack on the plot (drawing order)
    component_order = ["Total_median", "Mass_median", "Data_median"]
    component_labels = {
        "Total_median": "Total",
        "Mass_median":  "Mass",
        "Data_median":  "Data",
    }

    # ----- adopted operating point (single knob for the whole figure) -----
    # CLAUDE 2026-08-10: p = 2.0 is the default operating point everywhere.
    # Panels A and C must pick one p and take it from here.  Panels B, D, E
    # and F sweep p by construction; there `default_svp` is the highlighted
    # (D) / solid-anchored (E) value rather than a filter.
    default_svp = "2.0"

    # ----- panel 1: trajectory sources -----
    # exit_step = None -> resolve the highest Losses_step_*.tsv present in the
    # run's own checkpoints/ dir, so the trajectory always runs to that run's
    # early-stop point instead of a hardcoded step from a superseded run.
    trajectory_runs   = [
        ("Poplar",  default_svp, None),
        ("Sorghum", default_svp, None),
    ]
    trajectory_stride = 20

    # ----- panel 4 (pareto): operating point -----
    pareto_highlight  = default_svp

    # ----- panel 4 (pareto): manual per-point label offsets -----
    # Hand override of the automatic label placement, keyed by (species, p).
    # Value is (dx, dy) in POINTS from that point's marker: +dx right, +dy up.
    # A key present here wins outright — the auto-placement pass is skipped
    # for that label and no leader line is drawn.  Anything left out (or the
    # whole dict left empty) still gets automatic placement, so you can pin
    # only the one or two labels that are misbehaving.
    # Uncomment and tweak; ~10 pt is one marker radius.
    pareto_label_offsets = {
        # ("Poplar",  "0.1"): (-10,  10),
        # ("Poplar",  "0.5"): (-10, -14),
        # ("Poplar",  "1.0"): (-28,  10),
        # ("Poplar",  "2.0"): ( 10,  10),
        # ("Sorghum", "0.1"): ( 10, -10),
        # ("Sorghum", "0.5"): ( 10,  10),
        # ("Sorghum", "1.0"): ( 12, -10),
        # ("Sorghum", "2.0"): ( 10, -22),
    }

    # ----- panel 5 (CDF): threshold reference lines -----
    cdf_thresholds = {
        0.05: ("#2c7fb8", "5%"),
        0.15: ("#d95f0e", "15%"),
        0.30: ("#a63603", "30%"),
    }
    # Panel E: line-pattern per mass-penalty (p) value, with `default_svp`
    # (p = 2.0) as the solid anchor -- the adopted operating point.
    cdf_svp_linestyles = {
        "0.1": ":",
        "0.5": "--",
        "1.0": "-.",
        "2.0": "-",
    }

    # ----- panel 6 (triangle heatmap) -----
    corr_cmap      = "Blues"     # sequential blue; high correlation = dark
    # CLAUDE 2026-08-10: vmin lowered 0.985 -> 0.975 so the shading spans the
    # full range actually present in the rebuilt run (Sorghum p=0.1 vs p=2.0
    # falls to 0.9735).  Any cell below vmin clips to pure white.
    # CLAUDE 2026-08-14: raised 0.975 -> 0.9995. Fixing the panel F data source
    # (analyze_both_species.py had been reading the superseded qpsi-260406
    # sweep) removed the Sorghum 0.974-0.977 cells; the whole matrix now spans
    # 0.9996-1.0, which saturated the old scale into a uniform block.
    # Sam's choice, kept deliberately. The whole-network range runs down to
    # 0.99932 (Sorghum 0.1 vs 2.0), so that one cell sits below vmin and
    # renders at the palest end of the scale; the printed value still reads
    # 0.9993. Do not "fix" this by lowering the floor.
    corr_vmin      = 0.9995
    corr_vmax      = 1.0
    tri_label_offsets = {
        "Poplar":  (0.78, 0.92),
        "Sorghum": (0.22, 0.08),
    }

    # ----- colors / styles -----
    species_palette = dict(SPECIES_COLORS)
    svp_palette     = dict(SVP_COLORS)
    species_styles  = {"Poplar": "-", "Sorghum": "--"}
    loss_colors = {
        "Total_median": "#222222",
        "Mass_median":  "#d95f0e",
        "Data_median":  "#2c7fb8",
    }

    # ----- layout -----
    panel_size = 5.0
    figsize    = (panel_size * 3, panel_size * 2 + 1.0)   # +1 for suptitle pad
    layout     = (2, 3)

    # ----- output -----
    filename   = "figure_7"
# ==================================================================


# -------- Data loaders (ported from figure_1.py + figure_2.py) --------

def _final_loss_step(ck_dir):
    """Highest ``Losses_step_N.tsv`` present in ``ck_dir`` (the run's own
    early-stop point).  Used when CONFIG.trajectory_runs leaves exit_step
    as None so the panel never inherits a stale hardcoded step."""
    steps = [int(m.group(1))
             for f in os.listdir(ck_dir)
             for m in [re.match(r"Losses_step_(\d+)\.tsv$", f)] if m]
    if not steps:
        raise RuntimeError(f"No Losses_step_*.tsv in {ck_dir}")
    return max(steps)


def _load_panel1_trajectories():
    """Return list of (species, trajectory_df, n_metabolites, n_targets)."""
    out = []
    for sp, svp_str, exit_step in CONFIG.trajectory_runs:
        ck_dir = fresh_checkpoints_dir(sp, svp_str)
        if ck_dir is None:
            raise FileNotFoundError(
                f"No fresh checkpoints for species={sp}, svp={svp_str}"
            )
        if exit_step is None:
            exit_step = _final_loss_step(ck_dir)
            print(f"  panel A: {sp} p={svp_str} -> exit step {exit_step}")
        leaf_idx = leaf_condition_indices(sp)
        traj = load_loss_trajectory(
            ck_dir, sample_every=CONFIG.trajectory_stride,
            max_step=exit_step, cond_indices=leaf_idx,
        )
        npz_path = fresh_training_npz(sp)
        data_npz = np.load(npz_path, allow_pickle=True)
        n_mets = int(data_npz["S"].shape[0])
        Y = data_npz["Y"]
        nz = (Y != 0).sum(axis=1)
        n_targets = int(round(float(nz[leaf_idx].mean()))) if len(leaf_idx) else int(round(float(nz.mean())))
        out.append((sp, traj, n_mets, n_targets))
    return out


def _load_panel3_imbalance():
    """Return long-form DataFrame (Species, Condition, Imbalance) for the
    11 Leaf conditions of each species at CONFIG.default_svp (p = 2.0)."""
    rows = []
    for sp, svp_str, _ in CONFIG.trajectory_runs:
        npz_path = fresh_training_npz(sp)
        data_npz = np.load(npz_path, allow_pickle=True)
        S = np.asarray(data_npz["S"])
        treatments = [n.decode("utf-8") if isinstance(n, bytes) else str(n)
                      for n in data_npz["treatments"]]
        leaf_idx = leaf_condition_indices(sp)
        ck_dir = fresh_checkpoints_dir(sp, svp_str)
        run_dir = os.path.dirname(ck_dir.rstrip(os.sep))
        results_dir = os.path.join(run_dir, "results")
        candidates = (
            glob.glob(os.path.join(results_dir, "startVbfandZero_noRelu_V_headers.tsv"))
            or glob.glob(os.path.join(results_dir, "*_V_headers.tsv"))
        )
        if not candidates:
            continue
        v_df = pd.read_csv(candidates[0], sep="\t")
        if v_df.columns[0] == "" or v_df.columns[0].startswith("Unnamed"):
            v_df = v_df.iloc[:, 1:]
        v_arr = v_df.values
        if v_arr.shape[1] != S.shape[1]:
            print(f"WARN: V shape {v_arr.shape} mismatch S {S.shape} for {sp}")
            continue
        imb = S @ v_arr.T
        for t_idx in leaf_idx:
            cond = treatments[t_idx]
            for v in imb[:, t_idx]:
                rows.append({"Species": sp, "Condition": cond,
                             "Imbalance": float(v)})
    return pd.DataFrame(rows)


def _cond_sort_key(name: str) -> tuple:
    """Order Control_* before FeLim_*, then by timepoint."""
    tp_order = ["0h", "2d", "4d", "7d", "14d", "21d"]
    is_felim = 1 if "FeLim" in name else 0
    tp = name.rsplit("_", 1)[-1]
    try:
        tp_idx = tp_order.index(tp)
    except ValueError:
        tp_idx = 99
    return (is_felim, tp_idx)


def _build_corr_matrix(df: pd.DataFrame, species: str) -> np.ndarray:
    """Pearson correlation 4x4 across svp values, restricted to one species."""
    sub = df[df["species"] == species]
    long = sub.set_index(
        ["species", "condition", "base_rxn", "svp"])["V_net"].unstack("svp")
    long = long.dropna()
    if long.empty:
        return np.full((len(SVP_ORDER), len(SVP_ORDER)), np.nan)
    long = long[SVP_ORDER]
    return long.corr(method="pearson").to_numpy()


def _corr_matrix_network(species: str, tissue: str = "Leaf") -> np.ndarray:
    """Pearson correlation 4x4 across svp over the WHOLE flux vector.

    CLAUDE 2026-08-14: the curated-subset version above correlates only the 29
    photo/carbon reactions in curated_flux_both_species.tsv, which understates
    what panel F claims (reproducibility of "the converged fluxes"). This reads
    every reaction column of each arm's converged V and correlates the flattened
    Leaf block. Conclusion is unchanged -- all off-diagonals stay above 0.999 --
    but the claim now covers the network the other panels describe.
    """
    idx = leaf_condition_indices(species) if tissue == "Leaf" else None
    vecs = []
    for svp_v in SVP_ORDER:
        d = fresh_arm_dir(species, svp_v)
        path = os.path.join(d, f"ml/svp_{svp_v}/results",
                            "startVbfandZero_noRelu_V_headers.tsv")
        if not os.path.exists(path):
            return np.full((len(SVP_ORDER), len(SVP_ORDER)), np.nan)
        df = pd.read_csv(path, sep="\t", index_col=0)
        block = df.iloc[idx] if idx else df
        vecs.append(block.to_numpy(float).ravel())
    n = min(v.size for v in vecs)
    return np.corrcoef(np.vstack([v[:n] for v in vecs]))


# -------- Panel builders ---------------------------------------------

def _panel1_trajectory(ax, trajectories):
    """Top-left: per-term loss magnitude evolution, log-log.
    Color = species, line pattern = loss component. Total is solid
    so it reads as the sum; Mass / Data get distinct dash patterns.
    A shaded band shows the 5th-95th percentile of Total across the
    11 Leaf conditions per species (same species color, light alpha)."""
    from matplotlib.lines import Line2D
    # n_mets / n_targets come with each trajectory but are deliberately NOT
    # used to rescale the curves.  Every column written to Losses_step_*.tsv is
    # already a mean-per-row: Loss_SV divides by the metabolite count
    # (Build_Model.py:654) and Loss_Vout_constraint by the target count (:619).
    # Dividing again here normalised each curve a second time, sinking the
    # plateau from ~4e-2 to ~1e-4 and — because Mass and Data were divided by
    # different constants — putting the two components on incomparable scales.
    for sp, traj, _n_mets, _n_targets in trajectories:
        color = CONFIG.species_palette[sp]
        # 5th-95th percentile band on Total, drawn below the lines so the
        # solid median sits clearly on top.
        if "Total_p5" in traj.columns and "Total_p95" in traj.columns:
            ax.fill_between(traj["step"],
                            traj["Total_p5"], traj["Total_p95"],
                            color=color, alpha=0.18, lw=0, zorder=1)
        for comp in CONFIG.component_order:
            ax.plot(traj["step"], traj[comp],
                    color=color, lw=1.4,
                    linestyle=CONFIG.component_linestyles[comp],
                    zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    # Cap the early transient so the plateau — the part that carries the
    # convergence claim — is not squeezed into the bottom of the panel.  The
    # upper tail of the percentile band runs past this and is clipped.
    ax.set_ylim(top=1e4)
    ax.set_xlabel("Iteration")
    # Mass keeps its penalty multiplier here, unlike panel D: this panel shows
    # the objective actually being minimised at a single p, where Total reads
    # as Data + Mass (to within a couple of percent — these are medians across
    # conditions, and the median of a sum is not the sum of the medians).
    # Panel D divides p out because it compares across p.
    ax.set_ylabel("Loss")
    # Two-block legend: species swatches + pattern legend for Mass/Data
    species_handles = [
        Line2D([0], [0], color=CONFIG.species_palette[sp],
               lw=2.5, linestyle="-", label=sp)
        for sp, *_ in trajectories
    ]
    pattern_handles = [
        Line2D([0], [0], color="#444444", lw=1.6,
               linestyle=CONFIG.component_linestyles[comp],
               label=CONFIG.component_labels[comp])
        for comp in ["Mass_median", "Data_median"]
    ]
    ax.legend(handles=species_handles + pattern_handles,
              loc="upper right", fontsize=8, ncol=1,
              handlelength=2.6)


def _panel2_freeze(ax, freeze):
    """Top-middle: per-condition freeze step distribution across svp, Leaf."""
    species_list = list(CONFIG.species_palette.keys())
    for i, svp_v in enumerate(SVP_ORDER):
        for j, sp_name in enumerate(species_list):
            sub = freeze[(freeze["species"] == sp_name) & (freeze["svp"] == svp_v)]
            if sub.empty:
                continue
            y = sub["freeze_step"].to_numpy()
            x = i + (j - 0.5) * 0.30 + \
                np.random.RandomState(i * 10 + j).uniform(-0.07, 0.07, size=len(y))
            ax.scatter(x, y,
                       color=CONFIG.species_palette[sp_name],
                       s=28, alpha=0.7, edgecolor="white",
                       linewidth=0.4, zorder=3)
            ax.hlines(np.median(y),
                      i + (j - 0.5) * 0.30 - 0.13,
                      i + (j - 0.5) * 0.30 + 0.13,
                      color=CONFIG.species_palette[sp_name],
                      lw=1.6, zorder=4)
    ax.set_xticks(range(len(SVP_ORDER)))
    ax.set_xticklabels(SVP_ORDER)
    ax.set_xlabel("Mass penalty (p)")
    ax.set_ylabel("Iterations until convergence")
    for sp_name in species_list:
        ax.scatter([], [], color=CONFIG.species_palette[sp_name],
                   s=30, label=sp_name)
    ax.legend(loc="upper right", fontsize=8)


def _panel3_imbalance(ax, df_imb):
    """Top-right: combined Poplar+Sorghum metabolite imbalance violins
    per Leaf condition, staggered side-by-side. The Control 0h timepoint
    is dropped since FeLim doesn't exist there (no paired comparison)."""
    if df_imb.empty:
        ax.text(0.5, 0.5, "No imbalance data", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_axis_off()
        return
    # Drop the 0h baseline (Control-only; no matching FeLim row).
    df_imb = df_imb[~df_imb["Condition"].str.endswith("_0h")]
    cond_order = sorted(df_imb["Condition"].unique(), key=_cond_sort_key)
    sns.violinplot(
        data=df_imb, x="Condition", y="Imbalance",
        hue="Species", order=cond_order,
        hue_order=list(CONFIG.species_palette.keys()),
        palette=CONFIG.species_palette,
        split=False, dodge=True,
        inner="quartile", cut=0, linewidth=0.6,
        density_norm="width", ax=ax,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    # +/-1 rather than +/-5: the widest per-condition imbalance SD is 0.20
    # (Poplar Control 14d) and the tightest 0.023 (Sorghum FeLim 14d), so at
    # +/-5 every violin was a hairline. The tails outside +/-1 are clipped.
    ax.set_ylim(-1, 1)
    # Short x-tick labels: keep just the timepoint (drop "Leaf_<treatment>_")
    short_labels = [c.rsplit("_", 1)[-1] for c in cond_order]
    ax.set_xticklabels(short_labels)
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
        tick.set_horizontalalignment("center")
    ax.set_xlabel("")
    ax.set_ylabel("Net flux balance per metabolite")
    ax.legend(loc="upper right", fontsize=8)

    # Below-x-axis treatment labels (Control / FeLim) sit under the
    # timepoint tick labels, with a single dotted vertical separator
    # between the two groups instead of horizontal brackets.
    from matplotlib.transforms import blended_transform_factory
    n_ctrl = sum(1 for c in cond_order if "_Control_" in c)
    n_felim = len(cond_order) - n_ctrl
    if n_ctrl > 0 and n_felim > 0:
        # Vertical dotted separator spanning the full panel height,
        # placed midway between the last Control and first FeLim position.
        ax.axvline(n_ctrl - 0.5, color="black",
                   linestyle=(0, (8, 4)),     # heavier, longer-dash pattern
                   linewidth=1.4, alpha=0.7, zorder=0)

        # Treatment labels below the x-axis: blended (x = data, y = axes).
        # y must be just below the tick labels but still visually attached
        # to panel C, not falling into the row gap above panel F.
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text((n_ctrl - 1) / 2.0, -0.09, "Control",
                ha="center", va="top", transform=trans,
                fontsize=10, clip_on=False)
        ax.text(n_ctrl + (n_felim - 1) / 2.0, -0.09, "FeLim",
                ha="center", va="top", transform=trans,
                fontsize=10, clip_on=False)


def _pareto_undo_penalty(loss_df):
    """Divide the mass penalty back out of `Loss_SV`, for the Pareto panel.

    Both axes stay the two loss terms themselves — each is a mean squared
    deviation per row (`Build_Model.py:654` divides by the metabolite count,
    `:619` by the reaction count).  The only correction is to p:

    `Mass_Loss` is written to `Losses_step_*.tsv` **after** the mass penalty has
    been applied (`Library/Build_Model.py:774`, `L2 = L2 * p_sv`), so the raw
    column is that term's contribution to the objective, not the imbalance it
    measures.  Dividing by p recovers the residual itself, which is what makes
    the four arms comparable.  Leaving the multiplier in inverts the ordering of
    the sweep — it shrinks the p = 0.1 value tenfold and doubles the p = 2.0
    one, a factor of 20 across the four arms — so that the loosest penalty looks
    like the best on both axes.  Do not "simplify" this away.

    Note the residual is the *square* of the resolution floor quoted in the
    Results: floor = sqrt(Loss_SV / p), so 0.02380 here is a floor of 0.1543.
    """
    out = loss_df.copy()
    out["Loss_C"]  = out["Loss_C"].astype(float)
    out["Loss_SV"] = out["Loss_SV"].astype(float) / out["svp"].astype(float)
    return out


def _panel4_pareto(ax, loss_df):
    """Bot-left: mass-balance vs data-fit Pareto curve across svp.

    Takes the raw loader columns and corrects them itself
    (`_pareto_undo_penalty`), so every caller gets the same units as the axis
    labels below.

    The adopted operating point (CONFIG.default_svp, p = 2.0) is marked by a
    black ring around each species marker — no in-plot text callout. In-plot
    p labels are drawn in a slightly darker shade of the species color."""
    loss_df = _pareto_undo_penalty(loss_df)
    # Darker tint of each species color for the in-plot p value labels.
    species_label_color = {"Poplar": "#08406b", "Sorghum": "#8c3604"}
    # Once the penalty multiplier is divided out the curve is monotone
    # *decreasing* — tightening the mass balance costs data fit — so the
    # perpendicular to the connecting line runs up-right / down-left.  One
    # species is pushed to each side so the two never collide.
    label_side = {"Poplar": (+1.0, +1.0), "Sorghum": (-1.0, -1.0)}
    species_list = list(CONFIG.species_palette.keys())
    pending = []
    for sp in species_list:
        sub = loss_df[loss_df["species"] == sp].sort_values("svp")
        if sub.empty:
            continue
        x = sub["Loss_C"].to_numpy()
        y = sub["Loss_SV"].to_numpy()
        c = CONFIG.species_palette[sp]
        ax.plot(x, y, color=c, lw=1.2, alpha=0.6, zorder=1)
        ax.scatter(x, y, s=80, color=c, edgecolor="white",
                   linewidth=1.0, label=sp, zorder=3)
        # Black ring around the chosen operating point (CONFIG.default_svp).
        op = sub[sub["svp"] == CONFIG.pareto_highlight]
        if not op.empty:
            ax.scatter([float(op["Loss_C"].iloc[0])],
                       [float(op["Loss_SV"].iloc[0])],
                       s=200, facecolors="none",
                       edgecolors="#222222", linewidths=1.5, zorder=2)
        pending.append((sp, x, y, list(sub["svp"]), c))
    ax.set_xscale("log"); ax.set_yscale("log")
    # Headroom so the corner marker + its label are not clipped, and so the
    # top-right panel letter has clear space.
    ax.margins(x=0.14, y=0.20)
    # On a narrow decade span matplotlib labels the minor ticks too, which
    # run into each other; keep the decade labels only.
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Data loss")
    ax.set_ylabel("Mass loss")
    ax.legend(loc="lower right", fontsize=8)

    # ---- inset: resolution floor at the end of the run, vs p ---------------
    # The floor is the RMS mass imbalance per metabolite, i.e. the square root
    # of this panel's y axis (Loss_SV already has p divided out above).  It is
    # the quantity the Results quote as the limit of what the model can
    # resolve, so the inset shows directly what tightening p buys.  Placed
    # bottom-left, the one corner the Pareto curve leaves empty.
    inset = ax.inset_axes([0.17, 0.13, 0.34, 0.31])
    for sp in species_list:
        sub = loss_df[loss_df["species"] == sp].sort_values("svp")
        if sub.empty:
            continue
        inset.plot(sub["svp"].astype(float), sub["Loss_SV"] ** 0.5,
                   color=CONFIG.species_palette[sp], lw=1.2,
                   marker="o", ms=3.2, mec="white", mew=0.6, zorder=3)
        op = sub[sub["svp"] == CONFIG.pareto_highlight]
        if not op.empty:
            inset.scatter([float(op["svp"].iloc[0])],
                          [float(op["Loss_SV"].iloc[0]) ** 0.5],
                          s=55, facecolors="none", edgecolors="#222222",
                          linewidths=1.0, zorder=4)
    inset.set_xscale("log")
    inset.set_xticks([0.1, 0.5, 1.0, 2.0])
    inset.set_xticklabels(["0.1", "0.5", "1", "2"])
    inset.xaxis.set_minor_locator(mticker.NullLocator())
    inset.tick_params(labelsize=6, length=2, pad=1.5)
    inset.set_xlabel("mass penalty (p)", fontsize=6.5, labelpad=1.5)
    inset.set_ylabel("resolvable flux limit", fontsize=6.5, labelpad=1.5)
    for spine in inset.spines.values():
        spine.set_linewidth(0.6)
    inset.patch.set_alpha(0.92)

    # ---- label placement pass, done in display space so that near-coincident
    # points (p values that converge to almost the same loss pair) get pushed
    # apart by an amount that reflects what the reader actually sees.
    # Must run after the scales/limits above are final.
    markers = np.vstack([np.column_stack([x, y]) for _, x, y, _, _ in pending])
    markers_px = ax.transData.transform(markers)
    placed_px = []
    for sp, x, y, svps, c in pending:
        sx, sy = label_side[sp]
        pts_px = ax.transData.transform(np.column_stack([x, y]))
        for (px, py), (xi, yi), svp_val in zip(pts_px, zip(x, y), svps):
            # Hand-pinned offset for this point, if CONFIG supplies one.
            manual = CONFIG.pareto_label_offsets.get((sp, svp_val))
            if manual is not None:
                dx_pt, dy_pt = manual
                ax.annotate(svp_val, xy=(xi, yi),
                            xytext=(dx_pt, dy_pt), textcoords="offset points",
                            ha="right" if dx_pt < 0 else "left",
                            va="bottom" if dy_pt > 0 else "top",
                            fontsize=8, fontweight="bold", zorder=4,
                            color=species_label_color.get(sp, c))
                # Record where it landed (points -> pixels) so any label still
                # under automatic placement steps around this one too.
                pt2px = ax.figure.dpi / 72.0
                placed_px.append((px + dx_pt * pt2px, py + dy_pt * pt2px))
                continue
            # Step outward along the perpendicular until the label clears every
            # label already placed and every marker other than its own.
            for step in range(8):
                dist = 10.0 + 10.0 * step
                lx_px, ly_px = px + sx * 0.8 * dist, py + sy * dist
                clear_lbl = all(np.hypot(lx_px - qx, ly_px - qy) > 15.0
                                for qx, qy in placed_px)
                clear_mrk = all(np.hypot(lx_px - qx, ly_px - qy) > 11.0
                                for qx, qy in markers_px
                                if np.hypot(px - qx, py - qy) > 1e-6)
                if clear_lbl and clear_mrk:
                    break
            placed_px.append((lx_px, ly_px))
            # A label that had to step out past its first position is no longer
            # unambiguously "next to" its marker, so give it a hairline leader.
            arrow = None
            if step > 0:
                arrow = dict(arrowstyle="-", lw=0.6, alpha=0.55, color=c,
                             shrinkA=1.0, shrinkB=4.0)
            ax.annotate(svp_val, xy=(xi, yi),
                        xytext=(sx * 0.8 * dist, sy * dist),
                        textcoords="offset points",
                        ha="right" if sx < 0 else "left",
                        va="bottom" if sy > 0 else "top",
                        fontsize=8, fontweight="bold", zorder=4,
                        color=species_label_color.get(sp, c),
                        arrowprops=arrow)


def _panel5_cdf(ax):
    """Bot-middle: CDF of worst adjacent metabolite |SV| / throughput, over
    EVERY base reaction in the network.
    Color = species (matches Panel A); line pattern = mass penalty (p)
    with CONFIG.default_svp (p = 2.0) as the solid anchor.

    CLAUDE 2026-08-14: was `iron_timecourse.tsv` — the 36 iron-binding
    reactions only — while sitting in a figure whose other five panels are
    network-wide, and that file was additionally never refreshed after the
    arms-260812 sweep (it is written by timecourse_and_relaxed.py, which is
    not in the documented cache-rebuild order). Now reads the network-wide
    table written by analyze_both_species.py. The metric is unchanged and is
    bounded in [0, 1] by the triangle inequality: it is NOT clipped."""
    from matplotlib.lines import Line2D
    iron_tc = load_tsv("network_resid_both_species.tsv")
    species_list = list(CONFIG.species_palette.keys())
    for sp in species_list:
        sub = iron_tc[(iron_tc["species"] == sp)
                      & (iron_tc["tissue"] == "Leaf")]
        color = CONFIG.species_palette[sp]
        for svp_v in SVP_ORDER:
            vals = sub[sub["svp"] == svp_v]["adj_max_resid"].dropna().to_numpy()
            if vals.size == 0:
                continue
            sorted_v = np.sort(vals)
            cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
            ax.plot(sorted_v, cdf,
                    color=color, lw=1.5,
                    linestyle=CONFIG.cdf_svp_linestyles.get(svp_v, "-"))
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Per-reaction metabolite imbalance (relative to flux)")
    ax.set_ylabel("Fraction of reactions (CDF)")
    # Two-block legend mirroring Panel A: species swatches + pattern entries
    species_handles = [
        Line2D([0], [0], color=CONFIG.species_palette[sp],
               lw=2.5, linestyle="-", label=sp)
        for sp in species_list
    ]
    pattern_handles = [
        Line2D([0], [0], color="#444444", lw=1.6,
               linestyle=CONFIG.cdf_svp_linestyles[svp_v],
               label=f"p = {svp_v}")
        for svp_v in SVP_ORDER
    ]
    ax.legend(handles=species_handles + pattern_handles,
              loc="lower right", fontsize=8, ncol=1,
              handlelength=2.6)


def _panel6_triangle_heatmap(ax, fig):
    """Bot-right: triangle-split correlation heatmap.
        Upper triangle = Poplar  (Blues   cmap)
        Lower triangle = Sorghum (Oranges cmap)
        Diagonal       = 1.000   (Greys   cmap, neutral self-corr)
    All three sub-maps share the same value-to-darkness mapping (same
    vmin/vmax), and the colorbar is drawn in grayscale so a given
    darkness reads off the same Pearson r regardless of which species
    triangle it sits in."""
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    df = load_tsv("curated_flux_both_species.tsv")
    corr_pop = _build_corr_matrix(df, "Poplar")
    corr_sor = _build_corr_matrix(df, "Sorghum")

    n = len(SVP_ORDER)
    mat_pop  = np.full((n, n), np.nan)   # Poplar's upper triangle
    mat_sor  = np.full((n, n), np.nan)   # Sorghum's lower triangle
    mat_diag = np.full((n, n), np.nan)   # diagonal (self-correlation)
    for i in range(n):
        for j in range(n):
            if i == j:
                mat_diag[i, j] = 1.0
            elif i < j:
                mat_pop[i, j] = corr_pop[i, j]
            else:
                mat_sor[i, j] = corr_sor[i, j]

    norm = Normalize(vmin=CONFIG.corr_vmin, vmax=CONFIG.corr_vmax)

    # Cap each cmap below pure black/maximally-saturated so the highest-
    # value cells (r = 1.0, including the diagonal) read as a strong but
    # not visually heavy dark shade. The cap (0.70) sets the darkness of
    # the top-of-scale cells; all three cmaps cap at the same level so
    # equal correlation values look equally dark across triangles.
    def _capped(cmap_name, top=0.70):
        base = plt.get_cmap(cmap_name)
        return LinearSegmentedColormap.from_list(
            f"{cmap_name}_capped", base(np.linspace(0.0, top, 256)),
        )
    cmap_pop  = _capped("Blues")
    cmap_sor  = _capped("Oranges")
    cmap_diag = _capped("Greys")
    cmap_cbar = cmap_diag   # grayscale colorbar matches the cell scaling

    # Three layered imshow calls, one per cmap. NaN cells render as
    # transparent so the layers don't fight each other.
    ax.imshow(mat_pop,  cmap=cmap_pop,  norm=norm, interpolation="nearest")
    ax.imshow(mat_sor,  cmap=cmap_sor,  norm=norm, interpolation="nearest")
    ax.imshow(mat_diag, cmap=cmap_diag, norm=norm, interpolation="nearest")

    ax.set_xticks(range(n)); ax.set_xticklabels(SVP_ORDER)
    ax.set_yticks(range(n)); ax.set_yticklabels(SVP_ORDER)
    ax.set_xlabel("Mass penalty (p)")
    ax.set_ylabel("Mass penalty (p)")

    # Cell text — with the capped cmaps the upper-half values still land
    # on dark cells, so flip to white text above ~0.995 and stay black
    # below. (The threshold is lower than before because the cmap no
    # longer drives all the way to black at value 1.0.)
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
        _mid = (CONFIG.corr_vmin + CONFIG.corr_vmax) / 2.0
        ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                fontsize=8,
                color="white" if val > _mid else "black")

    # Anti-diagonal separator between the two species triangles
    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5],
            color="black", lw=1.2, alpha=0.7, linestyle=":")

    # Grayscale colorbar — same capped value-to-darkness mapping that
    # the two tinted sub-maps use. Reader's takeaway: darker = higher
    # Pearson r, regardless of which triangle the cell sits in.
    sm = ScalarMappable(norm=norm, cmap=cmap_cbar)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Coefficient", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


# -------- Top-level builder ------------------------------------------

def build_figure():
    apply_style()

    print("Loading panel 1 trajectories...")
    trajectories = _load_panel1_trajectories()

    print("Loading panel 2 freeze events...")
    freeze = load_freeze_events()
    freeze = freeze[freeze["tissue"] == "Leaf"]

    print("Loading panel 3 imbalance from V_headers...")
    df_imb = _load_panel3_imbalance()

    print("Loading panel 4 final-loss summary...")
    loss_df = load_loss_summary_all()

    print("Building 2x3 figure...")
    fig, axes = plt.subplots(*CONFIG.layout, figsize=CONFIG.figsize)
    axes = np.atleast_2d(axes)

    _panel1_trajectory(axes[0, 0], trajectories)
    _panel2_freeze(axes[0, 1], freeze)
    _panel3_imbalance(axes[0, 2], df_imb)
    _panel4_pareto(axes[1, 0], loss_df)
    _panel5_cdf(axes[1, 1])
    _panel6_triangle_heatmap(axes[1, 2], fig)

    # Panel letters A-F in the bottom-left of each axes, in row-major order.
    # E gets nudged slightly to the right so it doesn't clip the CDF curves
    # rising from the origin.
    flat_axes = [axes[0, 0], axes[0, 1], axes[0, 2],
                 axes[1, 0], axes[1, 1], axes[1, 2]]
    for ax, letter in zip(flat_axes, CONFIG.panel_letters):
        kwargs = dict(CONFIG.panel_letter_kwargs)
        # Panel D needs no override: with the penalty multiplier divided out
        # the curve slopes down to the right (p = 0.1 top-left, p = 2.0
        # bottom-right), which leaves the default bottom-left corner clear.
        if letter == "E":
            kwargs["xy"] = (0.10, 0.03)
        if letter == "F":
            # Heatmap cells make a white bbox jarring against the tinted
            # background; drop the bbox so the letter sits on the cell.
            kwargs["bbox"] = None
        ax.annotate(letter, xytext=(0, 0), textcoords="offset points",
                    **kwargs)

    if CONFIG.title:
        fig.suptitle(CONFIG.title, y=1.00, fontsize=12)
    fig.tight_layout()
    return fig


def main(out_dir: str = None) -> None:
    # Write outputs alongside the script (preprint_figures/) by default.
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    fig = build_figure()
    dpi = get_dpi()
    png = os.path.join(out_dir, CONFIG.filename + ".png")
    svg = os.path.join(out_dir, CONFIG.filename + ".svg")
    fig.savefig(png, dpi=dpi)
    fig.savefig(svg)
    plt.close(fig)
    print(f"[OK] {png} ({os.path.getsize(png)//1024} KB)")


if __name__ == "__main__":
    main()
