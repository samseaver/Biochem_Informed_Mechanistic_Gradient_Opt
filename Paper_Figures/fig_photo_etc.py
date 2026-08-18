"""
Figure 8: Photosynthetic ETC response paired with GLK-regulon transcript
=========================================================================

2 x 2 layout, one column per species (Sorghum left, Poplar right):

    Top row (A, B) — per-ETC-reaction net flux DIFFERENCE (FeLim - Control)
                     across the leaf time course (2d/4d/7d/14d/21d).
                     5 lines per panel (PSII, Cyt b6f, PSI, FNR, ATP
                     synthase). ATPsyn / PSII ratio inset (Control vs
                     FeLim, dashed reference at canonical 3:1) sits in
                     an interior corner of each top panel.

    Bottom row (C, D) — log2(FeLim/Control) transcript abundance for the
                        two GLK-family orthologs (GLK1, GLK2) predicted
                        to be the master regulators of the nuclear-
                        encoded photosynthesis regulon that the top-row
                        driver reactions belong to. Same 5-timepoint
                        x-axis so the transcript trajectory can be
                        read against the flux trajectory directly.

Data sources
------------
- Top row : Biochem_*/projects/<spc>/ml/svp_1.0/results/startVbfandZero_noRelu_V_headers.tsv
- Bottom  : RNASeq-Review/RNASeq_Enzyme_Abundance/projects/qpsi-plastidial/
            rnaseq-data/{Poplar,Sorghum}_raw_genes_tmm_mean.tsv[.xz]
            filtered by the four GLK ortholog IDs listed in CONFIG.glk_orthologs.

To tweak this figure
--------------------
- Target reactions / legend order:   CONFIG.target_reactions / CONFIG.legend_order
- Which reactions form the ratio:    CONFIG.ratio_numerator_id / CONFIG.ratio_denominator_id
- Canonical ratio reference line:    CONFIG.ratio_reference
- GLK ortholog IDs / data path:      CONFIG.glk_orthologs / CONFIG.glk_data_paths
- GLK colors / linestyle:            CONFIG.glk_colors / CONFIG.glk_styles
- Bottom-row y-range:                CONFIG.panel_ylim_bottom
- Species panel order:               CONFIG.species_configs
- Output filename:                   CONFIG.filename
"""

from __future__ import annotations

import os
import sys
import lzma
import re
import csv
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cobra


def _analysis_tables() -> str:
    """Directory holding the cross-species analysis TSVs this deck reads.

    Delegates to io_utils.cross_species_dir() so every consumer resolves the
    tables the same way: analysis_tables/ beside this file by default,
    overridable with BIOFLUX_DATA_DIR / BIOFLUX_CROSS_DIR.

    The tables are derived from both species, and the Poplar half is not
    distributed with this repository (see README.md), so they are not tracked
    here. Regenerate them with svp_analysis/analyze_both_species.py from a tree
    that has both species, or point the env var at an existing copy.
    """
    import io_utils
    return io_utils.cross_species_dir()


def _base_rxn(target_id: str) -> str:
    """Strip compartment/direction suffix (_y0, _d0, _f, _r, _i, _o) to get the
    base reaction id used by the classifier."""
    return re.sub(r'_(y[0-9]+|d[0-9]+|[frio])$', '', target_id)


@functools.lru_cache(maxsize=1)
def _load_qualifying_tps() -> dict:
    """Return {(species, base_rxn): set_of_qualifying_timepoint_strs} loaded
    once from the central-carbon classifier output. Empty set = the reaction
    was classified but no timepoint met the defensibility criterion."""
    path = os.path.join(_analysis_tables(), "all_central_carbon_classification.tsv")
    result = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row.get('tissue') != 'Leaf':
                continue
            tps = row.get('qualifying_tps', '') or ''
            qtps = set(tps.split(',')) if tps else set()
            result[(row['species'], row['base_rxn'])] = qtps
    return result


@functools.lru_cache(maxsize=1)
def _load_per_tp_metrics() -> dict:
    """Return {(species, base_rxn, timepoint): (rel_mag, dir_flip)}.
    rel_mag = |mean(delta across p)| / max(mean_ctl, mean_fel, 1e-6).
    dir_flip = the four per-p deltas are not all the same sign.
    Computed from all_central_carbon_timecourse.tsv (Leaf only)."""
    import collections as _c
    path = os.path.join(_analysis_tables(), "all_central_carbon_timecourse.tsv")
    grouped = _c.defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row.get('tissue') != 'Leaf':
                continue
            k = (row['species'], row['base_rxn'], row['timepoint'])
            grouped[k].append((float(row['abs_ctl']), float(row['abs_fel']),
                               float(row['delta'])))
    result = {}
    for k, rows in grouped.items():
        if not rows:
            continue
        mc = sum(r[0] for r in rows) / len(rows)
        mf = sum(r[1] for r in rows) / len(rows)
        md = sum(r[2] for r in rows) / len(rows)
        deltas = [r[2] for r in rows]
        denom = max(mc, mf, 1e-6)
        rel_mag = abs(md) / denom
        dir_flip = (any(d > 1e-4 for d in deltas)
                    and any(d < -1e-4 for d in deltas))
        result[k] = (rel_mag, dir_flip)
    return result


def _alpha_for_rel_mag(rel_mag: float) -> float:
    """Map rel_mag to marker opacity. rel_mag >= 10% -> fully opaque;
    below 0.5% -> minimum opacity. Linear scale in between so the
    reader gets a visible confidence gradient across the interesting
    range (0.5% - 10%)."""
    if rel_mag >= 0.10:
        return 1.0
    if rel_mag <= 0.005:
        return 0.15
    return 0.15 + (1.0 - 0.15) * (rel_mag - 0.005) / (0.10 - 0.005)

# Helper modules are vendored under figures_src/; prepend it so the
# imports below resolve. _HERE is this directory; the helper modules are
# vendored under figures_src/ so this deck runs from a bare checkout.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "figures_src"))
from style import apply_style, SPECIES_COLORS, get_dpi
# CLAUDE 2026-08-12: run directories moved to the arms-260811 sweep, one per
# (species, svp, min_delta). io_utils is the single resolver; see its header
# for the BIOFLUX_MIN_DELTA / BIOFLUX_LEGACY_LAYOUT overrides.
from io_utils import fresh_arm_dir
import io_utils


# ==== CONFIG (edit me) ============================================
class CONFIG:
    title         = ""    # no figure-level title; caption names panels A/B
    panel_titles  = None  # no panel titles either; caption owns them
    # Panel letters row-major: A/B = ETC flux, C/D = GLK transcript,
    # E/F..M/N = one phenotype row per entry in phenotype_columns.
    # Extended to cover the max we might use in the exploratory pass;
    # unused letters are harmless.
    panel_letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N"]
    panel_letter_kwargs = dict(
        xy=(0.02, 0.97), xycoords="axes fraction",
        ha="left", va="top",
        fontsize=14, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                  edgecolor="none", alpha=0.85),
    )
    panel_labels = {
        "top_y":     r"Net flux difference ($v_{FeLim} - v_{Control}$)",
        "bottom_y":  r"Transcript difference (FeLim − Control) TMM",
        "x_top":     "Time (Days)",
        "x_bottom":  "Time (Days)",
    }
    # Every row uses raw (FeLim − Control) so each panel keeps the
    # natural units + amplitudes of its measurement. Consequences:
    # (a) the 5 ETC reactions in the top row visually separate again
    #     (their raw flux magnitudes differ even when the log2 fold
    #     changes are nearly identical), (b) high-abundance transcripts
    #     dominate the GLK row (paleopolyploid low-abundance paralogs
    #     will look flat), (c) each phenotype row needs its own ylim
    #     matched to that measurement's natural scale.
    # CLAUDE 2026-08-12: widened for the arms-260811 runs. Initialising the
    # media transport chain at the pFVA maxima raised the absolute flux scale
    # ~2.5x, so the net svp_2.0 delta-flux range is now (-352..+25) for Sorghum
    # and (-94..+16) for Poplar (was -211..+43 shared). Panels A and B share
    # this axis deliberately, so Sorghum sets the floor; the headroom above
    # zero is reserved for the group legend in B. (The ATPsyn/PSII inset used
    # to occupy the headroom in A; it now sits low in B beside the Fe / Chl
    # insets, so A's headroom is free if the range is ever retightened.)
    panel_ylim = (-375, 120)

    # ----- display sign per reaction -----
    # ModelSEED writes some reactions in the opposite direction to the
    # one they run in vivo, so their net flux is negative by convention
    # even though the pathway is carrying forward flux.  Negate those for
    # display so every member of a pathway group moves together; the
    # caption states which reactions are flipped.
    #   rxn00782 (NADPH-GAPDH) is written oxidatively (G3P -> 1,3-BPG);
    #   the Calvin cycle runs it reductively.
    display_direction = {"rxn00782": -1}
    # GLK row uses a TWIN y-axis so GLK regulators (low-abundance,
    # ±60-306 TMM) are visible alongside their high-abundance target
    # subunits (±8000+ TMM). BOTH axes are symmetric around zero so
    # the y=0 line physically aligns at 50% panel height on both
    # sides, making Sorghum 7d / Poplar 21d drops directly comparable.
    # Left / right axis tick pairs picked so the ticks line up visually:
    # primary +300 ↔ twin +8000, +150 ↔ +4000, 0 ↔ 0, etc. The ylim on
    # each side is set slightly wider than its tick range for padding;
    # both ylims use the SAME padding fraction (320/300) so the tick
    # alignment is preserved.
    panel_ylim_bottom              = (-320, 320)                       # left: GLK  ΔTMM
    panel_yticks_bottom            = [-300, -150, 0, 150, 300]
    panel_ylim_bottom_subunit      = (-320 * 8000 / 300,
                                       320 * 8000 / 300)               # right: subunit ΔTMM
    panel_yticks_bottom_subunit    = [-8000, -4000, 0, 4000, 8000]
    # ----- inset placement per species (axes-fraction: x0, y0, width, height) -----
    # Squished vertically (height 0.18) and slightly widened (width 0.24)
    # so the 5 x-axis tick labels have horizontal breathing room. Pushed
    # a bit further from center to give the species title room to land.
    # y0 is set so the inset sits in the empty headroom above the data
    # (panel_ylim tops out at +130; the largest datapoint is ~+43).
    # The ratio inset now rides in panel B (Poplar) immediately right of the
    # measured Fe / Chl insets (which occupy x 0.13-0.42, y 0.05-0.45), so the
    # stoichiometric check sits beside the experimental data. The Sorghum entry
    # is retained unused in case the inset is moved back to panel A.
    inset_bounds_by_species = {
        "Sorghum": (0.70, 0.76, 0.24, 0.17),   # unused: former panel-A slot
        # y0 and height match the Chl inset (0.13, 0.05, 0.29, 0.20) so the two
        # x-axes sit on the same baseline and read as one row.
        "Poplar":  (0.50, 0.05, 0.26, 0.20),   # right of the Fe / Chl insets
    }
    inset_ylim   = (2.0, 4.0)
    inset_yticks = [2, 3, 4]

    # ----- per-svp project paths -----
    # Under the arms layout the svp is part of the directory name as well as
    # the run subdirectory, so `dir` is resolved per species from SVP_SUBDIR
    # rather than assembled from a shared PROJECTS root.
    SVP_SUBDIR  = "svp_2.0"

    # ----- species panel order: left-to-right -----
    species_configs = [
        {
            "name": "Sorghum",
            "dir": fresh_arm_dir("Sorghum", SVP_SUBDIR.replace("svp_", "")),
        },
        {
            "name": "Poplar",
            "dir": fresh_arm_dir("Poplar", SVP_SUBDIR.replace("svp_", "")),
        },
    ]

    # ----- target reactions (top row): ETC + Calvin driver set -----
    # 10 reactions total, 5 per group. Coloured by group + given
    # distinct dash patterns within the group so a reader can see
    # "these 5 lines all move together, and so do these 5" without
    # having to disentangle individual reaction identities.
    target_reactions = {
        # ETC (Photosynthesis electron transport + coupled ATPase)
        "rxn20632_y0": "Photosystem II",
        "rxn20595_y0": "Cytochrome b6f",
        "rxn26754":    "Photosystem I",
        "rxn17196":    "Fd-NADP+ reductase",
        "rxn08173_y0": "Plastidial ATP Synthase",
        # Calvin Cycle (carbon reduction + regeneration)
        "rxn00018":    "RuBisCO",
        "rxn01100":    "PGK",
        "rxn00782":    "NADPH-GAPDH",
        "rxn01111":    "PRK",
        "rxn01345":    "SBPase",
    }
    # Grouping drives the color assignment + the collapsed legend.
    reaction_groups = {
        "ETC":    ["Photosystem II", "Cytochrome b6f", "Photosystem I",
                   "Fd-NADP+ reductase", "Plastidial ATP Synthase"],
        "Calvin": ["RuBisCO", "PGK", "NADPH-GAPDH", "PRK", "SBPase"],
    }
    group_colors = {
        "ETC":    "#762a83",   # dark violet — distinct from Poplar blue, Sorghum orange, GLK gold
        "Calvin": "#01665e",   # dark teal — designed complement to violet on the ColorBrewer BrBG diverging palette
    }
    # All-solid within each group. The visual point is that every
    # ETC reaction moves together and every Calvin reaction moves
    # together — individual reaction identity is de-emphasised. The
    # caption explains that the 5 lines per group are distinct
    # reactions.
    group_linestyles = ["-"] * 5
    legend_order = [
        "Photosystem II", "Cytochrome b6f", "Photosystem I",
        "Fd-NADP+ reductase", "Plastidial ATP Synthase",
        "RuBisCO", "PGK", "NADPH-GAPDH", "PRK", "SBPase",
    ]

    # ----- bottom-row ratio reactions (rendered as inset inside top panel) -----
    ratio_numerator_id   = "rxn08173_y0"   # Plastidial ATP Synthase
    ratio_denominator_id = "rxn20632_y0"   # Photosystem II
    ratio_reference      = 3.0
    ratio_reference_label = "canonical 3:1"

    # ----- GLK ortholog transcript (bottom row, panels C & D) -----
    # Two GLK-family orthologs per species, predicted by cross-species
    # ortholog analysis to be the master regulators of the nuclear-
    # encoded photosynthesis regulon.
    # Per-gene TMM tables; resolved via BIOFLUX_RNASEQ_DIR.
    _RNASEQ_BASE = os.path.join(io_utils.rnaseq_dir(), "rnaseq-data")
    glk_data_paths = {
        "Poplar":  f"{_RNASEQ_BASE}/Poplar_raw_genes_tmm_mean.tsv",
        "Sorghum": f"{_RNASEQ_BASE}/Sorghum_raw_genes_tmm_mean.tsv.xz",
    }
    glk_orthologs = {
        # (species, slot) -> gene id in the abundance TSV.
        # Sorghum slots swapped 2026-08-04 so slot GLK1 = the grass "GLK1"-clade
        # gene (Sobic.010G096300, clusters with rice Os06g24070). The grass and
        # eudicot GLK duplications are independent, so these labels come from the
        # gene tree, not from the Arabidopsis ortholog call, which disagrees and
        # is a near-tie. Evidence: data/glk_phylogeny/README.md.
        ("Poplar",  "GLK1"): "Potri.007G136901",
        ("Poplar",  "GLK2"): "Potri.017G015800",
        ("Sorghum", "GLK1"): "Sobic.010G096300",
        ("Sorghum", "GLK2"): "Sobic.003G002600",
    }
    # Display names: Sorghum paralogs are orthology-named (SbGLK1/SbGLK2); Poplar's
    # two copies are an indistinguishable Salicaceae-WGD pair, labeled A/B.
    glk_display = {
        ("Sorghum", "GLK1"): "SbGLK1", ("Sorghum", "GLK2"): "SbGLK2",
        ("Poplar",  "GLK1"): "PtGLK-A", ("Poplar",  "GLK2"): "PtGLK-B",
    }
    # Gold family, outside the ETC purples/blues and outside the species
    # reservation (blue/orange). GLK = Golden2-Like, so the color story
    # is on-topic.
    glk_colors = {
        "GLK1": "#8c6d31",   # dark khaki-gold
        "GLK2": "#dcb54f",   # brighter gold
    }
    glk_styles = {"GLK1": "-", "GLK2": "--"}

    # ----- Rate-limiting subunit overlay on the GLK panels -----
    # For each of the 10 shared driver reactions (see
    # svp_analysis/glk_regulon_summary.py) we overlay the top
    # limiting-subunit gene's raw-diff transcript trajectory as a thin
    # line in the SAME panel as GLK1/GLK2, on a twin y-axis (right).
    # ETC subunits share a blue shade, Calvin subunits share a green
    # shade — the visual argument is "all 10 subunits track the GLK
    # regulator", not "distinguish subunit A from subunit B".
    # (The TSV filename still says "shared9"; it holds 10 reactions
    # x 2 species.  Renaming it would break the other consumers.)
    ratelimiting_gene_tsv = os.path.join(_analysis_tables(),
                                         "glk_regulon_shared9.tsv")
    ratelimiting_group_colors = {
        "ETC":    "#762a83",   # dark lavender — same as row-1 ETC
        "Calvin": "#01665e",   # dark teal — same as row-1 Calvin
    }
    ratelimiting_line_kwargs = dict(lw=0.9, alpha=0.65)

    # (species, gene_id) → override kwargs for a specific subunit line.
    # Empty by default. Populate to visually flag an outlier subunit
    # trajectory (e.g. one that anti-tracks its GLK regulator).
    ratelimiting_highlight: dict = {}

    # ----- Phenotype panels (bottom rows) -----
    # ICP-MS + reflectance-spectroscopy leaf phenotype file. Each named
    # column becomes a row of 2 species panels showing
    # log2(FeLim / Control). Ranking driving this default list is from
    # svp_analysis/find_phenotype_correlate.py — top 3 by Pearson r with
    # the ETC log2 trajectory (water-status indices), plus Chl and Fe54
    # as biology-required references.
    # Measured leaf phenotype data, not generated by this repository. Set
    # BIOFLUX_PHENOTYPE_FILE to point at it; the default is data/ beside the
    # repository root.
    phenotype_data_path = os.environ.get(
        "BIOFLUX_PHENOTYPE_FILE",
        os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "data",
                         "E1.0_Sorghum_Poplar_ICP-MS_Spec_total.txt")))
    phenotype_species_map = {
        "Poplar":  "Populus trichocarpa",
        "Sorghum": "Sorghum bicolor",
    }
    # No standalone phenotype rows: the figure is a 2 x 2 (ETC/Calvin
    # flux over GLK transcripts).  The measured readouts that used to
    # occupy rows 2+ now ride as combined-species insets inside panel B
    # --- see pheno_inset_combined below.  Set this back to e.g.
    # ["Fe54"] to restore the taller layout.
    phenotype_columns: list = []

    # Legend text per phenotype column (used in the phenotype panel
    # legends). Kept short — full explanation belongs in the caption.
    phenotype_legend_labels = {
        "WBI":   "WBI",
        "SR1":   "SR1",
        "NDWI2": "NDWI2",
        "Chl":   "Chl",
        "Fe54":  "Fe",
    }
    # One color per row. Avoids species blue/orange, ETC blue-purple,
    # and GLK gold — teal, dark red, forest, chlorophyll green, rust.
    phenotype_colors = {
        "WBI":   "#17a2b8",   # teal (water-band)
        "SR1":   "#a41e11",   # dark red (simple ratio)
        "NDWI2": "#2f5f34",   # forest green (water index)
        "Chl":   "#2ba02b",   # chlorophyll green
        "Fe54":  "#8b4513",   # rust
    }
    # Per-column y-range — auto-scaled if None. WBI is near-zero (huge
    # relative correlation but tiny absolute changes) so it needs a
    # zoom. Chl and Fe54 span a wider range.
    # Raw-diff ylims chosen from the observed per-column ranges (see
    # find_phenotype_correlate.py output); each row now shows the
    # measurement's natural scale.
    phenotype_ylims = {
        "WBI":   (-0.011, 0.005),
        "SR1":   (-0.06,  0.12),
        "NDWI2": (-0.006, 0.013),
        "Chl":   (-220,   30),   # widened from (-200, 30) so Poplar 21d fits
        # widened again from (-90, 30) so the +/- SEM whiskers fit: the
        # Sorghum 21d bar reaches -129 and the 7d bar reaches +45.
        "Fe54":  (-140,   55),
    }

    # ----- time course -----
    time_labels    = ["2d", "4d", "7d", "14d", "21d"]
    days_numeric   = [2, 4, 7, 14, 21]

    # ----- measured-phenotype inset on the GLK row (row 1) -----
    # Set to an ICP-MS / spec column name (e.g. "Chl") to draw the raw
    # Control-vs-FeLim trajectory of that measurement as a small inset
    # inside each GLK panel, pairing the transcriptional regulator with
    # the phenotype it controls.  None disables the inset entirely, so
    # the production figure is unchanged unless this is set.
    pheno_inset_column = None
    pheno_inset_bounds_by_species = {
        "Sorghum": (0.66, 0.08, 0.30, 0.26),
        "Poplar":  (0.66, 0.08, 0.30, 0.26),
    }
    # Same mechanism on the ETC/Calvin row (row 0).  Setting this and
    # emptying phenotype_columns collapses the figure to 2 x 2, with the
    # measured driver (Fe) beside the modelled flux response.  Sorghum's
    # ATPsyn/PSII inset already occupies the top-right of panel A, so the
    # row-0 slots are placed left of it.
    pheno_inset_row0_column = None
    pheno_inset_row0_bounds_by_species = {
        "Sorghum": (0.08, 0.70, 0.28, 0.26),
        "Poplar":  (0.08, 0.70, 0.28, 0.26),
    }
    pheno_inset_title = {"Chl": "Chlorophyll", "Car": "Carotenoid",
                         "Fe54": "Leaf Fe"}
    # Combined-species insets: each carries BOTH species' FeLim-minus-
    # Control trajectory for one measured column, rather than one inset
    # per panel.  Each entry is (column, host row, host col, bounds in
    # that panel's axes fraction).  Empty = disabled.  Several entries
    # may share a host panel; place them with non-overlapping bounds.
    # An optional 5th element carries per-inset options.
    #
    # Production layout: leaf Fe above chlorophyll -- the driver over the
    # phenotype it drives -- as one contiguous stack in the bottom-left
    # of panel B, where neither species' flux traces reach.  Each name is
    # a left-hand y-axis label rather than a title, since the band above
    # each inset is exactly where panel B's traces run.  One legend
    # serves the pair and its title carries the difference form.
    pheno_inset_combined: list = [
        ("Fe54", 0, 1, (0.13, 0.25, 0.29, 0.20),
         {"hide_xticklabels": True,      # shares the lower inset's axis
          "label_as_ylabel": True,
          "legend_title": "FeLim − Control",
          "legend_fontsize": 7,
          "legend_loc": "lower left"}),
        ("Chl",  0, 1, (0.13, 0.05, 0.29, 0.20),
         {"label_as_ylabel": True,
          "hide_legend": True}),         # one legend for the pair
    ]

    # ----- redundant-axis-label suppression -----
    # Applied after all panels are drawn.  Each entry is (row, col).
    # With sharex="col" the row-0 x-labels merely repeat row 1's, and in
    # a 2 x 2 grid the right column's primary y-label and the left
    # column's twin y-label repeat their neighbours.
    suppress_xlabel: list = [(0, 0), (0, 1)]   # "Time (Days)" off A and B
    suppress_ylabel: list = [(0, 1), (1, 1)]   # primary y off B and D
    suppress_twin_ylabel: list = [(1, 0)]      # twin y off C
    # Tick labels, not just the axis titles.  Dropping these on the right
    # column lets the two columns be pushed together.
    suppress_yticklabels: list = [(0, 1), (1, 1)]
    suppress_twin_yticklabels: list = [(1, 0)]
    # Horizontal gap between the two columns, applied after tight_layout.
    # None = leave tight_layout's own spacing alone.
    subplots_wspace = 0.04

    # Anchor for the row-0 (ETC / Calvin) legend, in the host panel's axes
    # fraction.  None = plain loc="upper right".  Set to the top edge of
    # panel A's ATPsyn/PSII inset so the two line up across the row.
    row0_legend_bbox = (1.0, 0.93)

    # ----- per-reaction markers (top row) -----
    # Distinct marker per reaction so overlapping curves are visible
    # as marker stacks — mass balance forces the ETC log2 trajectories
    # to nearly coincide, so the lines alone would look like one.
    reaction_markers = {
        "Photosystem II":          "o",   # circle
        "Cytochrome bf-6":         "s",   # square
        "Photosystem I":           "^",   # up triangle
        "Fd-NADP+ reductase":      "D",   # diamond
        "Plastidial ATP Synthase": "v",   # down triangle
    }
    # ----- colors (top row) -----
    reaction_colors = {
        "Photosystem II":          "#2c7fb8",
        "Cytochrome bf-6":         "#7570b3",
        "Photosystem I":           "#1b9e77",
        "Fd-NADP+ reductase":      "#e7298a",
        "Plastidial ATP Synthase": "#d95f0e",
    }
    # ----- colors / styles (inset, top row) -----
    # Blue / orange are reserved for the species palette; the five ETC
    # reactions in the main panel claim blue, purple, green-teal,
    # magenta, and orange. Black + crimson is maximally print-safe
    # (high lightness contrast, survives B&W and low-quality color
    # printing) and stays outside the existing palette.
    treatment_style = {"Control": "-", "FeLim": "--"}
    treatment_color = {"Control": "#000000", "FeLim": "#b22222"}
    species_palette = dict(SPECIES_COLORS)

    # ----- layout -----
    # Rows = 2 (ETC flux + GLK transcript) + len(phenotype_columns).
    # Computed inside build_figure() so shortening / lengthening
    # phenotype_columns automatically adjusts the panel grid + height.
    # These class-level values act as fallbacks for old callers only.
    figsize = (13, 30)
    layout  = (7, 2)

    # ----- output -----
    filename = "fig_photo_etc"
# ==================================================================


def _extract_reaction_flux_per_cond(v_df: pd.DataFrame, model: cobra.Model,
                                    target_id: str, cond_names: list[str],
                                    cond_indices: list[int]) -> np.ndarray:
    """Sum signed-direction columns matching ``target_id`` in the V TSV.

    Reactions can be split into ``_f`` / ``_o`` (forward, sign +1) and
    ``_r`` / ``_i`` (reverse, sign -1) flux columns. We sum them with
    the appropriate sign to recover net flux. Returns an array aligned
    to ``cond_indices``.

    Two things this has to get right, both of which an earlier version
    did not:

    * ``base_id`` and ``clean_id`` are identical whenever the model id
      carries no ``R_``/``EX_`` prefix (which is every reaction here,
      since cobra strips the prefix on read).  Collecting candidate
      column names in a ``set`` keeps each column from being added
      twice -- the previous list-of-pairs form doubled every value.
    * The split halves of a reversible reaction are *separate model
      objects* whose ids already end in ``_f``/``_r``.  The sign must
      therefore be read off the column name itself; deriving it from a
      constructed ``f"{base_id}_r"`` never matched, so reverse flux was
      added rather than subtracted.
    """
    flux = np.zeros(len(cond_indices))
    seen: set[str] = set()
    for rxn in model.reactions:
        base_id = rxn.id
        clean_id = base_id.replace("R_", "").replace("EX_", "").strip()
        if clean_id != target_id and base_id != target_id and \
           not clean_id.startswith(target_id + "_"):
            continue
        candidates = {stem_sfx
                      for stem in (base_id, clean_id)
                      for stem_sfx in (stem, f"{stem}_f", f"{stem}_o",
                                       f"{stem}_r", f"{stem}_i")}
        for col_name in sorted(candidates):
            if col_name not in v_df.columns or col_name in seen:
                continue
            seen.add(col_name)
            sign = -1 if col_name.endswith(("_r", "_i")) else 1
            flux += sign * v_df[col_name].values[cond_indices]
    return flux


def _load_species_data(config: dict) -> dict | None:
    """Load V matrix and SBML model for a species; return parsed flux
    series for the 5 target reactions plus PSII/ATPsyn totals."""
    base = os.path.join(config["dir"], "ml")
    npz_path = os.path.join(base, "training", "training.npz")
    xml_path = os.path.join(base, "training", "training.xml")
    tsv_path = os.path.join(base, CONFIG.SVP_SUBDIR, "results",
                            "startVbfandZero_noRelu_V_headers.tsv")
    for p in (npz_path, xml_path, tsv_path):
        if not os.path.exists(p):
            print(f"  -> Missing {p}; skipping {config['name']}")
            return None

    data = np.load(npz_path, allow_pickle=True)
    cond_names = [n.decode("utf-8") if isinstance(n, bytes) else str(n)
                  for n in data["treatments"]]
    ctrl_conds  = [f"Leaf_Control_{t}" for t in CONFIG.time_labels]
    felim_conds = [f"Leaf_FeLim_{t}"   for t in CONFIG.time_labels]
    ctrl_idx  = [cond_names.index(c) for c in ctrl_conds  if c in cond_names]
    felim_idx = [cond_names.index(c) for c in felim_conds if c in cond_names]
    if len(ctrl_idx) != len(CONFIG.time_labels) or len(felim_idx) != len(CONFIG.time_labels):
        print(f"  -> Incomplete time-course for {config['name']}; skipping.")
        return None

    model = cobra.io.read_sbml_model(xml_path)
    v_df = pd.read_csv(tsv_path, sep="\t")
    v_df.columns = v_df.columns.str.strip()

    # Top row: raw (FeLim − Control) per target reaction. Each
    # reaction's absolute flux magnitude comes through directly, so
    # the 5 ETC lines visually separate (PSII ~40 delta, PSI ~300
    # delta at Sorghum 7d etc.). Same formulation as the GLK and
    # phenotype rows below.
    delta_flux = {}
    for target_id, label in CONFIG.target_reactions.items():
        flux_ctrl  = _extract_reaction_flux_per_cond(v_df, model, target_id,
                                                     cond_names, ctrl_idx)
        flux_felim = _extract_reaction_flux_per_cond(v_df, model, target_id,
                                                     cond_names, felim_idx)
        delta_flux[label] = (flux_felim - flux_ctrl) * \
            CONFIG.display_direction.get(target_id, 1)

    # Bottom row: per-condition ATPsyn / PSII ratio for Control and FeLim.
    atpsyn_ctrl  = _extract_reaction_flux_per_cond(v_df, model, CONFIG.ratio_numerator_id,
                                                    cond_names, ctrl_idx)
    psii_ctrl    = _extract_reaction_flux_per_cond(v_df, model, CONFIG.ratio_denominator_id,
                                                    cond_names, ctrl_idx)
    atpsyn_felim = _extract_reaction_flux_per_cond(v_df, model, CONFIG.ratio_numerator_id,
                                                    cond_names, felim_idx)
    psii_felim   = _extract_reaction_flux_per_cond(v_df, model, CONFIG.ratio_denominator_id,
                                                    cond_names, felim_idx)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_ctrl  = np.where(psii_ctrl  != 0, atpsyn_ctrl  / psii_ctrl,  np.nan)
        ratio_felim = np.where(psii_felim != 0, atpsyn_felim / psii_felim, np.nan)

    return {
        "name": config["name"],
        "delta_flux": delta_flux,
        "ratio_ctrl": ratio_ctrl,
        "ratio_felim": ratio_felim,
    }


def _load_glk_data(species_name: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {glk_label: (control_array, felim_array)} aligned to CONFIG.time_labels.

    Reads the per-species TMM-normalised transcript abundance TSV (xz
    decompression handled transparently), filters to the two GLK
    ortholog gene IDs listed in CONFIG.glk_orthologs, and picks out
    the ``Leaf_Control_<tp>`` and ``Leaf_FeLim_<tp>`` rows.
    """
    path = CONFIG.glk_data_paths[species_name]
    opener = lzma.open if path.endswith(".xz") else open
    with opener(path, "rt") as fh:
        df = pd.read_csv(fh, sep="\t")

    glk_ids = {label: gene
               for (sp, label), gene in CONFIG.glk_orthologs.items()
               if sp == species_name}
    df = df[df["Gene_ID"].isin(glk_ids.values())]

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, gene in glk_ids.items():
        sub = df[df["Gene_ID"] == gene].set_index("condition")["value"]
        ctrl  = np.array([float(sub.get(f"Leaf_Control_{t}", np.nan))
                          for t in CONFIG.time_labels])
        felim = np.array([float(sub.get(f"Leaf_FeLim_{t}",  np.nan))
                          for t in CONFIG.time_labels])
        out[label] = (ctrl, felim)
    return out


def _load_ratelimiting_genes(species_name: str
                             ) -> list[tuple[str, str, np.ndarray]]:
    """Return list of (group, gene_id, delta_trajectory) for the
    9 shared-driver rate-limiting genes for one species. Delta
    trajectory = FeLim - Control TMM per timepoint aligned with
    CONFIG.time_labels. Reads the ranked table produced by
    svp_analysis/glk_regulon_summary.py plus the raw TMM file."""
    ranked = pd.read_csv(CONFIG.ratelimiting_gene_tsv, sep="\t")
    ranked = ranked[ranked["species"] == species_name]

    # Reuse GLK loader's TMM opening logic
    tmm_path = CONFIG.glk_data_paths[species_name]
    opener = lzma.open if tmm_path.endswith(".xz") else open
    with opener(tmm_path, "rt") as fh:
        tmm = pd.read_csv(fh, sep="\t")

    genes_needed = ranked["gene_id"].tolist()
    tmm = tmm[tmm["Gene_ID"].isin(genes_needed)]

    out = []
    for _, row in ranked.iterrows():
        gene = row["gene_id"]
        group = row["group"]
        sub = tmm[tmm["Gene_ID"] == gene].set_index("condition")["value"]
        delta = np.array([
            (float(sub.get(f"Leaf_FeLim_{t}", np.nan))
             - float(sub.get(f"Leaf_Control_{t}", np.nan)))
            if pd.notna(sub.get(f"Leaf_FeLim_{t}", np.nan))
               and pd.notna(sub.get(f"Leaf_Control_{t}", np.nan))
            else np.nan
            for t in CONFIG.time_labels
        ])
        out.append((group, gene, delta))
    return out


def _load_phenotype_data(species_name: str) -> dict[str, tuple]:
    """Return {column: (control_mean, felim_mean, control_sem, felim_sem)}
    for the ICP-MS / spec phenotype columns listed in
    CONFIG.phenotype_columns, aggregated over replicates per
    (species, treatment, timepoint) and aligned with CONFIG.time_labels.
    Only Leaf tissue is used (matching Figure 8).

    The SEMs are returned per treatment arm rather than pre-combined so
    the caller can propagate them onto whatever contrast it plots; the
    panels plot FeLim minus Control, whose SEM is the quadrature sum.
    Groups with a single replicate get SEM = 0 (pandas std is NaN there).
    """
    df = pd.read_csv(CONFIG.phenotype_data_path, sep="\t")
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df = df[df["Tissue"] == "Leaf"].copy()
    # Normalize "2 d" -> "2d" so keys match CONFIG.time_labels.
    df["Timepoint"] = df["Timepoint"].str.replace(" ", "", regex=False)
    df = df[df["Species"] == CONFIG.phenotype_species_map[species_name]]
    df = df[df["Treatment"].isin(("Control", "FeLim"))]
    # Row columns plus, if enabled, the GLK-row inset column.
    cols = list(CONFIG.phenotype_columns)
    extras = [CONFIG.pheno_inset_column, CONFIG.pheno_inset_row0_column]
    extras += [spec[0] for spec in CONFIG.pheno_inset_combined]
    for extra in extras:
        if extra and extra not in cols:
            cols.append(extra)
    # Mean, replicate count and SD per (Treatment, Timepoint)
    grp = df.groupby(["Treatment", "Timepoint"])[cols]
    mean_df = grp.mean(numeric_only=True)
    sd_df   = grp.std(numeric_only=True, ddof=1)
    n_df    = grp.count()

    def _series(frame, treatment, col, default=np.nan):
        return np.array([
            float(frame.loc[(treatment, t), col])
            if (treatment, t) in frame.index else default
            for t in CONFIG.time_labels
        ])

    out: dict[str, tuple] = {}
    for col in cols:
        ctrl      = _series(mean_df, "Control", col)
        felim     = _series(mean_df, "FeLim",   col)
        ctrl_sd   = _series(sd_df,   "Control", col)
        felim_sd  = _series(sd_df,   "FeLim",   col)
        ctrl_n    = _series(n_df,    "Control", col, default=0.0)
        felim_n   = _series(n_df,    "FeLim",   col, default=0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ctrl_sem  = np.nan_to_num(ctrl_sd  / np.sqrt(ctrl_n))
            felim_sem = np.nan_to_num(felim_sd / np.sqrt(felim_n))
        out[col] = (ctrl, felim, ctrl_sem, felim_sem)
    return out


def _plot_phenotype_panel(ax, species_name: str,
                          pheno_data: dict[str, tuple],
                          col_name: str) -> None:
    """One phenotype row per species: raw (FeLim − Control)
    trajectory for a single ICP-MS / spec column, with error bars. Each
    row keeps its measurement's natural units — Chl and Fe54 in mass
    concentration, NDVI / NDWI / WBI / SR1 dimensionless — so per-row
    ylims are required to match each column's amplitude range. Row
    identity is conveyed by a short in-panel legend; full description
    belongs in the figure caption.

    The two treatment arms are independent samples (n = 3 biological
    replicates each), so the standard error of their difference is the
    quadrature sum of the per-arm standard errors.  Error bars matter
    here: the Sorghum ICP-MS replicates are far more dispersed than the
    Poplar ones, and without them the Sorghum trajectory reads as more
    certain than it is."""
    days = CONFIG.days_numeric
    ctrl, felim, ctrl_sem, felim_sem = pheno_data[col_name]
    diff = felim - ctrl
    sem_diff = np.sqrt(ctrl_sem ** 2 + felim_sem ** 2)
    color = CONFIG.phenotype_colors.get(col_name, "#444444")
    legend_label = CONFIG.phenotype_legend_labels.get(col_name, col_name)
    ax.errorbar(days, diff, yerr=sem_diff, color=color, marker="o",
                lw=2.2, markersize=6, capsize=4, elinewidth=1.4,
                capthick=1.4, label=legend_label)
    ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax.set_xticks(days)
    ax.set_xticklabels(CONFIG.time_labels)
    ax.set_xlabel(CONFIG.panel_labels["x_bottom"])
    ax.set_ylabel(r"FeLim − Control")
    ylim = CONFIG.phenotype_ylims.get(col_name)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _plot_bottom_panel(ax, species_name: str,
                       glk_data: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """GLK ortholog row: raw (FeLim − Control) TMM per ortholog on
    the LEFT y-axis (zoomed to ±320 TMM), plus the 9 rate-limiting-
    subunit gene trajectories for the shared driver reactions on a
    TWIN RIGHT y-axis (±9000 TMM). Both axes are symmetric around
    zero so the y=0 lines physically align at panel-center, making
    trajectory shape comparison direct despite the ~25x amplitude
    gap between low-abundance regulators and high-abundance targets."""
    days = CONFIG.days_numeric
    # GLK orthologs on the primary (left) axis
    for label in ("GLK1", "GLK2"):
        if label not in glk_data:
            continue
        ctrl, felim = glk_data[label]
        diff = felim - ctrl
        ax.plot(days, diff,
                color=CONFIG.glk_colors[label],
                linestyle=CONFIG.glk_styles[label],
                marker="o", lw=2.2, markersize=6,
                label=CONFIG.glk_display.get((species_name, label), label),
                zorder=3)   # keep GLKs above the subunit overlay
    # Rate-limiting subunit overlay on the twin (right) axis
    try:
        subunits = _load_ratelimiting_genes(species_name)
    except FileNotFoundError:
        subunits = []
    if subunits:
        ax_r = ax.twinx()
        seen_groups: set[str] = set()
        for group, gene_id, delta in subunits:
            highlight = CONFIG.ratelimiting_highlight.get((species_name, gene_id))
            if highlight is not None:
                # Distinct styling + explicit legend entry for outlier gene
                ax_r.plot(days, delta, zorder=4, **highlight)
                continue
            color = CONFIG.ratelimiting_group_colors.get(group, "#888888")
            lbl = None
            if group not in seen_groups:
                lbl = f"{group} subunits"
                seen_groups.add(group)
            ax_r.plot(days, delta, color=color, label=lbl,
                      **CONFIG.ratelimiting_line_kwargs)
        # Symmetric bounds → y=0 sits at the same 50% panel height on
        # both axes, so drop trajectories overlay cleanly at the same
        # visual reference line. The tick pair (yticks_bottom /
        # yticks_bottom_subunit) is set so ±300 primary lines up with
        # ±8000 twin, ±150 lines up with ±4000, etc.
        ax_r.set_ylim(*CONFIG.panel_ylim_bottom_subunit)
        ax_r.set_yticks(CONFIG.panel_yticks_bottom_subunit)
        ax_r.set_ylabel("Subunit ΔTMM (FeLim − Control)", fontsize=9)
        # Stash the twin so build_figure's legend logic can retrieve
        # its handles alongside the primary axis's GLK handles.
        ax._twin_ax_bottom = ax_r
    ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax.set_xticks(days)
    ax.set_xticklabels(CONFIG.time_labels)
    ax.set_xlabel(CONFIG.panel_labels["x_bottom"])
    ax.set_ylabel(CONFIG.panel_labels["bottom_y"])
    if CONFIG.panel_ylim_bottom is not None:
        ax.set_ylim(*CONFIG.panel_ylim_bottom)
    if CONFIG.panel_yticks_bottom is not None:
        ax.set_yticks(CONFIG.panel_yticks_bottom)

    # ---- Average Pearson r across the 9 rate-limiting subunit genes
    # vs GLK1 and GLK2 (already precomputed in glk_regulon_shared9.tsv).
    # Two-line summary aligned on the equal sign: prefix ("avg r(GLKn) = ")
    # is right-anchored at equals_x, and the value ("+0.446") is left-
    # anchored at the same equals_x. Result: the "=" sign sits in the
    # same column on both lines. Only the value on the winning paralog
    # line is bolded; text weight of the prefix stays normal.
    try:
        ranked = pd.read_csv(CONFIG.ratelimiting_gene_tsv, sep="\t")
        sub = ranked[ranked["species"] == species_name]
        if not sub.empty:
            avg_r1 = float(sub["r_GLK1"].mean())
            avg_r2 = float(sub["r_GLK2"].mean())
            g1_higher = avg_r1 > avg_r2
            trans = ax.transAxes
            equals_x, y_top, dy = 0.86, 0.955, 0.048

            def _write_line(y, label, val, bold_value):
                prefix = f"avg r({label}) = "
                value  = f"{val:+.3f}"
                # Prefix — right-anchored, so its right edge (past the "=")
                # sits at equals_x. Trailing space in the prefix pushes the
                # "=" glyph a touch to the left of equals_x, leaving a small
                # visual gap before the value.
                ax.text(equals_x, y, prefix,
                        transform=trans,
                        fontsize=9, fontweight="normal",
                        va="top", ha="right", color="#111111",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="white",
                                  edgecolor="none", alpha=0.85),
                        zorder=6)
                # Value — left-anchored at the same equals_x. Because both
                # lines share equals_x, the equal signs line up vertically
                # regardless of whether the value is bold on this line.
                ax.text(equals_x, y, value,
                        transform=trans,
                        fontsize=9,
                        fontweight=("bold" if bold_value else "normal"),
                        va="top", ha="left", color="#111111",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="white",
                                  edgecolor="none", alpha=0.85),
                        zorder=6)

            _write_line(y_top,      CONFIG.glk_display.get((species_name, "GLK1"), "GLK1"), avg_r1, g1_higher)
            _write_line(y_top - dy, CONFIG.glk_display.get((species_name, "GLK2"), "GLK2"), avg_r2, not g1_higher)
    except (FileNotFoundError, KeyError) as e:
        print(f"  -> Could not annotate avg-r for {species_name}: {e}")


def _style_for_label(label: str) -> tuple[str, object, str | None]:
    """Return (color, linestyle, group_name) for a reaction label
    based on CONFIG.reaction_groups + group_colors + group_linestyles.
    Reactions in the same group share a color; their linestyle is
    picked from group_linestyles based on their index in the group."""
    for group_name, labels_in_group in CONFIG.reaction_groups.items():
        if label in labels_in_group:
            idx = labels_in_group.index(label)
            ls = CONFIG.group_linestyles[idx % len(CONFIG.group_linestyles)]
            return CONFIG.group_colors[group_name], ls, group_name
    return "#444444", "-", None


def _plot_top_panel(ax, species_data: dict):
    """Per-reaction net flux difference (FeLim − Control) vs time.
    Two reaction groups (ETC, Calvin), each in one color with 5
    distinct dash patterns.

    Per-timepoint marker opacity encodes rel_mag continuously
    (rel_mag = |mean_delta across p| / max(mean_ctl, mean_fel),
    computed from the raw timecourse data at that species/reaction/tp).
    Full opacity at rel_mag >= 10%; minimum opacity at rel_mag <= 0.5%.
    A black 'x' overlay marks timepoints where the direction of the
    response is not consistent across the four p values (i.e. the
    principled non-defensibility — signal actually flips signs)."""
    days = CONFIG.days_numeric
    time_labels = CONFIG.time_labels
    sp = species_data["name"]
    metrics = _load_per_tp_metrics()
    label_to_base = {label: _base_rxn(rid)
                     for rid, label in CONFIG.target_reactions.items()}
    for label in CONFIG.legend_order:
        if label not in species_data["delta_flux"]:
            continue
        color, linestyle, _ = _style_for_label(label)
        y = species_data["delta_flux"][label]
        # Line without markers (markers drawn per-point next)
        ax.plot(days, y, color=color, linestyle=linestyle,
                lw=1.6, label=label)
        base = label_to_base.get(label)
        for xi, yi, tp in zip(days, y, time_labels):
            met = metrics.get((sp, base, tp))
            if met is None:
                ax.scatter(xi, yi, s=28, color=color, alpha=0.4,
                           marker="o", edgecolors="none", zorder=5)
                continue
            rel_mag, dir_flip = met
            alpha = _alpha_for_rel_mag(rel_mag)
            ax.scatter(xi, yi, s=28, color=color, alpha=alpha,
                       marker="o", edgecolors="none", zorder=5)
            if dir_flip:
                ax.scatter(xi, yi, s=110, color="black", marker="x",
                           linewidths=1.8, zorder=6)
    ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax.set_xticks(days)
    ax.set_xticklabels(CONFIG.time_labels)
    ax.set_xlabel(CONFIG.panel_labels["x_top"])
    ax.set_ylabel(CONFIG.panel_labels["top_y"])
    if CONFIG.panel_ylim is not None:
        ax.set_ylim(*CONFIG.panel_ylim)
    # In-plot species title — placed at the top-center of the panel
    # (between the two corner regions where insets / legend live), in
    # the species color.
    sp_name = species_data["name"]
    sp_color = CONFIG.species_palette.get(sp_name, "#222")
    ax.text(0.5, 0.97, sp_name,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=14, fontweight="bold",
            color=sp_color)


def _plot_inset_ratio(parent_ax, species_data: dict):
    """Inset axes inside ``parent_ax`` showing ATPsyn / PSII ratio over
    the Leaf time course for Control vs FeLim. No reference line — the
    fixed y-range (2.0-4.0) makes 3.0 visually obvious as the centre."""
    days = CONFIG.days_numeric
    bounds = CONFIG.inset_bounds_by_species.get(
        species_data["name"], (0.06, 0.06, 0.28, 0.28))
    ax = parent_ax.inset_axes(bounds)
    # Control: small filled dark circle on solid line.
    # FeLim: larger fully-transparent diamond (markerfacecolor="none")
    # on dashed line — sits on top of Control with empty interior, so
    # the Control marker shows through. Distinguished from Control by
    # marker shape, line style, and lighter edge color.
    ax.plot(days, species_data["ratio_ctrl"],
            color=CONFIG.treatment_color["Control"],
            linestyle=CONFIG.treatment_style["Control"],
            marker="o", markerfacecolor=CONFIG.treatment_color["Control"],
            markeredgecolor=CONFIG.treatment_color["Control"],
            lw=1.4, markersize=3, label="Control")
    ax.plot(days, species_data["ratio_felim"],
            color=CONFIG.treatment_color["FeLim"],
            linestyle=CONFIG.treatment_style["FeLim"],
            marker="D", markerfacecolor="none",
            markeredgecolor=CONFIG.treatment_color["FeLim"],
            markeredgewidth=0.9,
            lw=1.4, markersize=4.5, label="FeLim")
    ax.set_ylim(*CONFIG.inset_ylim)
    ax.set_yticks(CONFIG.inset_yticks)
    ax.set_xticks(days)
    ax.set_xticklabels(CONFIG.time_labels, fontsize=6)
    ax.tick_params(axis="y", labelsize=6)
    # numerator is ATP synthase, denominator PSII -- the ratio plotted is
    # ATPsyn/PSII (= 3), so the title must not be written the other way up.
    ax.set_title("ATPsyn / PSII ratio", fontsize=7, pad=2)
    ax.legend(loc="upper right", fontsize=5, frameon=True, framealpha=0.85,
              handlelength=1.2, borderpad=0.2, labelspacing=0.15)
    # Light background so the inset is visually separated from the parent panel
    ax.set_facecolor((1.0, 1.0, 1.0, 0.85))
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def _plot_inset_pheno(parent_ax, species_name: str,
                      pheno_data: dict[str, tuple], col_name: str,
                      bounds_map: dict | None = None):
    """Inset inside a GLK panel showing the measured phenotype column
    ``col_name`` over the Leaf time course, Control vs FeLim, in its raw
    units with per-arm SEM bars.

    Raw arms rather than the FeLim-minus-Control difference plotted by
    the phenotype rows: at inset size the reader needs to see the FeLim
    arm fall away from a flat Control, and a difference trace hides the
    fact that Control itself barely moves.  Marker convention matches
    _plot_inset_ratio (filled circle / open diamond) so the two inset
    types read the same way."""
    if col_name not in pheno_data:
        return
    days = CONFIG.days_numeric
    ctrl, felim, ctrl_sem, felim_sem = pheno_data[col_name]
    if bounds_map is None:
        bounds_map = CONFIG.pheno_inset_bounds_by_species
    bounds = bounds_map.get(species_name, (0.66, 0.08, 0.30, 0.26))
    ax = parent_ax.inset_axes(bounds)
    ax.errorbar(days, ctrl, yerr=ctrl_sem,
                color=CONFIG.treatment_color["Control"],
                linestyle=CONFIG.treatment_style["Control"],
                marker="o", markerfacecolor=CONFIG.treatment_color["Control"],
                markeredgecolor=CONFIG.treatment_color["Control"],
                lw=1.4, markersize=3, capsize=2, elinewidth=0.8,
                capthick=0.8, label="Control")
    ax.errorbar(days, felim, yerr=felim_sem,
                color=CONFIG.treatment_color["FeLim"],
                linestyle=CONFIG.treatment_style["FeLim"],
                marker="D", markerfacecolor="none",
                markeredgecolor=CONFIG.treatment_color["FeLim"],
                markeredgewidth=0.9,
                lw=1.4, markersize=4.5, capsize=2, elinewidth=0.8,
                capthick=0.8, label="FeLim")
    ax.set_xticks(days)
    ax.set_xticklabels(CONFIG.time_labels, fontsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_title(CONFIG.pheno_inset_title.get(col_name, col_name),
                 fontsize=7, pad=2)
    ax.legend(loc="lower left", fontsize=5, frameon=True, framealpha=0.85,
              handlelength=1.2, borderpad=0.2, labelspacing=0.15)
    # Fully opaque: the GLK panel carries ~20 subunit traces, and at 0.85
    # they bleed through the inset and read as inset data.
    ax.set_facecolor("white")
    ax.set_zorder(6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    return ax


def _plot_inset_pheno_combined(parent_ax, pheno_by_species: dict,
                               col_name: str, bounds: tuple,
                               opts: dict | None = None):
    """Single inset carrying BOTH species' FeLim-minus-Control trajectory
    for one measured column, coloured by the species palette used in the
    panel titles.

    The difference form is what the standalone phenotype rows plot, and
    it is what makes the two species commensurable in one axes: the raw
    arms sit at different absolute levels (Poplar leaves hold more
    chlorophyll than Sorghum leaves throughout), so overlaying raw
    Control and FeLim for both species would need four traces and two
    y-scales. Differencing removes the species offset and leaves the
    treatment effect, which is the comparison the panel is making.

    Error bars are the quadrature sum of the two per-arm SEMs, matching
    _plot_phenotype_panel."""
    opts = opts or {}
    days = CONFIG.days_numeric
    ax = parent_ax.inset_axes(bounds)
    for sp_name, pheno in pheno_by_species.items():
        if col_name not in pheno:
            continue
        ctrl, felim, ctrl_sem, felim_sem = pheno[col_name]
        ax.errorbar(days, felim - ctrl,
                    yerr=np.sqrt(ctrl_sem ** 2 + felim_sem ** 2),
                    color=CONFIG.species_palette.get(sp_name, "#444444"),
                    marker="o", lw=1.6, markersize=3.5, capsize=2,
                    elinewidth=0.8, capthick=0.8, label=sp_name)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(days)
    # When two of these are stacked on one time axis, only the lower one
    # needs tick labels -- the upper one's would land on the lower one's
    # title.
    if opts.get("hide_xticklabels"):
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(CONFIG.time_labels, fontsize=6)
    ax.tick_params(axis="y", labelsize=6)
    name = CONFIG.pheno_inset_title.get(col_name, col_name)
    if opts.get("label_as_ylabel"):
        # Naming the quantity on the left frees the band above the axes,
        # which is where the parent panel's traces run.
        ax.set_ylabel(name, fontsize=7, labelpad=2)
    else:
        # Opaque bbox: the title is drawn OUTSIDE the inset axes, so
        # without it the parent's traces run through the lettering.
        ax.set_title("{} (FeLim − Control)".format(name),
                     fontsize=7, pad=2,
                     bbox=dict(boxstyle="square,pad=0.15",
                               facecolor="white", edgecolor="none"))
    if opts.get("hide_legend"):
        pass
    else:
        # The legend title carries the difference form for the whole
        # stack, so each inset does not have to repeat it.
        ax.legend(loc=opts.get("legend_loc", "lower left"),
                  title=opts.get("legend_title"),
                  fontsize=opts.get("legend_fontsize", 5),
                  title_fontsize=opts.get("legend_fontsize", 5),
                  frameon=True, framealpha=0.85,
                  handlelength=1.2, borderpad=0.2, labelspacing=0.15)
    ax.set_facecolor("white")
    ax.set_zorder(6)
    return ax


def build_figure():
    apply_style()
    n_pheno = len(CONFIG.phenotype_columns)
    n_rows  = 2 + n_pheno
    figsize = (13, 5 + 4 * n_rows)   # height scales with row count
    print(f"Building figure 8: {n_rows} x 2 layout "
          f"(ETC + GLK + {n_pheno} phenotype rows)")

    series = [_load_species_data(cfg) for cfg in CONFIG.species_configs]
    series = [s for s in series if s is not None]
    if not series:
        raise RuntimeError("No species data loaded; cannot build figure.")

    print("Loading GLK ortholog transcript abundances...")
    glk_by_species = {sp["name"]: _load_glk_data(sp["name"])
                      for sp in CONFIG.species_configs
                      if sp["name"] in [s["name"] for s in series]}

    pheno_by_species: dict[str, dict] = {}
    if (n_pheno > 0 or CONFIG.pheno_inset_column
            or CONFIG.pheno_inset_row0_column or CONFIG.pheno_inset_combined):
        print("Loading ICP-MS / spec phenotype data...")
        for sp in CONFIG.species_configs:
            if sp["name"] in [s["name"] for s in series]:
                pheno_by_species[sp["name"]] = _load_phenotype_data(sp["name"])

    fig, axes = plt.subplots(n_rows, 2, figsize=figsize,
                              sharey=False, sharex="col")
    # axes shape (n_rows, 2). col = species, row 0 = ETC, row 1 = GLK,
    # rows 2.. = phenotype (one per CONFIG.phenotype_columns entry).

    for col, sp_data in enumerate(series):
        _plot_top_panel(axes[0, col], sp_data)
        # Inset only on the Poplar (right-column) panel, beside the measured
        # Fe / Chl insets, so the modelled stoichiometric check reads next to
        # the experimental data. The ratio is 3.0 in both species — it is
        # forced by the H+ stoichiometry, not fitted — so one panel suffices
        # and it shows that panel's own species. Caption must say so.
        if col == 1:
            _plot_inset_ratio(axes[0, col], sp_data)
        glk = glk_by_species.get(sp_data["name"], {})
        _plot_bottom_panel(axes[1, col], sp_data["name"], glk)

        pheno = pheno_by_species.get(sp_data["name"], {})
        if CONFIG.pheno_inset_column:
            _plot_inset_pheno(axes[1, col], sp_data["name"], pheno,
                              CONFIG.pheno_inset_column)
        if CONFIG.pheno_inset_row0_column:
            _plot_inset_pheno(axes[0, col], sp_data["name"], pheno,
                              CONFIG.pheno_inset_row0_column,
                              CONFIG.pheno_inset_row0_bounds_by_species)
        for pcol_idx, pcol_name in enumerate(CONFIG.phenotype_columns):
            _plot_phenotype_panel(axes[2 + pcol_idx, col],
                                  sp_data["name"], pheno, pcol_name)

    # ---- Combined-species phenotype insets.
    for spec in CONFIG.pheno_inset_combined:
        pcol, host_row, host_col, bounds = spec[:4]
        opts = spec[4] if len(spec) > 4 else {}
        if host_row < axes.shape[0] and host_col < axes.shape[1]:
            _plot_inset_pheno_combined(axes[host_row, host_col],
                                       pheno_by_species, pcol, bounds, opts)

    # ---- Legends: one per row, in the right-column (Poplar) panel,
    # top-right corner. Row 1 uses collapsed group representatives
    # (ETC + Calvin) rather than 10 individual reactions.
    from matplotlib.lines import Line2D

    # Row 1 (ETC + Calvin): 2-entry group legend using representative
    # solid lines in each group's color.
    top_right = axes[0, -1]
    group_handles = [
        Line2D([0], [0], color=CONFIG.group_colors[g], lw=2.2, linestyle="-",
               marker="o", markersize=5)
        for g in CONFIG.reaction_groups
    ]
    group_labels = list(CONFIG.reaction_groups.keys())
    _row0_leg_kw = dict(loc="upper right", fontsize=9,
                        frameon=True, framealpha=0.9)
    if CONFIG.row0_legend_bbox is not None:
        _row0_leg_kw["bbox_to_anchor"] = CONFIG.row0_legend_bbox
    top_right.legend(group_handles, group_labels, **_row0_leg_kw)

    # Row 2 (GLK on primary axis + rate-limiting subunit on twin):
    # ONE shared legend, drawn on the Poplar (right-column) panel.
    #
    # The two GLK paralogs are named per species (Sorghum SbGLK1/SbGLK2,
    # Poplar PtGLK-A/PtGLK-B) but share a color and linestyle, because
    # glk_colors/glk_styles are keyed by SLOT, not by species. So one
    # legend key can serve both columns provided it carries both names.
    # We therefore build proxy handles whose label stacks the two names
    # on two lines -- Sorghum (panel C) above Poplar (panel D), matching
    # the left-to-right panel order. Matplotlib centers the line sample
    # vertically against a multi-line label, so the single swatch reads
    # as belonging to both names equally.
    #
    # The subunit-group entries (ETC / Calvin) are species-independent
    # and are taken from the Poplar twin axis as-is.
    from matplotlib.lines import Line2D

    poplar_ax = axes[1, -1]
    glk_handles = [
        Line2D([], [],
               color=CONFIG.glk_colors[slot],
               linestyle=CONFIG.glk_styles[slot],
               marker="o", lw=2.2, markersize=6)
        for slot in ("GLK1", "GLK2")
    ]
    glk_labels = [
        "{}\n{}".format(CONFIG.glk_display[("Sorghum", slot)],
                        CONFIG.glk_display[("Poplar", slot)])
        for slot in ("GLK1", "GLK2")
    ]
    twin = getattr(poplar_ax, "_twin_ax_bottom", None)
    if twin is not None:
        th, tl = twin.get_legend_handles_labels()
        glk_handles += th
        glk_labels  += tl
    if glk_handles:
        # Legend sits at the bottom-left of Panel D.  The panel letter is
        # top-left and the avg-r annotation top-right, so the bottom is
        # the only block of the panel free of both.
        poplar_ax.legend(glk_handles, glk_labels, loc="lower left",
                         bbox_to_anchor=(0.09, 0.02),
                         fontsize=8, frameon=True, framealpha=0.9,
                         labelspacing=0.4, ncol=2)
        for col in range(axes.shape[1] - 1):
            if axes[1, col].get_legend() is not None:
                axes[1, col].get_legend().remove()

    # Rows 3+ (phenotype): one legend per row, on the Poplar panel,
    # top-right.
    for row_idx in range(2, 2 + len(CONFIG.phenotype_columns)):
        p_ax = axes[row_idx, -1]
        p_h, p_l = p_ax.get_legend_handles_labels()
        if p_h:
            p_ax.legend(p_h, p_l, loc="upper right", fontsize=9,
                        frameon=True, framealpha=0.9)

    # ---- Drop redundant axis labels. Done as a post-pass rather than in
    # the panel plotters so the per-panel functions stay layout-agnostic.
    def _in_grid(r, c):
        return r < axes.shape[0] and c < axes.shape[1]

    for r, c in CONFIG.suppress_xlabel:
        if _in_grid(r, c):
            axes[r, c].set_xlabel("")
    for r, c in CONFIG.suppress_ylabel:
        if _in_grid(r, c):
            axes[r, c].set_ylabel("")
    for r, c in CONFIG.suppress_twin_ylabel:
        if _in_grid(r, c):
            twin_ax = getattr(axes[r, c], "_twin_ax_bottom", None)
            if twin_ax is not None:
                twin_ax.set_ylabel("")
    for r, c in CONFIG.suppress_yticklabels:
        if _in_grid(r, c):
            axes[r, c].tick_params(axis="y", labelleft=False)
    for r, c in CONFIG.suppress_twin_yticklabels:
        if _in_grid(r, c):
            twin_ax = getattr(axes[r, c], "_twin_ax_bottom", None)
            if twin_ax is not None:
                # Marks as well as numbers: with the columns squeezed
                # together, bare ticks on C's right edge sit next to D's
                # left edge and read as D's tick marks.
                twin_ax.tick_params(axis="y", labelright=False, right=False)

    # Panel letters row-major across all panels.
    for ax, letter in zip(axes.flat, CONFIG.panel_letters):
        ax.annotate(letter, xytext=(0, 0), textcoords="offset points",
                    **CONFIG.panel_letter_kwargs)

    if CONFIG.title:
        fig.suptitle(CONFIG.title, y=1.00, fontsize=12)
    fig.tight_layout()
    if CONFIG.subplots_wspace is not None:
        # After tight_layout, otherwise it recomputes the gap from the
        # tick labels we just switched off.
        fig.subplots_adjust(wspace=CONFIG.subplots_wspace)
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
