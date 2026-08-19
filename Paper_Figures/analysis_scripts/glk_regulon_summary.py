#!/usr/bin/env python3
"""
Produce a summary table + LaTeX for the paper: the 9 reactions
shared between Poplar's and Sorghum's top-10 flux-driver sets
(i.e. the coordinated GLK-regulon set), each with:

- the rate-limiting subunit gene(s) picked by the reaction-scoring
  pipeline (most-frequent limiting gene per (species, reaction));
- the pool-corrected r_flux (transcript vs deflated flux);
- the correlation of the gene transcript vs GLK1 alone;
- the correlation of the gene transcript vs GLK2 alone.

Correlations against GLK1 and GLK2 are reported SEPARATELY because
Poplar's GLK1 (Potri.007G136901, ~500 TMM baseline) is ~25x more
abundant than Poplar's GLK2 (Potri.017G015800, ~15 TMM baseline).
The "mean GLK" signal used by find_limiting_genes.py is dominated
by whichever paralog has larger absolute delta; splitting them
makes the correlation attributable and comparable across species.

Outputs
-------
- ``<analysis tables>/glk_regulon_shared9.tsv`` (io_utils.cross_species_dir())
- ``<analysis tables>/glk_regulon_shared9.tex``
- stdout: pretty-printed table
"""
from __future__ import annotations

import os
import sys
import lzma
import numpy as np
import pandas as pd

# --- path setup (analysis_scripts/ lives one level below Paper_Figures/) --------
import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_sys.path.insert(0, os.path.join(_HERE, "..", "figures_src"))
import io_utils                                              # noqa: E402

_ANA  = io_utils.cross_species_dir()

_RNA  = io_utils.rnaseq_dir()

TMM = {
    "Poplar":  f"{_RNA}/rnaseq-data/Poplar_raw_genes_tmm_mean.tsv",
    "Sorghum": f"{_RNA}/rnaseq-data/Sorghum_raw_genes_tmm_mean.tsv.xz",
}

# 10 reactions featured in Figure 8 top row (5 ETC + 5 Calvin).
# 9 are shared between both species' top-10 driver sets; PSII is
# Sorghum-only among the top drivers, but its rate-limiting gene is
# computed for Poplar too and belongs in the table so the table
# matches the set of curves plotted in Figure 8 panel A/B.
SHARED9 = {
    "rxn26754": ("ETC",    "PSI"),
    "rxn20632": ("ETC",    "PSII"),
    "rxn08173": ("ETC",    "ATP synthase"),
    "rxn20595": ("ETC",    "Cyt b6f"),
    "rxn17196": ("ETC",    "FNR"),
    "rxn00018": ("Calvin", "RuBisCO"),
    "rxn01100": ("Calvin", "PGK"),
    "rxn00782": ("Calvin", "NADPH-GAPDH"),
    "rxn01111": ("Calvin", "PRK"),
    "rxn01345": ("Calvin", "SBPase"),
}

# Sorghum slots swapped 2026-08-04: GLK1 = the grass "GLK1"-clade gene
# (Sobic.010G096300, clusters with rice Os06g24070). Poplar's two copies are a
# Salicaceae-WGD pair that cannot be split into GLK1/GLK2 (labeled A/B in figures).
GLK_ORTHOLOGS = {
    "Poplar":  {"GLK1": "Potri.007G136901", "GLK2": "Potri.017G015800"},
    "Sorghum": {"GLK1": "Sobic.010G096300", "GLK2": "Sobic.003G002600"},
}

TIMEPOINTS = ["2d", "4d", "7d", "14d", "21d"]

RANKING_TSV = f"{_ANA}/limiting_genes_ranking.tsv"
OUT_TSV     = f"{_ANA}/glk_regulon_shared9.tsv"
OUT_TEX     = f"{_ANA}/glk_regulon_shared9.tex"


def load_tmm(species: str) -> pd.DataFrame:
    p = TMM[species]
    if p.endswith(".xz"):
        with lzma.open(p, "rt") as fh:
            return pd.read_csv(fh, sep="\t")
    return pd.read_csv(p, sep="\t")


def gene_delta(tmm_df: pd.DataFrame, gene_id: str) -> np.ndarray:
    sub = tmm_df[tmm_df["Gene_ID"] == gene_id].set_index("condition")["value"]
    out = np.full(len(TIMEPOINTS), np.nan)
    for i, tp in enumerate(TIMEPOINTS):
        c = sub.get(f"Leaf_Control_{tp}", np.nan)
        f = sub.get(f"Leaf_FeLim_{tp}",   np.nan)
        try:
            out[i] = float(f) - float(c)
        except (TypeError, ValueError):
            pass
    return out


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    av, bv = a[m], b[m]
    if np.std(av) < 1e-12 or np.std(bv) < 1e-12:
        return np.nan
    return float(np.corrcoef(av, bv)[0, 1])


def main() -> None:
    rank = pd.read_csv(RANKING_TSV, sep="\t")
    tmm_by = {sp: load_tmm(sp) for sp in GLK_ORTHOLOGS}
    # Pre-compute per-species GLK1 / GLK2 delta trajectories
    glk_traj = {
        sp: {label: gene_delta(tmm_by[sp], gene)
             for label, gene in GLK_ORTHOLOGS[sp].items()}
        for sp in GLK_ORTHOLOGS
    }

    # Picker: for each (species, reaction), consider every candidate
    # gene in the ranking file, restrict to those consistently in the
    # rate-limiting subunit (limiting_pct >= MIN_LIMITING_PCT), then
    # among those pick the gene with the best POSITIVE r_GLK — the
    # paralog that best fits the GLK regulator, per the user's rule:
    #  * single-subunit reactions (all isoforms SUM'd): every isoform
    #    is trivially "limiting"; pick reduces to "best positive GLK".
    #  * multi-subunit reactions (MIN across subunits): the
    #    limiting_pct filter enforces "must be in the rate-limiting
    #    subunit"; the pick within that set is best positive GLK.
    MIN_LIMITING_PCT = 50.0

    rows = []
    for base_rxn, (group, label) in SHARED9.items():
        for sp in ("Poplar", "Sorghum"):
            candidates = rank[(rank["species"] == sp)
                              & (rank["base_rxn"] == base_rxn)].copy()
            if candidates.empty:
                continue
            candidates = candidates[
                candidates["limiting_pct"].astype(float) >= MIN_LIMITING_PCT
            ]
            if candidates.empty:
                continue

            # Compute per-candidate r_GLK1 / r_GLK2; keep the paralog
            # with the highest max(r_GLK1, r_GLK2). Ties broken by
            # limiting_pct then absolute r as a fallback.
            best_row = None
            best_score = (-np.inf, -np.inf, -np.inf)  # (max_r_glk, limiting_pct, abs_max_r)
            best_r_glk1 = np.nan
            best_r_glk2 = np.nan
            for _, cand in candidates.iterrows():
                gene_id = cand["gene_id"]
                gene_traj = gene_delta(tmm_by[sp], gene_id)
                r_glk1 = pearson(gene_traj, glk_traj[sp]["GLK1"])
                r_glk2 = pearson(gene_traj, glk_traj[sp]["GLK2"])
                max_r = max(r_glk1 if np.isfinite(r_glk1) else -np.inf,
                            r_glk2 if np.isfinite(r_glk2) else -np.inf)
                score = (max_r,
                         float(cand["limiting_pct"]),
                         max(abs(r_glk1) if np.isfinite(r_glk1) else 0.0,
                             abs(r_glk2) if np.isfinite(r_glk2) else 0.0))
                if score > best_score:
                    best_score = score
                    best_row = cand
                    best_r_glk1 = r_glk1
                    best_r_glk2 = r_glk2

            if best_row is None:
                continue
            rows.append({
                "reaction":     label,
                "group":        group,
                "base_rxn":     base_rxn,
                "species":      sp,
                "gene_id":      best_row["gene_id"],
                "limiting_pct": float(best_row["limiting_pct"]),
                "r_flux_corr":  float(best_row["r_flux_corr"]) if pd.notna(best_row["r_flux_corr"]) else np.nan,
                "r_GLK1":       best_r_glk1,
                "r_GLK2":       best_r_glk2,
            })

    df = pd.DataFrame(rows)
    # Order: ETC first, Calvin second; within each, alphabetical by label
    order = list(SHARED9.values())
    df["_ord"] = df["reaction"].map({v[1]: i for i, v in enumerate(SHARED9.items())})
    df = df.sort_values(["_ord", "species"]).drop(columns="_ord")
    df.to_csv(OUT_TSV, sep="\t", index=False, float_format="%.3f")
    print(f"[OK] wrote {OUT_TSV}\n")

    # ---- Pretty print
    with pd.option_context("display.max_colwidth", 22, "display.width", 200,
                            "display.float_format", lambda x: f"{x:+.3f}"):
        print(df.to_string(index=False))

    # ---- LaTeX — condensed layout:
    #   Group column uses \multirow spanning all reactions in the group.
    #   Reaction column uses \multirow spanning its 2 species sub-rows.
    #   Species column dropped — the gene id prefix (Potri vs Sobic)
    #   already tells the species.
    #   Species order within each reaction: Poplar first, Sorghum second.
    #   A short \cmidrule is drawn under Sorghum FNR's r_GLK1 / r_GLK2
    #   cells to flag it as an outlier (negative correlations, opposite
    #   direction from every other row).
    def _fmt(x):
        if pd.isna(x):
            return "--"
        if isinstance(x, float):
            return f"${x:+.3f}$"
        return str(x)

    # Sort rows: group order (ETC, Calvin), reaction order within group
    # (as in SHARED9), species order (Poplar, Sorghum) within reaction.
    group_order = ["ETC", "Calvin"]
    rxn_order = [v[1] for v in SHARED9.values()]
    species_order = ["Poplar", "Sorghum"]
    df["_g_ord"] = df["group"].map({g: i for i, g in enumerate(group_order)})
    df["_r_ord"] = df["reaction"].map({r: i for i, r in enumerate(rxn_order)})
    df["_s_ord"] = df["species"].map({s: i for i, s in enumerate(species_order)})
    df = df.sort_values(["_g_ord", "_r_ord", "_s_ord"]).drop(
        columns=["_g_ord", "_r_ord", "_s_ord"]
    )

    n_per_group = df.groupby("group").size().to_dict()
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Rate-limiting subunit genes for the ten reactions "
        r"featured in Figure~\ref{fig:figure_8} panels A/B: the five "
        r"photosynthetic ETC reactions (PSII, Cyt~b6f, PSI, FNR, "
        r"plastidial ATP synthase) and the five Calvin-cycle reactions "
        r"(RuBisCO, PGK, NADPH-GAPDH, PRK, SBPase). Nine of the ten "
        r"appear as top-10 flux drivers in both \emph{Populus "
        r"trichocarpa} and \emph{Sorghum bicolor}; PSII is a top-10 "
        r"driver only in \emph{Sorghum}, but its \emph{Populus} "
        r"rate-limiting gene is reported for completeness. For each "
        r"reaction the top row names the \emph{Populus} rate-limiting "
        r"gene (\texttt{Potri.\dots}) and the bottom the \emph{Sorghum} "
        r"one (\texttt{Sobic.\dots}); species membership follows from "
        r"the gene identifier. Where the reaction-scoring pipeline "
        r"lists multiple paralogous isoforms within the "
        r"rate-limiting subunit (SUM logic across alternatives), the "
        r"paralog reported is the one whose $\Delta$TMM "
        r"(FeLim$-$Control) trajectory best matches its "
        r"species-specific GLK regulator ortholog (largest positive "
        r"$\max(r_{\text{GLK1}}, r_{\text{GLK2}})$). Pearson "
        r"correlations $r_{\text{GLK1}}$ and $r_{\text{GLK2}}$ are "
        r"computed across the five leaf iron-limitation timepoints "
        r"(2d/4d/7d/14d/21d, $n=5$).}",
        r"\label{tab:glk_regulon_shared9}",
        r"\begin{tabular}{lllrr}",
        r"\toprule",
        r"Group & Reaction & Rate-limiting gene "
        r"& $r_{\text{GLK1}}$ & $r_{\text{GLK2}}$ \\",
        r"\midrule",
    ]

    prev_group = None
    prev_reaction = None
    seen_in_group = 0
    for i, r in df.iterrows():
        # Group cell: opens with \multirow only on the group's first row
        if r["group"] != prev_group:
            grp_cell = rf"\multirow{{{n_per_group[r['group']]}}}{{*}}{{{r['group']}}}"
            prev_group = r["group"]
            seen_in_group = 0
        else:
            grp_cell = ""
        # Reaction cell: opens with \multirow{2} only on the reaction's first row
        if r["reaction"] != prev_reaction:
            rxn_cell = rf"\multirow{{2}}{{*}}{{{r['reaction']}}}"
            prev_reaction = r["reaction"]
        else:
            rxn_cell = ""
        gene_tex = r["gene_id"].replace("_", r"\_")
        lines.append(
            f"{grp_cell} & {rxn_cell} & \\texttt{{{gene_tex}}} "
            f"& {_fmt(r['r_GLK1'])} & {_fmt(r['r_GLK2'])} \\\\"
        )
        seen_in_group += 1
        # Full-width rule at the end of each group (except the last)
        if seen_in_group == n_per_group[r["group"]] and r["group"] != group_order[-1]:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(OUT_TEX, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[OK] wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
