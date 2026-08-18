#!/usr/bin/env python3
"""
For each of the 10 driver reactions in figure_8 (5 ETC + 5 Calvin
Cycle), identify which INDIVIDUAL GENES:

  (a) define the reaction's protein-complex bottleneck — i.e. sit in
      the `limiting_subunit` column of the reaction-scoring pipeline
      output (`<species>_reaction_molar_fractions.tsv`); and
  (b) whose transcript trajectory best MATCHES the biological signal
      — measured here as Pearson r against
        - the reaction's converged flux delta (FeLim - Control), and
        - the GLK1/GLK2 mean transcript delta (the regulator signal).

The limiting-subunit information comes directly from the scoring
pipeline in `RNASeq-Review/RNASeq_Enzyme_Abundance/src/reactionScoresHelper.py`
which uses SUM within subunits, MIN across subunits (bottleneck),
and SUM across isozymes.

Rank output per (species, reaction): genes sorted by
`combined_score = limiting_pct * max(|r_flux|, |r_glk|)`.

Outputs
-------
- svp_analysis/cross_species_analysis_fresh/limiting_genes_ranking.tsv
- top hits per (species, reaction) printed to stdout

Runs cheap in memory — no model / SBML loading required.
"""
from __future__ import annotations

import os
import sys
import lzma
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RNA  = "/scratch/seaver/Claude_Projects/RNASeq-Review/RNASeq_Enzyme_Abundance/projects/qpsi-plastidial"

MOLAR = {
    "Poplar":  f"{_RNA}/integration_results/Poplar_reaction_molar_fractions.tsv",
    "Sorghum": f"{_RNA}/integration_results/Sorghum_reaction_molar_fractions.tsv",
}
TMM = {
    "Poplar":  f"{_RNA}/rnaseq-data/Poplar_raw_genes_tmm_mean.tsv",
    "Sorghum": f"{_RNA}/rnaseq-data/Sorghum_raw_genes_tmm_mean.tsv.xz",
}
CURATED_FLUX = f"{_REPO}/svp_analysis/cross_species_analysis_fresh/curated_flux_both_species.tsv"
OUT_TSV      = f"{_REPO}/svp_analysis/cross_species_analysis_fresh/limiting_genes_ranking.tsv"

DRIVER_RXNS = {
    "rxn20632": ("ETC",    "PSII"),
    "rxn20595": ("ETC",    "Cyt b6f"),
    "rxn26754": ("ETC",    "PSI"),
    "rxn17196": ("ETC",    "FNR"),
    "rxn08173": ("ETC",    "ATP synthase"),
    "rxn00018": ("Calvin", "RuBisCO"),
    "rxn01100": ("Calvin", "PGK"),
    "rxn00782": ("Calvin", "NADPH-GAPDH"),
    "rxn01111": ("Calvin", "PRK"),
    "rxn01345": ("Calvin", "SBPase"),
}

GLK_ORTHOLOGS = {
    "Poplar":  ["Potri.007G136901", "Potri.017G015800"],
    "Sorghum": ["Sobic.003G002600", "Sobic.010G096300"],
}

TIMEPOINTS = ["2d", "4d", "7d", "14d", "21d"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_molar(species: str) -> pd.DataFrame:
    df = pd.read_csv(MOLAR[species], sep="\t")
    df["base_rxn"] = df["reaction_id"].str.split("_").str[0]
    df = df[df["condition"].str.startswith("Leaf_")].copy()
    parts = df["condition"].str.split("_")
    df["treatment"] = parts.str[1]
    df["timepoint"] = parts.str[2]
    df = df[df["treatment"].isin(["Control", "FeLim"])]
    df = df[df["timepoint"].isin(TIMEPOINTS)]
    return df


def plastid_pool_per_condition(molar: pd.DataFrame) -> dict[str, float]:
    """Recover the total-plastid-proteome denominator that the
    reaction-scoring pipeline used, one value per condition.

    The pipeline stores ``relative_reaction_score = reaction_score / pool``
    where pool = total plastid proteome abundance for that condition.
    Any single reaction lets us invert this: ``pool = reaction_score
    / relative_reaction_score``. Every reaction in a given condition
    gives the same pool (it's a per-condition constant), so we
    average across reactions for numerical robustness.
    """
    m = molar.dropna(subset=["reaction_score", "relative_reaction_score"])
    m = m[m["relative_reaction_score"] > 0]
    m = m.assign(pool=m["reaction_score"] / m["relative_reaction_score"])
    return m.groupby("condition")["pool"].mean().to_dict()


def pool_ratio_trajectory(molar: pd.DataFrame) -> np.ndarray:
    """FeLim / Control plastid-pool ratio per timepoint aligned to
    TIMEPOINTS. Under FeLim this ratio is < 1 (the pool contracts),
    which is exactly the multiplier we need to deflate the observed
    FeLim flux back to a Control-pool-scaled comparison."""
    pool = plastid_pool_per_condition(molar)
    out = np.full(len(TIMEPOINTS), np.nan)
    for i, tp in enumerate(TIMEPOINTS):
        pc = pool.get(f"Leaf_Control_{tp}"); pf = pool.get(f"Leaf_FeLim_{tp}")
        if pc and pf and pc > 0:
            out[i] = pf / pc
    return out


def load_tmm(species: str) -> pd.DataFrame:
    path = TMM[species]
    if path.endswith(".xz"):
        with lzma.open(path, "rt") as fh:
            df = pd.read_csv(fh, sep="\t")
    else:
        df = pd.read_csv(path, sep="\t")
    return df


def gene_delta_trajectory(tmm_df: pd.DataFrame, gene_id: str) -> np.ndarray:
    """FeLim - Control per timepoint for one gene; NaN when missing."""
    sub = tmm_df[tmm_df["Gene_ID"] == gene_id].set_index("condition")["value"]
    diff = np.full(len(TIMEPOINTS), np.nan)
    for i, tp in enumerate(TIMEPOINTS):
        c = sub.get(f"Leaf_Control_{tp}", np.nan)
        f = sub.get(f"Leaf_FeLim_{tp}",   np.nan)
        try:
            diff[i] = float(f) - float(c)
        except (TypeError, ValueError):
            pass
    return diff


def flux_delta_trajectory(cf: pd.DataFrame, species: str, base_rxn: str,
                          svp: str = "1.0", tissue: str = "Leaf",
                          pool_ratios: np.ndarray | None = None
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw_delta, corrected_delta) trajectories per timepoint.

    - raw_delta       = V_FeLim − V_Control (what the model outputs)
    - corrected_delta = V_FeLim × pool_ratio − V_Control
      where pool_ratio(tp) = pool(FeLim, tp) / pool(Control, tp).
      Deflating V_FeLim by pool_ratio removes the normalization-artifact
      inflation that comes from relative_reaction_score = raw/pool
      when the total plastid pool contracts under FeLim.
    """
    s = cf[(cf["species"] == species) & (cf["svp"] == svp) &
           (cf["tissue"] == tissue) & (cf["base_rxn"] == base_rxn)]
    n = len(TIMEPOINTS)
    if s.empty:
        return np.full(n, np.nan), np.full(n, np.nan)
    parts = s["condition"].str.rsplit("_", n=1, expand=True)
    s = s.assign(timepoint=parts[1])
    piv = s.pivot_table(index="timepoint", columns="treatment",
                        values="V_net", aggfunc="first")
    piv = piv.reindex(index=TIMEPOINTS)
    if "Control" not in piv.columns or "FeLim" not in piv.columns:
        return np.full(n, np.nan), np.full(n, np.nan)
    v_c = piv["Control"].values
    v_f = piv["FeLim"].values
    raw_delta = v_f - v_c
    if pool_ratios is None:
        return raw_delta, raw_delta.copy()
    corrected_delta = v_f * pool_ratios - v_c
    return raw_delta, corrected_delta


def glk_delta_trajectory(tmm_df: pd.DataFrame, species: str) -> np.ndarray:
    """Mean FeLim - Control across the two GLK orthologs per timepoint."""
    arrs = []
    for gid in GLK_ORTHOLOGS[species]:
        d = gene_delta_trajectory(tmm_df, gid)
        if np.any(np.isfinite(d)):
            arrs.append(d)
    if not arrs:
        return np.full(len(TIMEPOINTS), np.nan)
    return np.nanmean(np.vstack(arrs), axis=0)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    av, bv = a[mask], b[mask]
    if np.std(av) < 1e-12 or np.std(bv) < 1e-12:
        return np.nan
    return float(np.corrcoef(av, bv)[0, 1])


def rank_species(species: str, cf: pd.DataFrame) -> pd.DataFrame:
    print(f"\n== {species} ==")
    molar = load_molar(species)
    tmm   = load_tmm(species)
    pool_r = pool_ratio_trajectory(molar)
    print(f"  Plastid pool ratio (FeLim/Control) per tp: {np.round(pool_r, 3).tolist()}")
    print(f"  → deflation factors applied to observed V_FeLim before correlating")
    glk_traj = glk_delta_trajectory(tmm, species)
    print(f"  GLK mean delta trajectory: {np.round(glk_traj, 1).tolist()}")

    rows = []
    for base_rxn, (group, label) in DRIVER_RXNS.items():
        rxn_rows = molar[molar["base_rxn"] == base_rxn]
        if rxn_rows.empty:
            print(f"  {label} ({base_rxn}): missing from molar_fractions")
            continue

        gene_conds: dict[str, set[str]] = {}
        for _, row in rxn_rows.iterrows():
            cond = row["condition"]
            lim_str = str(row["limiting_subunit"]) if pd.notna(row["limiting_subunit"]) else ""
            for g in [x.strip() for x in lim_str.split(",") if x.strip()]:
                gene_conds.setdefault(g, set()).add(cond)

        n_total = rxn_rows["condition"].nunique()
        flux_raw, flux_corr = flux_delta_trajectory(
            cf, species, base_rxn, pool_ratios=pool_r
        )

        for gene_id, conds in gene_conds.items():
            limiting_pct = 100.0 * len(conds) / max(n_total, 1)
            gene_diff = gene_delta_trajectory(tmm, gene_id)
            r_flux_raw  = _pearson(gene_diff, flux_raw)
            r_flux_corr = _pearson(gene_diff, flux_corr)
            r_glk       = _pearson(gene_diff, glk_traj)
            # Combined score now uses the POOL-CORRECTED r_flux since
            # that measures the flux we'd expect if the model didn't
            # inflate FeLim Vbf via the shrinking proteome denominator.
            best_r = max(abs(r_flux_corr) if np.isfinite(r_flux_corr) else 0.0,
                         abs(r_glk)       if np.isfinite(r_glk)       else 0.0)
            combined = (limiting_pct / 100.0) * best_r
            rows.append({
                "species":         species,
                "group":           group,
                "reaction":        label,
                "base_rxn":        base_rxn,
                "gene_id":         gene_id,
                "limiting_pct":    round(limiting_pct, 1),
                "r_flux_raw":      round(r_flux_raw,  3) if np.isfinite(r_flux_raw)  else np.nan,
                "r_flux_corr":     round(r_flux_corr, 3) if np.isfinite(r_flux_corr) else np.nan,
                "r_glk":           round(r_glk,       3) if np.isfinite(r_glk)       else np.nan,
                "best_abs_r":      round(best_r, 3),
                "combined":        round(combined, 3),
                **{f"delta_{tp}":  round(float(gene_diff[i]), 2)
                   for i, tp in enumerate(TIMEPOINTS)},
            })
    return pd.DataFrame(rows)


def main() -> None:
    print("Loading curated flux table...")
    cf = pd.read_csv(CURATED_FLUX, sep="\t")
    cf["svp"] = cf["svp"].astype(str)

    all_rows = []
    for sp in ("Poplar", "Sorghum"):
        df = rank_species(sp, cf)
        all_rows.append(df)
    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(["species", "group", "reaction", "combined"],
                          ascending=[True, True, True, False])
    out.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\n[OK] wrote {OUT_TSV}  ({len(out)} gene x reaction rows)")

    # Top gene per (species, reaction), by combined score
    top = (out.sort_values("combined", ascending=False)
              .groupby(["species", "base_rxn"], as_index=False)
              .head(1)
              .sort_values(["species", "group", "reaction"]))
    print("\n=== TOP LIMITING+CORRELATING GENE PER (species, reaction) ===")
    print("r_flux_raw  = corr(gene ΔTMM, observed model ΔV)          — has the pool-inflation artifact")
    print("r_flux_corr = corr(gene ΔTMM, pool-deflated ΔV)           — removes the artifact")
    print("r_glk       = corr(gene ΔTMM, mean GLK ortholog ΔTMM)     — regulator signal")
    print()
    cols = ["species", "group", "reaction", "gene_id", "limiting_pct",
            "r_flux_raw", "r_flux_corr", "r_glk", "combined"]
    with pd.option_context("display.max_colwidth", 22, "display.width", 200):
        print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
