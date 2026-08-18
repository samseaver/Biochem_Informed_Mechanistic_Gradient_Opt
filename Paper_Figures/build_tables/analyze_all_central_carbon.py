#!/usr/bin/env python
"""Per-reaction cross-svp + peak_robust analysis for ALL central-carbon
subsystems (PlantSEED 'Central Carbon' class minus transports), plus the
photosynthetic ETC + ferredoxin reductases that were previously curated.

Same robustness filter as the iron-enzyme analysis. Output: per-reaction
classification across both species, focusing on which reactions in
Pentose phosphate / Glycolysis / TCA / Rubisco shunt / Sucrose / Starch
show defensible FeLim responses.
"""
import os, json, re, collections
import numpy as np, pandas as pd


def _final_step(ckpt_dir):
    """Highest V_step_N.tsv number in ``ckpt_dir``.

    Overrides the stale hardcoded step numbers in RUNS below — early
    stopping triggers at different steps each run, so we detect the
    actual final step from disk.
    """
    files = os.listdir(ckpt_dir)
    steps = [int(f[len("V_step_"):-len(".tsv")])
             for f in files
             if f.startswith("V_step_") and f.endswith(".tsv")]
    if not steps:
        raise RuntimeError(f"No V_step_*.tsv checkpoints in {ckpt_dir}")
    return max(steps)

# --- path setup (build_tables/ lives one level below Paper_Figures/) --------
import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_sys.path.insert(0, os.path.join(_HERE, "..", "figures_src"))
import io_utils                                              # noqa: E402

ROLES_JSON = io_utils.plantseed_roles()
# ==== DATA SOURCE TOGGLE (see analyze_both_species.py for context) =
DATA_SOURCE = "fresh"

if DATA_SOURCE == "fresh":
    OUT = io_utils.cross_species_dir()
    # CLAUDE 2026-08-12: the publication runs moved to the arms-260811 sweep,
    # where every (species, svp, min_delta) cell is its own project directory.
    # Path resolution now lives in figures_src/io_utils so this script, the
    # fig_ml_rslt deck and carbon_flow cannot drift onto different runs; the
    # step numbers below are ignored anyway (_final_step reads them off disk).
    # Set BIOFLUX_LEGACY_LAYOUT=1 to reproduce the pre-260811 outputs.
    from io_utils import fresh_arm_dir, fresh_training_npz   # noqa: E402

    RUNS = {(sp, s): (f"{fresh_arm_dir(sp, s)}/ml/svp_{s}/checkpoints", 0)
            for sp in ("Poplar", "Sorghum")
            for s in ("0.1", "0.5", "1.0", "2.0")}
    NPZ = {sp: fresh_training_npz(sp, "2.0") for sp in ("Poplar", "Sorghum")}
elif DATA_SOURCE == "archive":
    OUT = "cross_species_analysis"
    RUNS = {
        ("Poplar",  "0.1"): ("ext_svp01/projects/qpsi-260406-plastid-poplar/ml/checkpoints", 214900),
        ("Poplar",  "0.5"): ("ext_svp05/projects/qpsi-260406-plastid-poplar/ml/checkpoints", 124300),
        ("Poplar",  "1.0"): ("ext_svp1/projects/qpsi-260406-plastid-poplar/ml/checkpoints",  110600),
        ("Poplar",  "2.0"): ("ext_svp2/projects/qpsi-260406-plastid-poplar/ml/checkpoints",  112400),
        ("Sorghum", "0.1"): ("ext_sorghum_svp01/projects/qpsi-260406-plastid-sorghum/ml/checkpoints", 214300),
        ("Sorghum", "0.5"): ("ext_sorghum_svp05/projects/qpsi-260406-plastid-sorghum/ml/checkpoints", 118700),
        ("Sorghum", "1.0"): ("ext_sorghum_svp1/projects/qpsi-260406-plastid-sorghum/ml/checkpoints",  103000),
        ("Sorghum", "2.0"): ("ext_sorghum_svp2/projects/qpsi-260406-plastid-sorghum/ml/checkpoints",  99500),
    }
    NPZ = {
        "Poplar":  "ext_svp05/projects/qpsi-260406-plastid-poplar/ml/training/training.npz",
        "Sorghum": "ext_sorghum_svp05/projects/qpsi-260406-plastid-sorghum/ml/training/training.npz",
    }
else:
    raise ValueError(f"DATA_SOURCE must be 'fresh' or 'archive', got {DATA_SOURCE!r}")
os.makedirs(OUT, exist_ok=True)

# Expanded subsystem list — all PlantSEED Central Carbon class non-transport,
# plus the photo ETC complexes already analyzed (kept for completeness)
EXPANDED_SUBSYSTEMS = {
    # Photo (already covered, kept for cross-check)
    "Calvin-Benson-Bassham_cycle":             "Calvin",
    "Photorespiration_(oxidative_C2_cycle)":   "Photorespiration",
    "Photosystem_II":                          "PSII",
    "Cytochrome_b6-f_complex_(plastidial)":    "Cyt_b6f",
    "Photosystem_I":                           "PSI",
    "F0F1-type_ATP_synthase_(plastidial)":     "ATP_synthase_pl",
    # NEW: central-carbon high-flux
    "Pentose_phosphate_pathway":               "PPP",
    "Glycolysis_and_Gluconeogenesis":          "Glycolysis_GNG",
    "TCA_cycle":                               "TCA",
    "Rubisco_shunt":                           "Rubisco_shunt",
    "Sucrose_metabolism":                      "Sucrose",
    "Starch_biosynthesis":                     "Starch_bio",
    "Starch_degradation":                      "Starch_deg",
    "Acetyl-CoA_biosynthesis":                 "AcetylCoA_bio",
}
EXTRA_REACTIONS = {
    "rxn17196": ("Ferredoxin_reductases", "FNR"),
    "rxn19701": ("Ferredoxin_reductases", "Fd-nitrite reductase"),
}
COMPARTMENT_RE = re.compile(r"_[a-z]+\d+$")
TP_ORDER = ["1h","2d","4d","7d","14d","21d"]
SVP_LIST = ["0.1","0.5","1.0","2.0"]
REL_THRESH = 0.05
ADJ_THRESH = 0.30

def split_id(rxn_id):
    s = rxn_id; d = None
    for suf in ("_f","_r"):
        if s.endswith(suf): d = suf[1:]; s = s[:-2]; break
    m = COMPARTMENT_RE.search(s)
    if m: s = s[:m.start()]
    return s, d

# ---- Build curated reaction set ----
roles = json.load(open(ROLES_JSON))
target = {}
for r in roles:
    if not r.get("include", True): continue
    for ss in r.get("subsystems", []):
        if ss in EXPANDED_SUBSYSTEMS:
            for rxn in r.get("reactions", []):
                target.setdefault(rxn, {"group": EXPANDED_SUBSYSTEMS[ss], "role": r["role"]})
for rxn, (grp, name) in EXTRA_REACTIONS.items():
    target[rxn] = {"group": grp, "role": name}
print(f"Curated reactions across {len(set(m['group'] for m in target.values()))} groups: {len(target)}")
group_counts = collections.Counter(m["group"] for m in target.values())
for g, n in sorted(group_counts.items()):
    print(f"  {g}: {n}")

# ---- Extract per-(species, svp, condition, base_rxn) flux + adjacent SV residual ----
flux_rows = []
for species in ["Poplar","Sorghum"]:
    npz = np.load(NPZ[species], allow_pickle=True)
    reactions = [str(x) for x in npz["reactions"]]
    treatments = [str(x) for x in npz["treatments"]]
    S = np.asarray(npz["S"], dtype=float)
    # Map base->cols and base->adj metabolites
    base_cols = collections.defaultdict(lambda: {"f":[], "r":[], "none":[]})
    base_neigh = {}
    for j, rid in enumerate(reactions):
        b, d = split_id(rid)
        if b not in target: continue
        key = d if d in ("f","r") else "none"
        base_cols[b][key].append(j)
        base_neigh.setdefault(b, set()).update(np.where(S[:, j] != 0)[0].tolist())
    for (sp, svp), (ck, step) in RUNS.items():
        if sp != species: continue
        step = _final_step(ck)   # override stale hardcoded step
        V = np.loadtxt(os.path.join(ck, f"V_step_{step}.tsv"), delimiter="\t")
        SV = V @ S.T
        absSV = np.abs(SV)
        throughput = np.abs(V) @ np.abs(S.T)
        rel_resid = absSV / np.maximum(throughput, 1e-30)
        for base, meta in target.items():
            cols = base_cols.get(base)
            if cols is None: continue
            for ci, cond in enumerate(treatments):
                vf = float(V[ci, cols["f"]].sum()) if cols["f"] else 0.0
                vr = float(V[ci, cols["r"]].sum()) if cols["r"] else 0.0
                vn = float(V[ci, cols["none"]].sum()) if cols["none"] else 0.0
                neigh = base_neigh.get(base, set())
                adj_max = float(rel_resid[ci, list(neigh)].max()) if neigh else 0.0
                flux_rows.append({
                    "species": sp, "svp": svp, "condition": cond,
                    "tissue": cond.split("_")[0], "treatment": cond.split("_")[1],
                    "timepoint": cond.split("_")[2] if cond.count("_") >= 2 else "",
                    "group": meta["group"], "base_rxn": base, "role": meta["role"],
                    "V_net": (vf - vr) + vn, "adj_max_resid": adj_max,
                })
flux_df = pd.DataFrame(flux_rows)
flux_df.to_csv(f"{OUT}/all_central_carbon_flux.tsv", sep="\t", index=False, float_format="%.6g")
print(f"\nWrote per-condition flux table: {len(flux_df)} rows")

# ---- Build per-timepoint per-(svp,base,species,tissue) delta table ----
# Match Control_T with FeLim_T per tissue
tc_rows = []
for (sp, svp, tissue, base), g in flux_df.groupby(["species","svp","tissue","base_rxn"]):
    tps = sorted(set(g["timepoint"]),
                 key=lambda t: TP_ORDER.index(t) if t in TP_ORDER else 99)
    for t in tps:
        ctl = g[(g.treatment=="Control") & (g.timepoint==t)]
        fel = g[(g.treatment=="FeLim") & (g.timepoint==t)]
        if ctl.empty or fel.empty: continue
        vc = abs(float(ctl["V_net"].iloc[0]))
        vf = abs(float(fel["V_net"].iloc[0]))
        adj = max(float(ctl["adj_max_resid"].iloc[0]), float(fel["adj_max_resid"].iloc[0]))
        tc_rows.append({"species": sp, "svp": svp, "tissue": tissue, "base_rxn": base,
                        "group": g["group"].iloc[0], "role": g["role"].iloc[0],
                        "timepoint": t, "abs_ctl": vc, "abs_fel": vf,
                        "delta": vf - vc, "ratio": vf/vc if vc > 1e-4 else np.nan,
                        "adj_max_resid": adj})
tc = pd.DataFrame(tc_rows)
tc.to_csv(f"{OUT}/all_central_carbon_timecourse.tsv", sep="\t", index=False, float_format="%.6g")

# ---- Peak_robust classification ----
def classify(g):
    qualifying = []
    best_score = 0.0
    peak_tp = peak_dir = peak_log2 = peak_delta = peak_adj = None
    # Also compute full-trajectory robustness (svp_dependent / sign_flip)
    deltas_all = g["delta"].values
    ratios_all = g["ratio"].dropna().values
    sign_flip = (any(d>1e-4 for d in deltas_all) and any(d<-1e-4 for d in deltas_all))
    if len(ratios_all) >= 2:
        factor = max(ratios_all)/min(ratios_all) if min(ratios_all)>0 else np.inf
        rng = max(ratios_all)-min(ratios_all)
        svp_dep = (factor > 2.0 or rng > 0.5)
    else:
        svp_dep = False
    max_adj_overall = g["adj_max_resid"].max()
    mean_ctl_overall = g["abs_ctl"].mean()
    mean_fel_overall = g["abs_fel"].mean()
    low_flux = (mean_ctl_overall < 0.01 and mean_fel_overall < 0.01)
    in_imb = (max_adj_overall > ADJ_THRESH)
    for tp in TP_ORDER:
        gtp = g[g.timepoint==tp]
        if len(gtp) < len(SVP_LIST): continue
        d = gtp["delta"].values
        if not (all(x>0 for x in d) or all(x<0 for x in d)):
            continue
        mean_ctl = float(gtp["abs_ctl"].mean())
        mean_fel = float(gtp["abs_fel"].mean())
        if mean_ctl < 0.01 and mean_fel < 0.01: continue
        denom = max(mean_ctl, mean_fel)
        mean_delta = float(np.mean(d))
        rel = abs(mean_delta)/denom if denom > 0 else 0
        if rel < REL_THRESH: continue
        max_adj_tp = float(gtp["adj_max_resid"].max())
        if max_adj_tp > ADJ_THRESH: continue
        log2r = float(np.log2(mean_fel/mean_ctl)) if mean_ctl > 1e-4 else np.nan
        qualifying.append((tp, rel, mean_delta, log2r))
        if rel > best_score:
            best_score = rel
            peak_tp, peak_dir, peak_log2, peak_delta, peak_adj = (
                tp, "+" if mean_delta>0 else "-", log2r, mean_delta, max_adj_tp)
    # Tier
    if sign_flip and not low_flux and not in_imb and len(qualifying) == 0:
        tier = "DROP_sign_flip"
    elif in_imb:
        tier = "DROP_imbalanced"
    elif low_flux:
        tier = "DROP_low_flux"
    elif not sign_flip and not svp_dep and not low_flux and not in_imb and len(qualifying) > 0:
        tier = "ROBUST"
    elif not sign_flip and svp_dep and not low_flux and not in_imb:
        tier = "PLAUSIBLE_svp_dependent"
    elif len(qualifying) > 0:
        tier = "PEAK_ROBUST"
    else:
        tier = "DROP_other"
    return pd.Series({
        "mean_abs_ctl": float(mean_ctl_overall), "mean_abs_fel": float(mean_fel_overall),
        "max_adj_overall": float(max_adj_overall),
        "flag_sign_flip": bool(sign_flip), "flag_svp_dependent": bool(svp_dep),
        "flag_in_imbalanced": bool(in_imb), "flag_low_flux": bool(low_flux),
        "peak_robust_any_tp": len(qualifying) > 0,
        "peak_tp": peak_tp, "peak_dir": peak_dir,
        "peak_log2_ratio": peak_log2, "peak_rel_mag": best_score if best_score > 0 else np.nan,
        "n_qualifying_tps": len(qualifying),
        "qualifying_tps": ",".join(q[0] for q in qualifying),
        "tier": tier,
    })

cls = tc.groupby(["species","tissue","group","base_rxn","role"]).apply(classify, include_groups=False).reset_index()
cls.to_csv(f"{OUT}/all_central_carbon_classification.tsv", sep="\t", index=False, float_format="%.4g")

# ---- Report ----
print("\n" + "="*92)
print("Tier distribution per group (Leaf only)")
print("="*92)
leaf = cls[cls["tissue"]=="Leaf"]
dist = leaf.groupby(["species","group","tier"]).size().unstack(fill_value=0)
print(dist.to_string())

# Newly-defensible (ROBUST/PLAUSIBLE/PEAK_ROBUST) reactions NOT in photo+iron groups
new_groups = {"PPP","Glycolysis_GNG","TCA","Rubisco_shunt","Sucrose","Starch_bio","Starch_deg","AcetylCoA_bio"}
new_keep = leaf[(leaf["group"].isin(new_groups)) &
                (leaf["tier"].isin(["ROBUST","PLAUSIBLE_svp_dependent","PEAK_ROBUST"]))]
print(f"\n--- Newly defensible reactions in non-photo, non-iron subsystems (Leaf) ---")
print(f"Total: {len(new_keep)}")
disp_cols = ["species","group","base_rxn","role","tier","mean_abs_ctl","mean_abs_fel",
             "peak_tp","peak_dir","peak_log2_ratio","peak_rel_mag","max_adj_overall"]
disp = new_keep[disp_cols].copy()
for c in ["mean_abs_ctl","mean_abs_fel","peak_log2_ratio","peak_rel_mag","max_adj_overall"]:
    disp[c] = pd.to_numeric(disp[c], errors="coerce").round(3)
print(disp.sort_values(["group","base_rxn","species"]).to_string(index=False))

# Cross-species concordance for new groups
print(f"\n--- Cross-species concordance for new groups (Leaf), keep tiers only ---")
piv = new_keep.pivot_table(index=["group","base_rxn","role"], columns="species",
                            values=["tier","peak_dir","peak_log2_ratio","peak_tp"],
                            aggfunc="first")
piv.columns = [f"{a}|{b}" for a,b in piv.columns]
piv = piv.reset_index()
print(piv.to_string(index=False))
