#!/usr/bin/env python
"""Cross-species (Poplar, Sorghum) cross-svp ({0.1, 0.5, 1.0, 2.0}) analysis
covering both the curated photo/carbon subset and the supplied iron-binding
enzyme list. Applies the cross-svp stability filter to discriminate real
biology from gradient-descent artifact for low-flux iron enzymes.
"""
import os, json, collections, re
import numpy as np, pandas as pd


def _final_step(ckpt_dir):
    """Highest V_step_N.tsv number in ``ckpt_dir``.

    Preferred over hardcoded step numbers in RUNS below — early stopping
    triggers at different step counts across runs, so the "final" step
    is only known after the sweep finishes.
    """
    files = os.listdir(ckpt_dir)
    steps = [int(f[len("V_step_"):-len(".tsv")])
             for f in files
             if f.startswith("V_step_") and f.endswith(".tsv")]
    if not steps:
        raise RuntimeError(f"No V_step_*.tsv checkpoints in {ckpt_dir}")
    return max(steps)

# --- path setup (analysis_scripts/ lives one level below Paper_Figures/) --------
import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_sys.path.insert(0, os.path.join(_HERE, "..", "figures_src"))
import io_utils                                              # noqa: E402

ROLES_JSON = io_utils.plantseed_roles()
IRON_CSV   = os.path.join(_HERE, "iron_binding_reactions.csv")

# ==== DATA SOURCE TOGGLE ===========================================
# "fresh"  : read the LaTeX-aligned per-svp checkpoints under
#            ../Biochem_*/projects/<spc>/ml/svp_<svp>/  and write to
#            cross_species_analysis_fresh/  (current default).
# "archive": read the legacy ext_* extracted checkpoints and write to
#            cross_species_analysis/  (preserved for rubisco_shunt
#            fresh-vs-archive comparison).
DATA_SOURCE = "fresh"

if DATA_SOURCE == "fresh":
    OUT = io_utils.cross_species_dir()
    # CLAUDE 2026-08-14: this script was MISSED in the 2026-08-12 migration to
    # io_utils path resolution. It went on reading the superseded qpsi-260406
    # sweep (Aug 3-10) with hardcoded paths while its sibling
    # analyze_all_central_carbon.py read arms-260812, which put Figure 6 panel F
    # -- the only manuscript figure fed by curated_flux_both_species.tsv -- on
    # different data from every other panel. Resolution now matches the sibling
    # exactly, so the two cannot drift apart again. The step numbers that used
    # to be hardcoded here are ignored regardless: _final_step reads them off
    # disk.
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

SUBSYSTEM_LABEL = {
    "Calvin-Benson-Bassham_cycle":             "Calvin",
    "Photorespiration_(oxidative_C2_cycle)":   "Photorespiration",
    "Photosystem_II":                          "PSII",
    "Cytochrome_b6-f_complex_(plastidial)":    "Cyt_b6f",
    "Photosystem_I":                           "PSI",
    "F0F1-type_ATP_synthase_(plastidial)":     "ATP_synthase_pl",
}
EXTRA_REACTIONS = {
    "rxn17196": ("Ferredoxin_reductases", "FNR"),
    "rxn19701": ("Ferredoxin_reductases", "Fd-nitrite reductase"),
}
COMPARTMENT_RE = re.compile(r"_[a-z]+\d+$")

def split_id(rxn_id):
    """Strip _f/_r and any ModelSEED compartment suffix."""
    stem = rxn_id; direction = None
    for d in ("_f", "_r"):
        if stem.endswith(d):
            direction = d[1:]; stem = stem[:-2]; break
    m = COMPARTMENT_RE.search(stem)
    if m: stem = stem[:m.start()]
    return stem, direction

# ---- Load curated subset (photosynthesis groups) ----
roles = json.load(open(ROLES_JSON))
curated = {}
for r in roles:
    if not r.get("include", True): continue
    for ss in r.get("subsystems", []):
        if ss in SUBSYSTEM_LABEL:
            for rxn in r.get("reactions", []):
                curated.setdefault(rxn, {"group": SUBSYSTEM_LABEL[ss], "role": r["role"]})
for rxn, (grp, name) in EXTRA_REACTIONS.items():
    curated[rxn] = {"group": grp, "role": name}

# ---- Load iron-binding enzyme list ----
iron_df = pd.read_csv(IRON_CSV)
iron_df["base_rxn"] = iron_df["reaction_id"].apply(lambda s: split_id(s)[0])
iron_meta = (iron_df.groupby("base_rxn")
                    .agg(role=("reaction_name","first"),
                         cofactors=("raw_cofactors","first"),
                         n_genes=("gene_id","nunique"))
                    .reset_index())
iron_base_ids = set(iron_meta["base_rxn"])
print(f"Iron enzymes: {len(iron_base_ids)} unique base reactions")

# ---- Process every (species, svp) run ----
flux_rows, iron_flux_rows, sv_per_rxn = [], [], []
network_resid_rows = []

for species in ["Poplar", "Sorghum"]:
    npz = np.load(NPZ[species], allow_pickle=True)
    reactions = [str(x) for x in npz["reactions"]]
    treatments = [str(x) for x in npz["treatments"]]
    S = np.asarray(npz["S"], dtype=float)
    n_mets = S.shape[0]
    print(f"\n{species}: {len(reactions)} reactions, {n_mets} metabolites, {len(treatments)} conditions")

    # Per-base-id column index map (curated + iron)
    base_cols = collections.defaultdict(lambda: {"f": [], "r": [], "none": []})
    base_to_metabolite_neighbors = {}
    # CLAUDE 2026-08-14: build these for EVERY base reaction, not just the
    # curated and iron subsets. fig_ml_rslt panel E needs the per-reaction
    # adjacent-metabolite residual network-wide; the curated and iron loops
    # below index by a specific base id, so a larger map is harmless to them.
    for j, rid in enumerate(reactions):
        base, dirn = split_id(rid)
        key = dirn if dirn in ("f","r") else "none"
        base_cols[base][key].append(j)
        base_to_metabolite_neighbors.setdefault(base, set()).update(np.where(S[:, j] != 0)[0].tolist())

    for (sp, svp), (ck, step) in RUNS.items():
        if sp != species: continue
        step = _final_step(ck)   # override stale hardcoded step
        V = np.loadtxt(os.path.join(ck, f"V_step_{step}.tsv"), delimiter="\t")
        if V.shape != (len(treatments), len(reactions)):
            print(f"  WARN {sp} svp={svp}: V {V.shape} != ({len(treatments)},{len(reactions)})")
            continue
        # SV residuals
        SV = V @ S.T
        absSV = np.abs(SV)
        throughput = np.abs(V) @ np.abs(S.T)
        rel_resid = absSV / np.maximum(throughput, 1e-30)
        # Per (svp x condition x metabolite) — kept per-reaction below

        # Process curated reactions
        for base, meta in curated.items():
            cols = base_cols.get(base)
            if cols is None: continue
            for ci, cond in enumerate(treatments):
                vf = float(V[ci, cols["f"]].sum()) if cols["f"] else 0.0
                vr = float(V[ci, cols["r"]].sum()) if cols["r"] else 0.0
                vn = float(V[ci, cols["none"]].sum()) if cols["none"] else 0.0
                flux_rows.append({
                    "species": sp, "svp": svp, "condition": cond,
                    "tissue": cond.split("_")[0], "treatment": cond.split("_")[1],
                    "group": meta["group"], "base_rxn": base, "role": meta["role"],
                    "V_net": (vf - vr) + vn,
                })

        # Process iron reactions
        for base in iron_base_ids:
            cols = base_cols.get(base)
            if cols is None: continue  # not in this species' model
            for ci, cond in enumerate(treatments):
                vf = float(V[ci, cols["f"]].sum()) if cols["f"] else 0.0
                vr = float(V[ci, cols["r"]].sum()) if cols["r"] else 0.0
                vn = float(V[ci, cols["none"]].sum()) if cols["none"] else 0.0
                vnet = (vf - vr) + vn
                # adjacent-metabolite SV residual (worst across this reaction's neighbors)
                neighbors = base_to_metabolite_neighbors.get(base, set())
                if neighbors:
                    adj_rels = rel_resid[ci, list(neighbors)]
                    adj_max  = float(adj_rels.max())
                    adj_med  = float(np.median(adj_rels))
                else:
                    adj_max = adj_med = 0.0
                iron_flux_rows.append({
                    "species": sp, "svp": svp, "condition": cond,
                    "tissue": cond.split("_")[0], "treatment": cond.split("_")[1],
                    "base_rxn": base,
                    "V_f": vf, "V_r": vr, "V_net": vnet,
                    "adj_metabolite_max_rel_residual": adj_max,
                    "adj_metabolite_median_rel_residual": adj_med,
                })

        # CLAUDE 2026-08-14: network-wide per-reaction adjacent-metabolite
        # residual, for fig_ml_rslt panel E. Same metric as the iron block
        # above (worst adjacent metabolite |SV| / that metabolite's total
        # throughput, bounded in [0,1] by the triangle inequality), but over
        # every base reaction rather than the 36 iron-binding ones.
        for base, cols in base_cols.items():
            nb = list(base_to_metabolite_neighbors.get(base, ()))
            if not nb:
                continue
            for ci, cond in enumerate(treatments):
                vf = float(V[ci, cols["f"]].sum()) if cols["f"] else 0.0
                vr = float(V[ci, cols["r"]].sum()) if cols["r"] else 0.0
                vn = float(V[ci, cols["none"]].sum()) if cols["none"] else 0.0
                adj = rel_resid[ci, nb]
                network_resid_rows.append({
                    "species": sp, "svp": svp, "condition": cond,
                    "tissue": cond.split("_")[0], "treatment": cond.split("_")[1],
                    "base_rxn": base, "V_net": (vf - vr) + vn,
                    "adj_max_resid": float(adj.max()),
                    "adj_median_resid": float(np.median(adj)),
                })

flux_df = pd.DataFrame(flux_rows)
iron_df_out = pd.DataFrame(iron_flux_rows)
network_resid_df = pd.DataFrame(network_resid_rows)
network_resid_df.to_csv(f"{OUT}/network_resid_both_species.tsv", sep="\t",
                        index=False, float_format="%.6g")
print(f"Wrote network_resid_both_species.tsv ({len(network_resid_df)} rows, "
      f"{network_resid_df['base_rxn'].nunique()} base reactions)")

flux_df.to_csv(f"{OUT}/curated_flux_both_species.tsv", sep="\t", index=False, float_format="%.6g")
iron_df_out.to_csv(f"{OUT}/iron_flux_both_species.tsv", sep="\t", index=False, float_format="%.6g")
print(f"\nWrote curated_flux_both_species.tsv ({len(flux_df)} rows)")
print(f"Wrote iron_flux_both_species.tsv ({len(iron_df_out)} rows)")

# ---- Iron-enzyme summary: cross-svp FeLim/Control ratio + artifact flag ----
# Compute per (species, svp, tissue, base_rxn) mean V_net under Control vs FeLim
iron_df_out["abs_V_net"] = np.abs(iron_df_out["V_net"])
sub = iron_df_out.copy()
ctl = sub[sub["treatment"]=="Control"].groupby(
    ["species","svp","tissue","base_rxn"]).agg(
        V_ctl=("V_net","mean"),
        absV_ctl=("abs_V_net","mean"),
        adj_resid_ctl=("adj_metabolite_max_rel_residual","mean")).reset_index()
fel = sub[sub["treatment"]=="FeLim"].groupby(
    ["species","svp","tissue","base_rxn"]).agg(
        V_fel=("V_net","mean"),
        absV_fel=("abs_V_net","mean"),
        adj_resid_fel=("adj_metabolite_max_rel_residual","mean")).reset_index()
m = ctl.merge(fel, on=["species","svp","tissue","base_rxn"])
m["FeLim_minus_Control"] = m["V_fel"] - m["V_ctl"]
m["abs_log_ratio"] = np.where(
    (np.abs(m["absV_ctl"]) > 1e-4) & (np.abs(m["absV_fel"]) > 1e-4),
    np.log10(m["absV_fel"] / m["absV_ctl"]),
    np.nan,
)
m["FeLim_over_Control"] = np.where(
    np.abs(m["absV_ctl"]) > 1e-4, m["absV_fel"]/m["absV_ctl"], np.nan
)
m["adj_resid"] = m[["adj_resid_ctl","adj_resid_fel"]].max(axis=1)
m.to_csv(f"{OUT}/iron_ratios_per_run.tsv", sep="\t", index=False, float_format="%.5g")

# Now collapse across svp values: per (species, tissue, base_rxn) compute
# the range of FeLim/Control ratio across the 4 svp values
def classify(group):
    """Return a dict summarizing cross-svp stability for one species/tissue/reaction."""
    ratios = group["FeLim_over_Control"].dropna().to_list()
    diffs  = group["FeLim_minus_Control"].to_list()
    abs_ctl = group["absV_ctl"].mean()
    abs_fel = group["absV_fel"].mean()
    adj    = group["adj_resid"].max()
    if not ratios:
        return pd.Series({
            "median_ratio": np.nan,
            "ratio_range":  np.nan,
            "ratio_minmax_factor": np.nan,
            "median_abs_diff": np.median(np.abs(diffs)) if diffs else np.nan,
            "mean_absV_ctl": abs_ctl, "mean_absV_fel": abs_fel,
            "worst_adj_residual": adj,
            "flag": "no_meaningful_flux",
            "n_svp_observed": 0,
        })
    med = float(np.median(ratios))
    rng = float(max(ratios) - min(ratios))
    factor = float(max(ratios)/min(ratios)) if min(ratios) > 0 else np.inf
    flag = "robust"
    # Sign flip across svp
    if any(d > 0 for d in diffs) and any(d < 0 for d in diffs):
        flag = "sign_flip_across_svp"
    elif factor > 2.0 or rng > 0.5:
        flag = "svp_dependent"
    elif adj > 0.30:
        flag = "in_imbalanced_neighborhood"
    elif abs_ctl < 0.01 and abs_fel < 0.01:
        flag = "low_flux_uninterpretable"
    return pd.Series({
        "median_ratio": med,
        "ratio_range":  rng,
        "ratio_minmax_factor": factor,
        "median_abs_diff": float(np.median(np.abs(diffs))),
        "mean_absV_ctl": float(abs_ctl), "mean_absV_fel": float(abs_fel),
        "worst_adj_residual": float(adj),
        "flag": flag,
        "n_svp_observed": len(ratios),
    })

summary = (m.groupby(["species","tissue","base_rxn"])
            .apply(classify, include_groups=False)
            .reset_index())
# Bring in role/cofactors
summary = summary.merge(iron_meta, on="base_rxn", how="left")
summary = summary[[c for c in [
    "species","tissue","base_rxn","role","cofactors",
    "mean_absV_ctl","mean_absV_fel","median_ratio","ratio_range","ratio_minmax_factor",
    "worst_adj_residual","flag","n_svp_observed"] if c in summary.columns]]
summary.to_csv(f"{OUT}/iron_summary.tsv", sep="\t", index=False, float_format="%.4g")
print(f"Wrote iron_summary.tsv ({len(summary)} rows)")
