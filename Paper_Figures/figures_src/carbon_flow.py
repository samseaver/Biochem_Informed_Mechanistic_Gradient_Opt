#!/usr/bin/env python3
"""
Shared carbon-flow analysis for the "biomass does not dictate flux" section.

For a given run (species + suffix), this module quantifies, per condition:

  1. Calvin carbon influx      - net CO2 fixation (CO2 exchange + RuBisCO).
  2. Precursor throughput      - net production of each biomass precursor by the
                                 (transcript-constrained) network, grouped by the
                                 precursor's chemical class.
  3. Biomass drain             - bio1 flux (the passive mass-balance sink).
  4. Mass-balance slack        - ||S.v|| (the soft-constraint residual that lets
                                 biomass run without balanced precursor supply).

and, per reaction, an ATTRIBUTION of whether its flux is transcript-driven
(tracks its V_bf target) or flexbio-enabled (rides the flexible-biomass headroom
above its transcript target), used to explain why the inter-precursor flows vary.

All figures/tables in the carbon-flow bundle import from here so the numbers are
computed one way. Reuses:
  - find_limiting_genes.pool_ratio_trajectory / load_molar (pool contraction)
  - PlantSEED_Roles.json subsystems (reaction -> pathway)
Run standalone for a validation dump:
  micromamba run -n bf-runtime python svp_analysis/carbon_flow.py Sorghum nocap-Bcomp
"""
from __future__ import annotations
import os, re, sys, json, glob, collections
import numpy as np
import pandas as pd
import cobra

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJ = f"{_REPO}/Biochem_Informed_Mechanistic_Gradient_Opt/projects"

# CLAUDE 2026-08-12: project-directory resolution now lives in figures_src so
# carbon_flow and the fig_ml_rslt deck cannot drift onto different runs. See
# io_utils.fresh_arm_dir for the arms-260811 layout and its env overrides.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "figures_src"))
from io_utils import fresh_arm_dir            # noqa: E402
import io_utils                              # noqa: E402
# PlantSEED_Roles.json, resolved at call time via BIOFLUX_PLANTSEED_DIR.
# See io_utils.plantseed_roles(): ModelSEED/PlantSEED at tag v2.5.

TIMEPOINTS = ["2d", "4d", "7d", "14d", "21d"]
TREATMENTS = ["Control", "FeLim"]
LEAF_CONDS = [f"Leaf_{t}_{tp}" for tp in TIMEPOINTS for t in TREATMENTS]

CO2 = "cpd00011"          # carbon-fixation substrate
RUBISCO = "rxn00018"      # RuBisCO carboxylase (Calvin entry)

# ---------------------------------------------------------------------------
# base-reaction stripping (matches figure_8 / figure_9 convention)
# ---------------------------------------------------------------------------
def base_rxn(col: str) -> str:
    s = col
    while True:
        n = re.sub(r'_(y\d+|d\d+|c\d+|[frio])$', '', s)
        if n == s:
            return s
        s = n


def _sign(col: str) -> int:
    return -1 if col.endswith(("_r", "_i")) else 1


# ---------------------------------------------------------------------------
# precursor chemical-class map (for grouping the carbon branches)
# ---------------------------------------------------------------------------
ASPARTATE_FAMILY = {"L-Aspartate", "L-Asparagine", "L-Lysine", "L-Threonine",
                    "L-Methionine", "L-Isoleucine", "Oxaloacetate", "L-Malate",
                    "L-Homoserine"}
_AA = {"glycine", "l-alanine", "l-serine", "l-valine", "l-leucine", "l-isoleucine",
       "l-proline", "l-threonine", "l-cysteine", "l-methionine", "l-lysine",
       "l-arginine", "l-histidine", "l-phenylalanine", "l-tyrosine", "l-tryptophan",
       "l-aspartate", "l-asparagine", "l-glutamate", "l-glutamine", "l-homoserine"}


def precursor_group(name: str) -> str:
    n = name.lower()
    if name in ASPARTATE_FAMILY or n in _AA:
        return "Amino acids"
    if any(k in n for k in ("glucose", "fructose", "sucrose", "starch", "mannose",
                            "galactose", "xylose", "arabinose", "udp-", "adp-glucose",
                            "gdp-", "trehalose", "rhamnose", "glucan")):
        return "Sugars / starch"
    if any(k in n for k in ("palmit", "stear", "oleate", "linole", "acyl", "fatty",
                            "acp)", "acyl-carrier", "myrist", "laur")):
        return "Lipids / FA"
    if any(k in n for k in ("atp", "gtp", "ctp", "utp", "amp", "gmp", "cmp", "ump",
                            "datp", "dgtp", "dctp", "dttp", "nucleot", "adenos",
                            "guanos", "cytid", "uridin", "thymid")):
        return "Nucleotides"
    if any(k in n for k in ("coumarate", "ferulate", "cellulose", "pectin", "lignin",
                            "cinnamate", "sinapate", "caffe")):
        return "Cell wall / phenylprop."
    return "Other precursors"


# ---------------------------------------------------------------------------
# Curated biomass-component classification (by ModelSEED cpd id) + carbon counts.
# Reviewed with the PI 2026-08-03. Replaces the fragile name-keyword grouping
# above for the biomass-carbon figures. Carbon counts sourced from the ModelSEED
# Biochemistry compounds table (cached in biomass_component_carbon.json).
# ---------------------------------------------------------------------------
_CARBON = json.load(open(os.path.join(os.path.dirname(__file__),
                                      "biomass_component_carbon.json")))

BIOMASS_CLASS_ORDER = ["Amino acids", "Organic acids", "FA",
                       "Nucleotides", "Cell wall", "Sugars"]
_CLASS_BY_CPD = {
    **{c: "Amino acids" for c in ("cpd00033 cpd00035 cpd00051 cpd00132 cpd00041 "
        "cpd00084 cpd00023 cpd00053 cpd00119 cpd00322 cpd00107 cpd00039 cpd00060 "
        "cpd00066 cpd00129 cpd00054 cpd00161 cpd00065 cpd00069 cpd00156").split()},
    **{c: "Organic acids" for c in ("cpd00130 cpd00032 cpd00137 cpd00331 "
        "cpd00159 cpd00080").split()},
    **{c: "FA" for c in ("cpd00536 cpd00214 cpd01080").split()},
    **{c: "Nucleotides" for c in ("cpd00091 cpd00114").split()},
    **{c: "Cell wall" for c in ("cpd00604 cpd00163").split()},
    **{c: "Sugars" for c in ("cpd19001 cpd19035").split()},
}


def _cpd_of(met_id: str) -> str:
    return re.sub(r'_[a-z]\d+$', '', met_id)


def biomass_component_class(met_id: str) -> str:
    return _CLASS_BY_CPD.get(_cpd_of(met_id), "Other")


def carbon_atoms(met_id: str) -> int:
    return _CARBON.get(_cpd_of(met_id), 0)


def biomass_carbon_by_class(run, cond: str) -> dict:
    """Carbon drawn INTO biomass from each class per unit time =
    bio1_flux * |coef_component| * C_atoms, summed over the curated classes.
    These bars sum exactly to the total carbon entering the bio1 reaction."""
    bf = run.V.loc[cond, "bio1"]
    g = collections.defaultdict(float)
    for m, coef in run.precursors.items():
        g[biomass_component_class(m.id)] += bf * (-coef) * carbon_atoms(m.id)
    return dict(g)


def biomass_carbon_supply_by_class(run, cond: str) -> dict:
    """Carbon actually SUPPLIED to each biomass-component class by the rest of
    the network (all non-bio1 reactions), in the same units as
    biomass_carbon_by_class: net_production(component) * C_atoms, summed per
    class. Because mass balance is soft (‖SV‖ != 0), this need not equal the
    bio1 demand; the signed gap demand - supply == -(SV)*C is the mass-balance
    slack per class (positive = under-supplied 'phantom' carbon, negative =
    over-supplied carbon that bio1 absorbs as a sink)."""
    g = collections.defaultdict(float)
    for m, _coef in run.precursors.items():
        g[biomass_component_class(m.id)] += run.met_supply(m, cond) * carbon_atoms(m.id)
    return dict(g)


def biomass_carbon_slack_by_class(run, cond: str) -> dict:
    """Signed mass-balance slack per class = demand - net supply (carbon units).
    Positive => class is under-supplied (bar overstates real supply)."""
    dem = biomass_carbon_by_class(run, cond)
    sup = biomass_carbon_supply_by_class(run, cond)
    return {cl: dem.get(cl, 0.0) - sup.get(cl, 0.0)
            for cl in set(dem) | set(sup)}


# ---------------------------------------------------------------------------
# subsystem (pathway) map from PlantSEED
# ---------------------------------------------------------------------------
_SUBSYS = None
def subsystem_map() -> dict:
    global _SUBSYS
    if _SUBSYS is None:
        m = collections.defaultdict(set)
        for r in json.load(open(io_utils.plantseed_roles())):
            for rxn in r.get("reactions", []):
                for s in r.get("subsystems", []):
                    m[base_rxn(rxn)].add(s)
        _SUBSYS = {k: sorted(v) for k, v in m.items()}
    return _SUBSYS


def role_map() -> dict:
    rmap = {}
    for r in json.load(open(io_utils.plantseed_roles())):
        for rxn in r.get("reactions", []):
            rmap.setdefault(base_rxn(rxn), r.get("role", ""))
    return rmap


# ---------------------------------------------------------------------------
# run loader
# ---------------------------------------------------------------------------
class Run:
    # Default is the adopted operating point (p = 2.0). Callers that want a
    # different arm must say so explicitly; several figure scripts construct
    # Run() without an svp and used to silently get p = 1.0.
    def __init__(self, species: str, suffix: str = "", svp: str = "svp_2.0"):
        self.species = species
        self.suffix = suffix
        sp = species.lower()
        # `suffix` selected a scratch variant of the old per-species project
        # dir; under the arms layout each svp is already its own directory, so
        # the resolver takes over unless a suffix is explicitly asked for.
        if suffix:
            self.dir = f"{_PROJ}/qpsi-260406-plastid-{sp}-{suffix}"
        else:
            self.dir = fresh_arm_dir(species.capitalize(),
                                     svp.replace("svp_", ""))
            if self.dir is None or not os.path.isdir(self.dir):
                raise FileNotFoundError(
                    f"no project directory for {species} {svp} at {self.dir}")
        self.model = cobra.io.read_sbml_model(glob.glob(f"{self.dir}/inputs/*_dup.xml")[0])
        self.V = pd.read_csv(f"{self.dir}/ml/{svp}/results/startVbfandZero_noRelu_V_headers.tsv",
                             sep="\t", index_col=0)
        self.V.columns = self.V.columns.str.strip()
        z = np.load(f"{self.dir}/ml/training/training.npz", allow_pickle=True)
        self.S = np.asarray(z["S"], float)
        self.reactions = [str(x) for x in z["reactions"]]
        # V_bf targets (per split reaction) and FVA ceilings
        vbf = pd.read_csv(f"{self.dir}/integration_results/vbf.tsv", sep="\t", index_col=0)
        self.vbf = vbf
        fva = pd.read_csv(f"{self.dir}/integration_results/fva.tsv", sep="\t", index_col=0)
        self.fva_max = fva["max"].to_dict()
        self.bio = self.model.reactions.get_by_id("bio1")
        self.precursors = {m: c for m, c in self.bio.metabolites.items() if c < 0}

    # --- flux helpers ---
    def net(self, base: str, cond: str) -> float:
        return sum(_sign(c) * self.V.loc[cond, c]
                   for c in self.V.columns if base_rxn(c) == base)

    def met_supply(self, met, cond: str) -> float:
        """Net production of a metabolite by all non-bio1 reactions."""
        return sum(self.V.loc[cond, r.id] * r.metabolites[met]
                   for r in met.reactions if r.id != "bio1" and r.id in self.V.columns)

    def sv_norm(self, cond: str) -> float:
        v = np.array([self.V.loc[cond, r] if r in self.V.columns else 0.0
                      for r in self.reactions])
        return float(np.linalg.norm(self.S @ v))

    def total_flux(self, cond: str) -> float:
        return float(self.V.loc[cond, [c for c in self.V.columns]].abs().sum())


# ---------------------------------------------------------------------------
# analyses
# ---------------------------------------------------------------------------
def carbon_budget(run: Run, cond: str) -> dict:
    """Whole-condition carbon accounting."""
    # CO2 influx: net across CO2 exchange columns (uptake positive)
    co2_cols = [c for c in run.V.columns if CO2 in c and ("EX_" in c or "_e0" in c)]
    # exchange _i = inward (uptake); report uptake as a positive influx
    co2_in = -sum(_sign(c) * run.V.loc[cond, c] for c in co2_cols)
    rubisco = run.net(RUBISCO, cond)
    over = 0.0
    demand = 0.0
    supplied = 0.0
    for m, coef in run.precursors.items():
        d = run.V.loc[cond, "bio1"] * (-coef)
        s = run.met_supply(m, cond)
        demand += d
        supplied += min(max(s, 0.0), d)
        if s > 0:
            over += s
    return dict(condition=cond,
                co2_influx=float(co2_in), rubisco=float(rubisco),
                bio1=float(run.V.loc[cond, "bio1"]),
                precursor_overproduction=float(over),
                biomass_demand=float(demand),
                pct_supported=float(100 * supplied / demand) if demand > 1e-9 else float("nan"),
                sv_slack=run.sv_norm(cond),
                total_flux=run.total_flux(cond))


def precursor_supply(run: Run, cond: str) -> pd.DataFrame:
    """Per biomass precursor: demand vs real pathway supply, with class."""
    rows = []
    bio1 = run.V.loc[cond, "bio1"]
    for m, coef in run.precursors.items():
        d = bio1 * (-coef)
        s = run.met_supply(m, cond)
        pct = 100 * s / d if d > 1e-9 else np.nan
        if s < -1e-3:
            cls = "net-consumed"
        elif d > 1e-9 and s > 1.25 * d:
            cls = "over-produced"
        elif d > 1e-9 and s >= 0.8 * d:
            cls = "supplied"
        elif s > 1e-3:
            cls = "partial"
        else:
            cls = "inactive"
        rows.append(dict(component=m.name, group=precursor_group(m.name),
                         aspartate_family=(m.name in ASPARTATE_FAMILY),
                         demand=float(d), supply=float(s), pct_supported=float(pct),
                         cls=cls))
    return pd.DataFrame(rows)


def carbon_by_group(run: Run, cond: str) -> dict:
    """Net precursor production aggregated by chemical class (Sankey branch widths)."""
    g = collections.defaultdict(float)
    for m in run.precursors:
        s = run.met_supply(m, cond)
        if s > 0:
            g[precursor_group(m.name)] += s
    return dict(g)


def _vbf_traj(run: Run, split_col: str) -> np.ndarray:
    """V_bf target trajectory across the 10 Leaf conditions for one split reaction."""
    out = np.full(len(LEAF_CONDS), np.nan)
    if split_col not in run.vbf.index:
        return out
    for i, cond in enumerate(LEAF_CONDS):
        col = f"vbf_{cond}"
        if col in run.vbf.columns:
            out[i] = run.vbf.loc[split_col, col]
    return out


def _pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def attribution(run: Run, corr_thr: float = 0.6, flux_thr: float = 1.0) -> pd.DataFrame:
    """Per split reaction: is its flux transcript-driven (tracks V_bf target) or
    flexbio-enabled (carries flux above/uncoupled from its transcript target)?"""
    smap = subsystem_map()
    rmap = role_map()
    rows = []
    for col in run.V.columns:
        if col == "bio1" or col.startswith("EX_"):
            continue
        vtraj = np.array([run.V.loc[c, col] for c in LEAF_CONDS])
        maxv = np.nanmax(np.abs(vtraj))
        if maxv < flux_thr:
            cls = "slack/inactive"
            corr = np.nan
        else:
            bt = _vbf_traj(run, col)
            has_target = np.isfinite(bt).any() and np.nanmax(bt) > 0
            corr = _pearson(vtraj, bt)
            if not has_target:
                cls = "no-target/transport"       # no GPR -> no transcript target at all
            elif np.isfinite(corr) and corr >= corr_thr:
                cls = "transcript-driven"          # flux tracks its transcript target
            else:
                cls = "mass-balance-driven"        # flux set by stoichiometric coupling
                                                   # within the flex-widened bounds, not its target
        b = base_rxn(col)
        subs = smap.get(b, [])
        rows.append(dict(split=col, base=b, role=rmap.get(b, ""),
                         subsystem=subs[0] if subs else "(unassigned)",
                         max_flux=float(maxv), corr_vbf=corr, cls=cls))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# validation dump
# ---------------------------------------------------------------------------
def _dump(species: str, suffix: str):
    run = Run(species, suffix)
    print(f"\n===== {species} / {suffix} =====")
    print("condition            co2_in  rubisco   bio1  overprod  %supp  ||SV||  totflux")
    for tp in TIMEPOINTS:
        for t in TREATMENTS:
            b = carbon_budget(run, f"Leaf_{t}_{tp}")
            print(f"  {b['condition']:18s} {b['co2_influx']:6.1f} {b['rubisco']:7.1f} "
                  f"{b['bio1']:6.2f} {b['precursor_overproduction']:8.2f} "
                  f"{b['pct_supported']:5.0f} {b['sv_slack']:6.1f} {b['total_flux']:8.0f}")
    att = attribution(run)
    print("  attribution counts:",
          att["cls"].value_counts().to_dict())
    print("  carbon by group @Control_7d:",
          {k: round(v, 1) for k, v in carbon_by_group(run, "Leaf_Control_7d").items()})


if __name__ == "__main__":
    sp = sys.argv[1] if len(sys.argv) > 1 else "Sorghum"
    suf = sys.argv[2] if len(sys.argv) > 2 else "nocap-Bcomp"
    _dump(sp, suf)
