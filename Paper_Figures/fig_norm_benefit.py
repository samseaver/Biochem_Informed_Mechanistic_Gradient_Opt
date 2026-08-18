#!/usr/bin/env python3
"""
What pool normalization BUYS (why we normalize, despite the transcript-coupling cost).

(A) Cross-species comparability: the two species' plastid proteome pools differ
    ~1.7x, so RAW reaction scores carry a species-scale offset - Sorghum looks
    systematically busier. Normalization (score / pool) removes it, putting C4
    Sorghum and C3 Poplar on a common 'share of proteome' scale.

(B) Prioritization under contraction: as the pool contracts under FeLim, a reaction
    that merely holds its raw transcript is GAINING share. The normalized
    FeLim/Control ratio (share change) by subsystem reveals what the plant protects
    while it downsizes - biology the raw score cannot show.

Reads the reaction-scoring molar fractions (transcript side only; independent of the
trained flux). Emits figure_normalization_benefit.png/.svg and tables/*.

Run: micromamba run -n bf-runtime python preprint_figures/figure_normalization_benefit.py
"""
import os, sys, collections, json, re, textwrap
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_src"))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from find_limiting_genes import load_molar
import carbon_flow as cf
import io_utils

SPECIES = ["Sorghum", "Poplar"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
SPC = {"Sorghum": "#d95f0e", "Poplar": "#2c7fb8"}


# Minimum share of the total Control reaction score a subsystem must hold to be
# eligible for panel B's reallocation ranking. Without a floor the ranking is
# dominated by subsystems holding 0.01-0.03% of the pool (proline metabolism,
# CDP-diacylglycerol biosynthesis), whose FeLim/Control ratios are noise.
# CLAUDE 2026-08-15: 0.5 -> 0.45. Methionine_and_cysteine_metabolism holds
# 0.499% of the pool, so a 0.5 cut excluded it by one thousandth of a point --
# even though it ranks 9th by the reallocation magnitude panel B actually plots
# (Sulfur_metabolism, which is shown, ranks 14th), and the Results contrast the
# two directly (sulfur 0.48 down, Met/Cys 1.65 up). Effect of the change is
# contained: it adds that one row and displaces Chlorophyll_Biosynthesis from
# topB, which still appears via topA. 19 rows -> 20.
MIN_SHARE_PCT = 0.45

_AA3 = {
    "alanine": "Ala", "arginine": "Arg", "asparagine": "Asn", "aspartate": "Asp",
    "cysteine": "Cys", "glutamate": "Glu", "glutamine": "Gln", "glycine": "Gly",
    "histidine": "His", "isoleucine": "Ile", "leucine": "Leu", "lysine": "Lys",
    "methionine": "Met", "phenylalanine": "Phe", "proline": "Pro", "serine": "Ser",
    "threonine": "Thr", "tryptophan": "Trp", "tyrosine": "Tyr", "valine": "Val",
}
# \b keeps 'Homomethionine' and 'Beta alanine' from being mangled into
# 'HomoMet' / 'Beta Ala' -- only free-standing amino acid names are replaced.
_AA_RE = re.compile(r"\b(" + "|".join(sorted(_AA3, key=len, reverse=True)) + r")\b", re.I)


def abstract_enzyme_map():
    """subsystem -> abstract_enzyme name, for subsystems that are a single enzyme.

    PlantSEED tags every role with the enzyme complex it belongs to. Where a
    whole subsystem resolves to exactly one such complex, that complex name is a
    cleaner, more precise axis label than the subsystem name: 'F0F1-type ATP
    synthase (plastidial)' becomes 'Plastidial ATP synthase' and 'Cytochrome
    b6-f complex (plastidial)' becomes 'Cytochrome b6-f'. Multi-enzyme
    subsystems (pathways) keep their own name and are absent from this map.
    """
    by_sub = collections.defaultdict(set)
    for r in json.load(open(io_utils.plantseed_roles())):
        ae = r.get("abstract_enzyme")
        if not ae:
            continue
        for s in r.get("subsystems", []):
            by_sub[s].add(ae)
    return {s: next(iter(a)) for s, a in by_sub.items() if len(a) == 1}


_AEMAP = None

# The y-label column competes with the two panels for width, and at the font
# size needed for a half-page figure the longest names cost ~0.5in apiece.
# These two are the only ones long enough to matter; both abbreviations are
# conventional and are stated in the caption.
_TRIM = {
    "Photorespiration (oxidative C2 cycle)": "Photorespiration (C2 cycle)",
    "Branched-chain amino acid metabolism": "BCAA metabolism",
}


def _short(s, width=38):
    """Readable axis label: single-enzyme name, amino acids abbreviated, wrapped.

    The original version truncated at 30 chars, which severed five of the top
    rows mid-word ("Pyrimidine de novo biosynthesi"). Wrap instead, so every
    name is legible in full, and shorten honestly rather than by cutting:
    single-enzyme subsystems take their abstract_enzyme name, and spelled-out
    amino acids collapse to their three-letter codes.
    """
    global _AEMAP
    if _AEMAP is None:
        _AEMAP = abstract_enzyme_map()
    s = s or "(unassigned)"
    s = _AEMAP.get(s, s)
    s = s.replace("_", " ").replace("(plastidial)", "").strip()
    s = _AA_RE.sub(lambda m: _AA3[m.group(1).lower()], s)
    s = _TRIM.get(s, s)
    return "\n".join(textwrap.wrap(s, width=width)) or s


def primary_subsystem(base, smap):
    """Pick one subsystem label for a base reaction.

    PlantSEED annotates some reactions with several compartment variants of the
    same subsystem, and the map stores them sorted, so a bare subs[0] returns
    whichever sorts first. For rxn08173 (ATP synthase) that is the
    '(mitochondrial)' name even though this is a plastid-scoped model -- which
    put a mitochondrial ATP synthase row in a plastid figure. The model is
    plastidial, so prefer the plastidial variant when the reaction carries one.
    Subsystems with no plastidial sibling (e.g. Quinol oxidases) keep their own
    name.
    """
    subs = smap.get(base, [])
    if not subs:
        return "(unassigned)"
    plastidial = [s for s in subs if "(plastidial)" in s]
    return plastidial[0] if plastidial else subs[0]


def per_base(sp):
    """mean raw + relative score per base reaction, for Control and FeLim."""
    m = load_molar(sp)
    g = (m.groupby(["base_rxn", "treatment"])[["reaction_score", "relative_reaction_score"]]
         .mean().reset_index())
    return g


def subsystem_of(base, smap):
    return primary_subsystem(base, smap)


def main():
    os.makedirs(OUT, exist_ok=True)
    smap = cf.subsystem_map()

    # ---------- (A) cross-species allocation (only meaningful after normalization) ----------
    # Raw scores have no 'fraction of proteome' meaning and sum to different totals per
    # species (pools differ ~1.7x). Converting to proteome SHARE makes the C4-vs-C3
    # pathway allocation directly comparable. Show per-subsystem share (%).
    d = {sp: per_base(sp) for sp in SPECIES}
    ctl = {sp: d[sp][d[sp].treatment == "Control"].set_index("base_rxn") for sp in SPECIES}
    alloc = {}
    for sp in SPECIES:
        c = ctl[sp]
        by = collections.defaultdict(float)
        for b in c.index:
            by[primary_subsystem(b, smap)] += c.loc[b, "relative_reaction_score"]
        tot = sum(by.values()) or 1.0
        alloc[sp] = {k: 100 * v / tot for k, v in by.items()}
    # Build panel-B data (subsystem FeLim/Control ratios) up front, so both panels
    # can share the UNION of the subsystems each would otherwise select on its own.
    rows = []
    for sp in SPECIES:
        g = d[sp].pivot_table(index="base_rxn", columns="treatment",
                              values=["reaction_score", "relative_reaction_score"])
        g = g.dropna()
        for b, r in g.iterrows():
            sub = primary_subsystem(b, smap)
            rows.append(dict(species=sp, base=b, subsystem=sub,
                             raw_c=r[("reaction_score", "Control")], raw_f=r[("reaction_score", "FeLim")],
                             rel_c=r[("relative_reaction_score", "Control")],
                             rel_f=r[("relative_reaction_score", "FeLim")]))
    df = pd.DataFrame(rows)

    # Each panel nominates its own top-12 and BOTH panels show the union, ordered by
    # combined allocation so the rows align across A and B.
    #
    # A nominates by proteome-share allocation (Control) -- what it plots.
    # B must nominate by what IT plots: the magnitude of the FeLim/Control share
    # change. It previously nominated by summed raw score, which is allocation
    # before the per-species pool division -- so it ranked subsystems in the same
    # order as A, the union collapsed to 12, and the strongest reallocators were
    # never shown. Sulfur metabolism is the clearest casualty: rank 6 by
    # reallocation (Sorghum halves its share, 0.50, while Poplar holds at 0.98)
    # but rank 13 by size, so it fell just outside on both of the old criteria.
    topA_cand = sorted(set(alloc["Sorghum"]) | set(alloc["Poplar"]),
                       key=lambda s: -(alloc["Sorghum"].get(s, 0) + alloc["Poplar"].get(s, 0)))[:12]

    sz = df.groupby("subsystem")["raw_c"].sum()
    share_pct = 100 * sz / sz.sum()
    ratio = (df.groupby(["subsystem", "species"])[["rel_c", "rel_f"]].sum()
               .assign(r=lambda x: x.rel_f / x.rel_c)["r"].unstack("species"))
    # largest share change in either species, among subsystems big enough to trust
    maxdev = (ratio - 1.0).abs().max(axis=1, skipna=True)
    eligible = maxdev[share_pct.reindex(maxdev.index).fillna(0) >= MIN_SHARE_PCT]
    topB_cand = eligible.sort_values(ascending=False).head(12).index.tolist()

    subsystems = sorted(set(topA_cand) | set(topB_cand),
                        key=lambda s: -(alloc["Sorghum"].get(s, 0) + alloc["Poplar"].get(s, 0)))
    print(f"[rows] {len(subsystems)} = {len(topA_cand)} allocation "
          f"+ {len(topB_cand)} reallocation, union")
    for s in subsystems:
        tag = ("A" if s in topA_cand else " ") + ("B" if s in topB_cand else " ")
        print(f"   [{tag}] {s}")
    shared = sorted(set(ctl["Sorghum"].index) & set(ctl["Poplar"].index))

    # Font sizing is set by how far the canvas is scaled down on the page. The
    # manuscript renders this at \linewidth = 469pt = 6.5in, so a label drawn at
    # FS points lands at FS * 6.5/FIG_W on paper. At the old 12in width an 11pt
    # label arrived at ~6pt, which is unreadable.
    FIG_W = 13.5
    FS = 18                       # y labels: ~8.7pt on the page
    FS_TICK, FS_AX, FS_LEG = 16, 17, 16

    # Panels sit side by side, not stacked: the 19 rows are drawn once instead of
    # twice, which is what lets the figure fit a half page. Stacked, the same rows
    # at this font size run past a full page.
    fig = plt.figure(figsize=(FIG_W, 1.5 + 0.46 * len(subsystems)))
    gs = fig.add_gridspec(1, 2, wspace=0.06)

    # ---------- (A) cross-species allocation ----------
    axA = fig.add_subplot(gs[0, 0])
    yA = np.arange(len(subsystems)); hA = 0.38
    for i, sp in enumerate(SPECIES):
        axA.barh(yA + (0.5 - i) * hA, [alloc[sp].get(s, 0) for s in subsystems], hA,
                 color=SPC[sp], edgecolor="white", lw=0.3, label=sp)
    axA.set_yticks(yA); axA.set_yticklabels([_short(s) for s in subsystems], fontsize=FS)
    axA.invert_yaxis()
    axA.set_xlabel("normalized-score allocation, Control", fontsize=FS_AX)
    # Bottom-left in both panels: the top-right of A is occupied by the Calvin bar and
    # the bottom rows hold almost none of the pool, so the corner is empty in A; in B
    # the bottom rows all gain share, so their bars run right of the x=1 line.
    axA.text(0.015, 0.012, "A", transform=axA.transAxes, ha="left", va="bottom",
             fontsize=FS + 8, fontweight="bold")
    axA.legend(fontsize=FS_LEG, loc="lower right"); axA.tick_params(labelsize=FS_TICK)

    # ---------- (B) prioritization under contraction ----------
    # per-subsystem FeLim/Control ratio of NORMALIZED score (share change), score-weighted
    top = subsystems
    # sharey keeps the two panels locked to the same rows; invert_yaxis on axA
    # therefore flips both, and must not be repeated here.
    ax = fig.add_subplot(gs[0, 1], sharey=axA)
    y = np.arange(len(top)); h = 0.38
    for i, sp in enumerate(SPECIES):
        share = []
        for s in top:
            sub = df[(df.species == sp) & (df.subsystem == s)]
            # score-weighted share change = sum(rel_f)/sum(rel_c)
            share.append(sub["rel_f"].sum() / sub["rel_c"].sum() if sub["rel_c"].sum() > 0 else np.nan)
        # Bars are drawn in log2 of the ratio, so that a halving and a doubling
        # draw the same length in opposite directions. On a linear ratio axis a
        # 2x loss can only reach 0.5 while a 2x gain reaches 2.0, which visually
        # understates every deprioritized subsystem -- including the sulfur bar
        # the text now leans on. Tick labels are relabeled back to ratios below.
        ax.barh(y + (0.5 - i) * h, np.log2(np.array(share, dtype=float)), h,
                color=SPC[sp], edgecolor="white", lw=0.3, label=sp)
    ax.axvline(0.0, color="k", lw=1)
    # Round ratios rather than exact reciprocal pairs: 0.7/1.4 are only approximate
    # reciprocals (1/0.7 = 1.43), but they read far more cleanly than 0.625/1.6 and
    # the bar lengths, not the ticks, carry the symmetry.
    rt = [0.5, 0.7, 1.0, 1.4, 2.0]
    ax.set_xticks(np.log2(rt))
    ax.set_xticklabels([f"{v:g}" for v in rt])
    # Autoscale puts the extreme bar (Sorghum sulfur, 0.50) flush against the frame,
    # where its end is indistinguishable from a clipped bar. Pad both sides explicitly.
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo - 0.05, hi + 0.05)
    plt.setp(ax.get_yticklabels(), visible=False)   # rows are labelled once, on A
    # One line only; the parenthetical gloss (share change, >1 gain / <1 loss) lives
    # in the caption, where there is room for the full wording.
    ax.set_xlabel("normalized-score FeLim / Control, $\\log_2$ scale", fontsize=FS_AX)
    ax.text(0.015, 0.012, "B", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=FS + 8, fontweight="bold")
    ax.tick_params(labelsize=FS_TICK)

    fig.tight_layout(rect=[0, 0, 1, 1])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_norm_benefit.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    # table
    df.round(4).to_csv(os.path.join(OUT, "normalization_prioritization.tsv"), sep="\t", index=False)
    print(f"[OK] {out}")
    print(f"cross-species median Sb/Pt: raw = {np.median([ctl['Sorghum'].loc[b,'reaction_score']/ctl['Poplar'].loc[b,'reaction_score'] for b in shared if ctl['Poplar'].loc[b,'reaction_score']>0 and ctl['Sorghum'].loc[b,'reaction_score']>0]):.2f}"
          f" ; normalized = {np.median([ctl['Sorghum'].loc[b,'relative_reaction_score']/ctl['Poplar'].loc[b,'relative_reaction_score'] for b in shared if ctl['Poplar'].loc[b,'relative_reaction_score']>0 and ctl['Sorghum'].loc[b,'relative_reaction_score']>0]):.2f}")


if __name__ == "__main__":
    main()
