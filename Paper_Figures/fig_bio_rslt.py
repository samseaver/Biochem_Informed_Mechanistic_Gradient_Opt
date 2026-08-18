#!/usr/bin/env python3
"""
Quick combined view: all biomass-component classes on ONE axis per species
(Sorghum top, Poplar bottom). Within each timepoint the 6 classes are grouped,
each with a Control + FeLim bar and its own demand->net-supply whisker. Shared y.
Exploratory ("just to see") companion to figure_carbon_budget_bars.py.

Run: micromamba run -n bf-runtime python preprint_figures/figure_carbon_budget_combined.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["hatch.linewidth"] = 0.6
import matplotlib.pyplot as plt
import carbon_flow as cf

SPECIES = ["Sorghum", "Poplar"]
CLASSES = cf.BIOMASS_CLASS_ORDER
CC = {"Amino acids": "#d95f0e", "Organic acids": "#8c564b", "FA": "#756bb1",
      "Nucleotides": "#31a354", "Cell wall": "#c51b8a", "Sugars": "#2c7fb8"}


BIO1_LABEL = "Biomass total (bio1)"
BIO1_COLOR = "#333333"


def _series(run, cl, t):
    """(demand, supply) carbon arrays over timepoints for one class+treatment.
    cl == BIO1_LABEL => total carbon into biomass (sum over all classes)."""
    dem, sup = [], []
    for tp in cf.TIMEPOINTS:
        cond = f"Leaf_{t}_{tp}"
        d = cf.biomass_carbon_by_class(run, cond)
        s = cf.biomass_carbon_supply_by_class(run, cond)
        if cl == BIO1_LABEL:
            dem.append(sum(d.values())); sup.append(sum(s.values()))
        else:
            dem.append(d.get(cl, 0.0)); sup.append(s.get(cl, 0.0))
    return np.array(dem), np.array(sup)


def _panel(ax, run):
    x = np.arange(len(cf.TIMEPOINTS))
    groups = [BIO1_LABEL] + CLASSES                   # bio1 total first, then classes
    nbar = len(groups) * len(cf.TREATMENTS)           # 14 sub-bars per timepoint
    span = 0.85                                        # <1 leaves a gap between days
    slot_w = span / nbar
    bw = slot_w                                        # bars flush within a day
    for gi, cl in enumerate(groups):
        col = BIO1_COLOR if cl == BIO1_LABEL else CC[cl]
        for ti, t in enumerate(cf.TREATMENTS):
            slot = gi * len(cf.TREATMENTS) + ti
            xpos = x + (slot - (nbar - 1) / 2) * slot_w
            dem, sup = _series(run, cl, t)
            # outlined bars (no fill) so whiskers -- incl. negative -- stay visible;
            # Control = plain solid outline, FeLim = wide diagonal hatch (transparent)
            ax.bar(xpos, dem, bw, facecolor="none", edgecolor=col, lw=1.3,
                   hatch=("//" if t != "Control" else None), zorder=3)
            # no whisker on the bio1 total (it just compounds the class-level slack)
            if cl == BIO1_LABEL:
                continue
            for xi, d, s in zip(xpos, dem, sup):
                ax.plot([xi, xi], [d, s], color="black", lw=0.5, zorder=5)
                ax.plot([xi - bw * 0.32, xi + bw * 0.32], [s, s], color="black",
                        lw=0.7, zorder=5)
    ax.axhline(0, color="#bbb", lw=0.5, zorder=0)
    for i in range(len(cf.TIMEPOINTS) - 1):            # dotted separator between days
        ax.axvline(i + 0.5, color="#999", ls=":", lw=0.7, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(cf.TIMEPOINTS)
    ax.set_xlim(-0.5, len(cf.TIMEPOINTS) - 0.5)        # tight edge margin (= half day-gap)
    ax.tick_params(labelsize=8)


def main():
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    # Default is the adopted operating point. A bare invocation used to build
    # the svp_1.0 figure, which is NOT what the manuscript includes.
    svp = sys.argv[1] if len(sys.argv) > 1 else "svp_2.0"
    ptag = svp.replace("svp_", "p")
    fig, axs = plt.subplots(len(SPECIES), 1, figsize=(13, 7), sharey=True,
                            squeeze=False)
    for i, sp in enumerate(SPECIES):
        ax = axs[i][0]
        try:
            run = cf.Run(sp, svp=svp)
        except Exception as e:
            ax.text(0.5, 0.5, f"{sp}\n(run pending)", ha="center", va="center",
                    transform=ax.transAxes, color="#999"); ax.axis("off")
            print(f"[skip] {sp}: {e}"); continue
        _panel(ax, run)
        ax.set_ylabel(f"{sp}\ncarbon (bio1 flux × C)", fontsize=9)
    # Legend laid out column-major into a 3-row x 5-col grid (row-major fill with
    # invisible spacers): col1 Biomass | col2/3 the six classes | col4 Control/FeLim
    # | col5 whisker.
    def blank():
        return Patch(facecolor="none", edgecolor="none", label="")
    hBio = Patch(facecolor="none", edgecolor=BIO1_COLOR, lw=1.3, label="Biomass")
    hCls = [Patch(facecolor="none", edgecolor=CC[c], lw=1.3, label=c) for c in CLASSES]
    hCtl = Patch(facecolor="none", edgecolor="#555", lw=1.3, label="Control")
    hFe = Patch(facecolor="none", edgecolor="#555", lw=1.3, hatch="//", label="FeLim")
    hWsk = Line2D([0], [0], color="black", lw=1.0, label="whisker = ‖SV‖ slack")
    # matplotlib legend fills column-major, so list each column top->bottom:
    # col1 Biomass | col2 AA/Org/FA | col3 Nuc/Cell/Sug | col4 Control/FeLim | col5 whisker
    order = [hBio,    blank(), blank(),
             hCls[0], hCls[1], hCls[2],
             hCls[3], hCls[4], hCls[5],
             hCtl,    hFe,     blank(),
             hWsk,    blank(), blank()]
    fig.legend(handles=order, ncol=5, fontsize=8, loc="lower center",
               frameon=False, bbox_to_anchor=(0.5, -0.04),
               handletextpad=0.5, columnspacing=1.4)
    # No figure-level title: the manuscript caption carries this information
    # (bars = bio1 demand, whiskers = network supply, mass-balance penalty p).
    # rect top returns to 1.0 now that no suptitle needs the reserved strip.
    fig.tight_layout(rect=[0, 0.06, 1, 1.0])
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, f"fig_bio_rslt_{ptag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[OK] {out}")
    # The manuscript includes the un-suffixed name. Write it directly at the
    # operating point instead of relying on a manual copy, which is how the
    # p-tagged and manuscript copies drifted apart before.
    if svp == "svp_2.0":
        canon = os.path.join(here, "fig_bio_rslt.png")
        fig.savefig(canon, dpi=150, bbox_inches="tight")
        print(f"[OK] {canon}  (manuscript copy)")
    plt.close(fig)


if __name__ == "__main__":
    main()
