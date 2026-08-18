# Paper_Figures

Everything the manuscript renders, plus the code that produces it.
Assembled 2026-08-07 for the Overleaf/preprint hand-off; moved inside this
repository on 2026-08-18 so the figures and their source ship together.

The manuscript is kept outside this repository and references these files as
`Biochem_Informed_Mechanistic_Gradient_Opt/Paper_Figures/<name>`.

Each generating script is named after the figure it produces, and writes that
same name as its output.

## Figure → source

| Manuscript file | Figure | Generating script |
|---|---|---|
| `fig_method.jpg` | 1 — overall approach | none (hand-drawn diagram) |
| `fig_score_method.png` | 2 — reaction-score logic | none (hand-drawn diagram) |
| `fig_proteome.png` | 3 — plastid vs non-plastid abundance | in the RNASeq repo — see below |
| `fig_scatter_rslt.png` | 4 — objective/relative reaction scores | in the RNASeq repo — see below |
| `fig_ml_detail.png` | — ML workflow schematic | none (hand-drawn diagram) |
| `fig_ml_rslt.png` | — gradient-descent dynamics | `fig_ml_rslt.py` (imports `fig_ml_rslt_base.py`) |
| `fig_bio_rslt.png` | 5 — carbon drawn into biomass | `fig_bio_rslt.py` |
| `fig_photo_etc.png` | 6 — ETC/Calvin + GLK regulon | `fig_photo_etc.py` |
| `fig_norm_benefit.png` | 7 — pool-normalized reallocation | `fig_norm_benefit.py` |

`.svg` siblings are included where a vector version exists. LaTeX builds
against the `.png`/`.jpg` files only, so the four SVGs are not referenced by
the manuscript; they are kept in case a production process wants vector art.

`fig_bio_rslt.py` writes one PNG per penalty as `fig_bio_rslt_p<svp>.png` and,
at the operating point only, a second copy under the un-suffixed name the
manuscript includes. So `fig_bio_rslt_p2.0.png` reappears on every run; it is
regenerated output, not a stray file.

`fig_ml_rslt_base.py` is not a figure generator on its own: it holds the shared
plotting machinery — the config, the data loaders, and panels 1–5 — that
`fig_ml_rslt.py` calls to build the figure. `fig_ml_rslt.py` supplies only the
Leaf-filtered panel 6 and the assembly.

## Figures 3 and 4 live in the RNASeq repo

The two transcript-level figures are produced in
**[samseaver/RNASeq_Enzyme_Abundance](https://github.com/samseaver/RNASeq_Enzyme_Abundance)**,
which is their canonical home — edit them there, not here. Local checkout:
`../../../RNASeq-Review/RNASeq_Enzyme_Abundance`.

| Manuscript file | Script in that repo |
|---|---|
| `fig_proteome.png` | [`figures/plotCombinedPaperFigure.py`](https://github.com/samseaver/RNASeq_Enzyme_Abundance/blob/main/figures/plotCombinedPaperFigure.py) |
| `fig_scatter_rslt.png` | [`figures/generate_reaction_scores_figure.py`](https://github.com/samseaver/RNASeq_Enzyme_Abundance/blob/main/figures/generate_reaction_scores_figure.py) |

`plotCombinedPaperFigure.py` also imports
[`figures/plotAbundanceDistributions.py`](https://github.com/samseaver/RNASeq_Enzyme_Abundance/blob/main/figures/plotAbundanceDistributions.py)
and `generate_reaction_scores_figure.py` for its data loading, so all three are
needed to regenerate figure 3.

Both rendered `.png` files have been copied into that repo's `figures/`
directory under these same names, so the figure and the code that makes it sit
together. The copies here are what LaTeX builds against; if either figure is
regenerated, update both places.

### Figure 4 was re-rendered 2026-08-07 — what changed and why

The published `fig_scatter_rslt.png` had been built from the **capped** reaction
score tables (`uncap_run/scores_cap/`), while the PiNN runs behind figures 5-7
were driven by the **uncapped** tables (`*_reaction_scores.tsv`, md5-identical
to `qpsi-260406-plastid-{poplar,sorghum}/inputs/`). Figure 4 therefore did not
describe the data the model actually saw. This was established by counting the
black-outlined top-5% markers per panel in the published PNG — pixel diffing is
useless here because the render engine changed (old kaleido/orca → kaleido v1 +
Chrome), which shifts every panel by 6-8% on its own.

Switching to the uncapped tables required two changes to
`generate_reaction_scores_figure.py`:

- **Log axes.** The script divided both conditions by a single `global_max`
  across species and days. Uncapped, that maximum is ~12× larger, which pushed
  93% of points below 0.05 and collapsed the figure into its bottom-left
  corner. Reaction scores are log-normal over ~5 decades, so percentile
  clipping cannot rescue a linear axis; both axes are now `log10` with a
  0.1%-quantile floor. Abs rows span 1e-1..1e5, Rel rows 1e-7..1e-1.
- **I-dist in log space.** `(log10(FeLim) - log10(Control)) / sqrt(2)`, replacing
  `(FeLim - Control) / global_max`. This is a scaled log fold change and is
  genuinely independent of score magnitude, which is what the manuscript claims
  of it. The old linear definition was confounded by the denominator.

Colour scale also moved from `icefire` to `RdBu_r` — `icefire` goes black at
both ends, hiding the black top-5% outlines once `cmin/cmax` were tightened onto
the data — and now matches the caption's stated blue-down / red-up convention.

Both the caption's day-7-Sorghum / day-21-Poplar claims survive the change and
are sharper than before. Top-5% counts (2d, 4d, 7d, 14d, 21d):

| Row | Poplar | Sorghum |
|---|---|---|
| Abs (`r_s`) | 1, 1, 0, 4, **27** | 1, 2, **62**, 42, 26 |
| Rel (`r̃_s`) | 1, 0, 0, 3, **21** | 1, 1, **72**, 45, 22 |

**Outstanding:** figure 3 panel B counts "reactions in the 95th percentile from
figure 4", so it should be re-rendered against the same log-space ranking.
`plotCombinedMethodComparison.py` has been updated to match, but
`plotCombinedPaperFigure.py` cannot run in this checkout — `load_classified_tmm`
expects a comma-separated TMM table with `tissue` / `treatment` / `time_stamp`
columns and no such file is present. Panel B's counts are therefore still on the
old linear ranking. The caption claim itself (Sorghum peaks day 7 and stays
elevated; Poplar peaks day 21) holds under both rankings.

## Running the scripts

    micromamba run -n bf-runtime python Paper_Figures/fig_photo_etc.py

The system python has no numpy — use the `bf-runtime` environment.

Helper modules are vendored in `figures_src/` (`io_utils.py`, `style.py`,
`carbon_flow.py`, `find_limiting_genes.py`), so the scripts resolve their
imports from this directory and nothing outside it.

Two inputs are **not** tracked here, and both scripts that need them will stop
with a plain file-not-found if they are absent:

| Input | Default location | Override |
|---|---|---|
| cross-species analysis tables | `Paper_Figures/analysis_tables/` | `BIOFLUX_CROSS_DIR` (the directory itself) |
| measured leaf phenotype data (ICP-MS + reflectance) | `../data/E1.0_Sorghum_Poplar_ICP-MS_Spec_total.txt` | `BIOFLUX_PHENOTYPE_FILE` |
| PlantSEED checkout at tag **v2.5** | `../../PlantSEED` | `BIOFLUX_PLANTSEED_DIR` |
| RNASeq_Enzyme_Abundance project dir at tag **bioflux-preprint-260813** | `../../RNASeq_Enzyme_Abundance/projects/qpsi-plastidial` | `BIOFLUX_RNASEQ_DIR` |

The last two are sibling repositories; see `requirements-runtime.txt`. Neither
is a Python package, so unlike ModelSEEDPy and cobrakbase they cannot be
pip-installed -- clone and point the variable at the checkout. Each raises a
message naming its variable if absent, rather than a bare file-not-found.

`BIOFLUX_CROSS_DIR` names the tables directory directly, which is normally what
you want:

    BIOFLUX_CROSS_DIR=/path/to/cross_species_analysis_fresh \
        micromamba run -n bf-runtime python Paper_Figures/fig_photo_etc.py

`BIOFLUX_DATA_DIR` also exists but names the *parent*, under which the
`analysis_tables/` subdirectory is then looked up.

The analysis tables are derived from **both** species and the Poplar half is
not distributed, so they are not tracked here. Rebuild them with
`build_tables/build_analysis_tables.sh`, which is self-contained: four scripts
in a traced dependency order, about 20 seconds, writing the six tables the
figures read. The phenotype data is measured, not generated by this repository.

### What the RNASeq repository must supply

Same treatment as PlantSEED: a checkout at a fixed tag, pointed at by an env
var. **samseaver/RNASeq_Enzyme_Abundance at tag `bioflux-preprint-260813`**,
with `BIOFLUX_RNASEQ_DIR` set to its `projects/qpsi-plastidial` directory. Four
files are read:

| File | Supplies |
|---|---|
| `integration_results/Sorghum_reaction_molar_fractions.tsv` | gene -> reaction associations |
| `integration_results/Poplar_reaction_molar_fractions.tsv` | same, Poplar |
| `rnaseq-data/Sorghum_raw_genes_tmm_mean.tsv.xz` | per-gene TMM |
| `rnaseq-data/Poplar_raw_genes_tmm_mean.tsv` | same, Poplar |

**Only the two Sorghum files are tracked at that tag.** The Poplar molar
fractions and TMM table are available from the corresponding authors on
request, as the manuscript states, so a public checkout regenerates the Sorghum
panels and not the Poplar ones. That is the same limitation as the run data:
`projects/` here ships Sorghum only.

The gene->reaction associations are worth singling out. They changed when the
plastidial models were rebuilt on 2026-08-12, which is what shifted the GLK
correlations in `fig_photo_etc` panels C and D. Pinning the tag pins them.

Run logs and checkpoints resolve to this repository's own `projects/`
directory, so the Sorghum panels work from a bare checkout once the tables
above are in place.
