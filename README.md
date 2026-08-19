# Biochem_Informed_Mechanistic_Gradient_Opt (BioFlux)

Mechanistic loss function and gradient descent for the plastidial metabolism of
*Sorghum bicolor* (BTx623) and *Populus trichocarpa* (Nisqually-1) under iron
limitation.

## Preprint

**Simulating Iron Deficiency in Plant Plastidial Metabolism With a Flexible
Neural-Mechanistic Hybrid Approach**

El Alaoui, S., Henry, C. S., Blaby-Haas, C., Paape, T., Xie, M., and
Seaver, S. M. bioRxiv, version 2, posted 2026-08-17.
<https://doi.org/10.1101/2025.06.10.658179>

## Context

This repository estimates metabolic fluxes by relaxing a transcript-derived
flux vector onto a mass-balance constraint set with gradient descent. It is one
of three components:

| | |
|---|---|
| Genome annotation and reconstruction | [ModelSEED/PlantSEED](https://github.com/ModelSEED/PlantSEED) at tag **`v2.5`** |
| Transcript processing and reaction scores | [samseaver/RNASeq_Enzyme_Abundance](https://github.com/samseaver/RNASeq_Enzyme_Abundance) |
| **Mechanistic loss function and gradient descent** | **this repository** |

PlantSEED supplies the plastidial reconstruction, RNASeq_Enzyme_Abundance turns
transcript abundances into the per-reaction capacity estimates ($V_{bf}$) that
bound the fluxes, and this repository solves for the fluxes themselves.

It does not maximise biomass, and it does not train a neural network: the
transcript-derived vector is both the starting point and the upper bound, and
the solver reports how much of that pattern survives being made mass-balanced.
The approach inherits the AMN architecture and its physics-informed loss
(Faure et al., 2023), but here the gradient descent solves for the flux vector
`V` directly. No network weights are learned.

## Two scripts

```bash
# 1. capacity envelope + transcript-derived flux bounds
#    -> projects/<species>-plastidial/integration_results/{fva.tsv,vbf.tsv}
micromamba run -n bf-runtime python generate_feasible_flux.py

# 2. gradient descent
#    -> projects/<species>-plastidial/ml/svp_<p>/{results,checkpoints}/
micromamba run -n bf-runtime python predict_bioinformed_flux.py
```

Settings live in `parameters.py`. Everything else is a library.

    generate_feasible_flux.py     step 1
    predict_bioinformed_flux.py   step 2
    parameters.py                 species, penalties, epochs, learning rate
    Library/                      imported by the two scripts
    Paper_Figures/                the manuscript's figures and the code for them
      analysis_scripts/               rebuilds the analysis tables those scripts read
    tools/                        optional analyses, see tools/README.md
    projects/                     inputs and results, one directory per species

Step 2 depends on step 1: without `integration_results/fva.tsv` it exits and
says so. Everything else it needs, including the media-chain table, it builds
itself.

### Reproducing the published Sorghum runs

`projects/sorghum-plastidial/` ships with the repository and holds the Sorghum
half of the sweep behind the preprint, at all four mass penalties. See its
`PROVENANCE.md`. To re-run it:

```bash
micromamba run -n bf-runtime python predict_bioinformed_flux.py \
    --svp 2.0 --seed 1786429390
```

**The seed is not optional.** It defaults to the wall clock, so omitting
`--seed 1786429390` produces a different run. Everything else — the
initialization, the early-stopping rule, the epoch budget — matches the
published configuration by default.

Poplar was used in the paper but is not distributed here; the full sweep is
deposited separately. `BF_SPECIES=Poplar` needs a project directory you supply.

### Useful flags

| | |
|---|---|
| `--svp X` | run one penalty instead of every value in `SVP_VALUES` |
| `--seed N` | fix the random seed; **required to reproduce a previous run** |
| `--epochs N` | override the iteration budget |
| `--test` | short sweep at `TEST_EPOCHS`, writes under `ml/test/` |
| `BF_SPECIES=` | `Sorghum` (default) or `Poplar` |
| `BF_PROJECT=` | point at a different project directory |
| `BF_MIN_DELTA=`, `BF_PATIENCE=` | override the early-stopping rule |

To run several penalties at once, launch one process per penalty with `--svp`
pinned and a separate `BF_PROJECT` for each, rather than letting one process
work through them in sequence.

## Settings worth knowing about

`parameters.py` holds the rest. Three you may want to change:

| | default | |
|---|---|---|
| `SVP_VALUES` | `[2.0]` | mass-balance penalties to run. `2.0` is the operating point adopted in the paper. Adding values explores the trade-off — see `PROCESS.md` — but each is another full run, and the smaller penalties are the slow ones. |
| `EPOCHS` | `2.5e6` | iteration budget. Training stops per condition when its loss plateaus, so this is a ceiling, not a target. |
| `TREATMENT_FILTERS` | `("Control", "FeLim")` | which treatments to train on. |

`PROCESS.md` covers what the penalty actually buys, how the starting flux
vector is built and why the medium needs explicit seeding, how per-condition
early stopping works, and the call chain through the solver.

## Installation

Python 3.11, TensorFlow 2.15, cobra 0.31. GLPK comes from conda-forge because
the pip wheel does not carry the solver library `cobra` needs.

```bash
micromamba create -y -n bf-runtime -c conda-forge python=3.11 glpk
micromamba run -n bf-runtime pip install -r requirements-runtime.txt
```

Two dependencies, ModelSEEDPy and cobrakbase, are not on PyPI. The requirements
file installs them straight from GitHub at a pinned commit, so the command above
is all you need — but it does mean `git` must be on your PATH and the machine
must be able to reach github.com. If it cannot, clone the two repositories at
those commits and `pip install --no-deps -e` each one.

cobrakbase is pinned to the `cobra-model` branch rather than `master`: master
lacks the KBase-model support this uses.

## Reference — the AMN architecture

Faure, L., et al. (2023). A neural-mechanistic hybrid approach improving the
predictive power of genome-scale metabolic models. *Nature Communications*.
