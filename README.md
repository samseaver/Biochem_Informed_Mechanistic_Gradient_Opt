# BioFlux — transcript-constrained flux for plant plastidial metabolism

Estimates metabolic fluxes in a plastidial reconstruction by relaxing a
transcript-derived flux vector onto a mass-balance constraint set with gradient
descent. It does not maximise biomass, and it does not train a neural network:
the transcript-derived vector is both the starting point and the upper bound,
and the solver reports how much of that pattern survives being made
mass-balanced.

The approach inherits the AMN architecture and its physics-informed loss
(Faure et al., 2023), but here the gradient descent solves for the flux vector
`V` directly. No network weights are learned.

Accompanies the preprint on *Sorghum bicolor* and *Populus trichocarpa*
plastidial metabolism under iron limitation.

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

See `requirements-runtime.txt` for pinned versions and the micromamba recipe.
Python 3.11, TensorFlow 2.15, cobra 0.31.

ModelSEEDPy and cobrakbase are **not** pip-installed; both scripts load them
from local checkouts via `sys.path.append`, and those paths are currently
hardcoded. Adjust them at the top of `predict_bioinformed_flux.py` and
`generate_feasible_flux.py`.

## Reference

Faure, L., et al. (2023). A neural-mechanistic hybrid approach improving the
predictive power of genome-scale metabolic models. *Nature Communications*.
