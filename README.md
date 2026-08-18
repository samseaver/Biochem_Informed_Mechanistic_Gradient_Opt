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
says so. Everything else it needs, including the media-chain table described
below, it builds itself.

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
loop — see the concurrency note below.

## The mass-balance penalty

The loss has two terms in tension: how far the fluxes sit below their
transcript-derived ceilings, and how badly mass balance is violated. The
penalty `p` weights the second against the first, and `SVP_VALUES` in
`parameters.py` sets which values to run. It defaults to `[2.0]`, the operating
point adopted in the paper.

Raising `p` tightens mass balance and loosens the transcript fit. That trade-off
is monotone: at converged optima a weighted objective cannot improve both, so
seeing both improve at once is a sign that an arm has not converged rather than
a better setting.

What `p` does **not** change is the flux pattern. Across the four penalties in
the paper, every pairwise correlation between converged flux vectors exceeds
0.999. What it changes is the resolution floor — the smallest flux difference
distinguishable from the residual mass imbalance:

| p | Sorghum | Poplar |
|---|---|---|
| 0.1 | 0.337 | 0.284 |
| 0.5 | 0.183 | 0.193 |
| 1.0 | 0.155 | 0.172 |
| 2.0 | 0.136 | 0.154 |

So `p` buys resolution, not a different answer, which is why the tightest
setting was adopted. Exploring it is worthwhile but not free: every added value
is another full run, and the *smaller* penalties are the slow ones — at
`p = 0.1` the slowest condition needed 1.67 million iterations against 1.28
million at `p = 2.0`. Run them concurrently, not serially.

## How the initial flux vector is built

Reactions with a transcript score start at their `V_bf`. Reactions without one
start at zero and are recruited only where mass balance demands flux, rather
than being imputed at a network average.

That leaves a gap: the transporters carrying the medium have no transcript
score either, so zeroing everything unscored would strand the medium. Each
medium compound reaches the stroma through a short series of reactions — the
boundary exchange, then the e0/c0 and c0/d0 transporters — and those are seeded
explicitly. At steady state a series carries one flux throughout, so every leg
gets the same value: the exchange's net FVA capacity, on the column matching
the exchange's direction, with the opposing column held at zero. That is 39
columns across the nine medium compounds (photon, CO2, ammonia, sulfate,
phosphate, chloride and water in; protons and O2 out).

`get_V0()` in `Library/Build_Model.py` builds that table at the start of every
run, from `integration_results/fva.tsv`, so it cannot go stale against a re-run
FVA. The alternative initializations are documented alongside `V0_init` in the
same function; the choice is a deliberate edit rather than a command-line
option, and result files are named for it — `startVbfandZero_*` for the setting
used here.

The medium itself is hard-wired autotrophic. All nine exchanges are
unidirectional by construction, so no organic carbon or nitrogen can enter
under any bound setting: every carbon atom originates as CO2 and every nitrogen
as NH3. Simulating mixotrophic or heterotrophic growth means adding or
replacing exchange reactions, not relaxing bounds.

## Early stopping

Each condition trains until its own loss plateaus. When a condition's relative
improvement stays below `BF_MIN_DELTA` for `BF_PATIENCE` consecutive iterations
it is frozen — its row of the gradient is zeroed — while the rest continue.
Conditions do not interact during training: each one's gradient depends only on
its own flux vector, so freezing one neither helps nor hinders the others. It
records that one as done, and yields a per-condition convergence step rather
than a single global stopping point.

Defaults are `patience = 500`, `min_delta = 1e-3`, as used in the paper.
`tools/replay_early_stopping.py` chooses such a rule offline from a completed
run's saved loss series, without retraining.

## Architecture

`MM_QP` → `run_MM_QP` → `QP_layers`, which calls `get_V0` for the starting
vector and then `Gradient_Descent`. The descent loop calls `Loss_all` each
iteration, summing the mass-balance term (`Loss_SV`), the transcript-ceiling
term (`Loss_Vout_constraint`), the medium bound (`Loss_Vin`), positivity
(`Loss_Vpos`) and a complementarity loop-law penalty (`Loss_loop`, always on at
lambda_c = 0.01) that suppresses futile cycles between duplicated reversible
pairs. After each update a proportional soft clamp holds every reversible pair
at or below 1.10 × its `V_bf`.

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
