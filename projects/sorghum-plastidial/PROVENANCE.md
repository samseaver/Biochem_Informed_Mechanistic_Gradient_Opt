# Sorghum plastidial results — preprint snapshot

The *Sorghum bicolor* half of the sweep behind the preprint, curated for
reproducibility. Poplar is not included here; the manuscript figures need both
species and are regenerated from the full sweep, which is deposited separately.

## Provenance

Produced 2026-08-12 by `predict_bioinformed_flux.py`, one process per mass
penalty, with:

    BF_SPECIES=Sorghum
    BF_PROJECT=<this directory>
    BF_PATIENCE=500  BF_MIN_DELTA=1e-3
    --svp <p> --epochs 2500000 --seed 1786429390

`BF_PATIENCE`, `BF_MIN_DELTA` and `--epochs` were explicit at the time because
the in-code defaults then differed; they now match, so only `BF_SPECIES`,
`BF_PROJECT`, `--svp` and `--seed` are strictly required. **The seed matters:**
it defaults to the wall clock, so omitting `--seed 1786429390` will not
reproduce these numbers.

Initialization was `V0_init = -2` (evidence-only): reactions with a transcript
score start at their V_bf, unscored reactions start at 0, and each media chain
is seeded from `integration_results/media_chain_init.tsv`. Result files are
named `startVbfandZero_*` accordingly.

## What is here, and what was dropped

Layout is one project directory with the four penalties under `ml/`. The
original run used one directory per (species, penalty) so the arms could run
concurrently without racing on shared inputs; `inputs/`, `integration_results/`
and `ml/training/` were byte-identical across all four and are stored once here.

Dropped to keep the snapshot small, all of it recomputable:

| dropped | size | why |
|---|---|---|
| `results/*_Pout.tsv` | 9.3 MB x4 | selection matrix, derived from the model |
| `ml/training/temp_Pout.tsv` | 9.3 MB | the same matrix again |
| most `V_step_*.tsv` | ~450 MB/arm | only step 0 and the final step are kept |
| most `Losses_step_*.tsv` | ~98 MB/arm | thinned to every 10,000 steps, plus the final |

The full checkpoint series (every 100 steps) is not published.

## Caveats

- `frozen_at_step.tsv` exists for `svp_0.1` only. It is written when *every*
  condition freezes, and the other three arms reached the 2,500,000 iteration
  cap with non-leaf conditions still moving. All eleven Sorghum leaf conditions
  froze in every arm; the per-condition freeze steps are in the run logs.
- `cobraname` and `mediumname` inside `training.npz` are absolute paths from the
  original run directory and no longer resolve. Every numeric array in that file
  is identical across the four arms and unaffected.

## Check

The resolution floor, sqrt(mean Leaf Mass_Loss / p), recomputed from the final
`Losses_step_*.tsv` of each arm, reproduces the published values:

| p | floor |
|---|---|
| 2.0 | 0.1359 |
| 1.0 | 0.1547 |
| 0.5 | 0.1831 |
| 0.1 | 0.3372 |
