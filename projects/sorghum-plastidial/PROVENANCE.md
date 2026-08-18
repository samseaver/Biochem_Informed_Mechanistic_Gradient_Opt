# Sorghum plastidial results — preprint snapshot

The *Sorghum bicolor* half of the sweep behind the preprint, curated for
reproducibility. Poplar is not included here; the manuscript figures need both
species and are regenerated from the full sweep, which is deposited separately.

## Provenance

Produced 2026-08-12 by `predict_bioinformed_flux.py`, one process per mass
penalty, with:

    BF_SPECIES=Sorghum
    BF_PROJECT=<this directory>
    --svp <p> --seed 1786429390

**The seed matters:** it defaults to the wall clock, so omitting `--seed 1786429390` will not reproduce these numbers.

Initialization was `V0_init = -2` (evidence-only): reactions with a transcript
score start at their V_bf, and reactions without one start at 0 rather than
being imputed, so they are recruited only where mass balance demands flux.
Result files are named `startVbfandZero_*` accordingly.

Reactions without a transcript score include the transporters that carry the
medium, so those are seeded explicitly from
`integration_results/media_chain_init.tsv`, which is rebuilt from
`integration_results/fva.tsv` at the start of every run. See the repository
README for how that table is constructed.

## What is here, and what was dropped

Layout is one project directory with the four penalties under `ml/`. The
original sweep used one directory per (species, penalty) so the runs could go
concurrently without racing on shared inputs; `inputs/`, `integration_results/`
and `ml/training/` were byte-identical across all four and are stored once here.

The full checkpoint series (every 100 steps) is not published.

## Check

The resolution floor, sqrt(mean Leaf Mass_Loss / p), recomputed from the final
`Losses_step_*.tsv` of each penalty, reproduces the published values:

| p | floor |
|---|---|
| 2.0 | 0.1359 |
| 1.0 | 0.1547 |
| 0.5 | 0.1831 |
| 0.1 | 0.3372 |
