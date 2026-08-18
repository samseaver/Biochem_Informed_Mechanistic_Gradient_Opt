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

Starting the unscored reactions at zero would strand the medium, because the
transporters that carry it have no transcript score either. Each medium
compound therefore reaches the plastid stroma through a short series of
reactions — the boundary exchange, the e0/c0 transporter, then the c0/d0
transporter — and all of them are seeded explicitly. At steady state a series
carries one flux throughout, so every leg is set to the same value: the
exchange's net FVA capacity, on the column matching the exchange's own
direction, with the opposing column held at 0. That is 39 columns across the
nine medium compounds (photon, CO2, ammonia, sulfate, phosphate, chloride,
water in; protons and O2 out).

Those values are in `integration_results/media_chain_init.tsv`, one row per
column with its compound and role. The table is derived entirely from
`integration_results/fva.tsv` and is rebuilt at the start of every run, so it
cannot go stale against a re-run FVA.

## What is here, and what was dropped

Layout is one project directory with the four penalties under `ml/`. The
original run used one directory per (species, penalty) so the arms could run
concurrently without racing on shared inputs; `inputs/`, `integration_results/`
and `ml/training/` were byte-identical across all four and are stored once here.

The full checkpoint series (every 100 steps) is not published.

## Check

The resolution floor, sqrt(mean Leaf Mass_Loss / p), recomputed from the final
`Losses_step_*.tsv` of each arm, reproduces the published values:

| p | floor |
|---|---|
| 2.0 | 0.1359 |
| 1.0 | 0.1547 |
| 0.5 | 0.1831 |
| 0.1 | 0.3372 |
