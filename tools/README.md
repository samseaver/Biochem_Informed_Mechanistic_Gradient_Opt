# tools/

Optional analyses. Neither pipeline script calls anything here, and nothing
here is needed to reproduce the published results.

## `replay_early_stopping.py`

Replays the per-condition early-stopping rule over the `Losses_step_*.tsv`
files a run has already written, for a grid of `(patience_limit, min_delta)`
pairs, and reports where each condition would have frozen and what its loss
would have been. No retraining: it reads checkpoints and re-applies the
decision rule offline.

The point is to choose a stopping rule without paying for a training run per
candidate. Early stopping is per condition — a condition freezes once its
relative improvement stays under `min_delta` for `patience_limit` iterations
(`Gradient_Descent` in `Library/Build_Model.py`) — so the question "would a
tighter rule have cost me anything?" is answerable from the saved loss series
alone.

That is how the published setting was chosen. `min_delta` had been `1e-2`;
replaying the grid showed `1e-3` was affordable, and the sweep behind the
preprint used `patience=500, min_delta=1e-3`. Tightening it lengthened the runs
considerably — freeze steps grew four- to tenfold — which is worth knowing
before changing it again.

Run it after at least one full run has populated a `checkpoints/` directory:

    micromamba run -n bf-runtime python tools/replay_early_stopping.py --help

Note the thinned snapshot in `projects/sorghum-plastidial/` keeps only every
10,000th `Losses_step` file, which is too coarse for a faithful replay — a rule
with `patience=500` operates on a scale of hundreds of iterations. Point this
at a full local run, not at the published subset.
