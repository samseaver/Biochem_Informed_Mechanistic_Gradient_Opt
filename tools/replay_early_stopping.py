#!/usr/bin/env python
"""
Replay the per-condition early-stopping logic over saved Losses_step_*.tsv files,
for a grid of (patience_limit, min_delta) pairs. The goal is to pick the tightest
pair that doesn't sacrifice the final loss quality on any condition.

This is a post-hoc analysis: no retraining required. Run after at least one full
gradient-descent run has populated the checkpoints directory. Designed to be
dropped in next to predict_bioinformed_flux.py.

Two input modes:
  1. Coarse mode (default): Losses_step_*.tsv files written every K iterations.
     Patience values supplied in iteration units are divided by K (rounded up to 1)
     to translate into "checkpoints of no improvement". Frozen-at-step reports are
     accurate to within +/-K iterations. Use this on existing runs without re-running.
  2. Exact mode (--per-iter <file>): a single TSV with one row per iteration and one
     column per condition. Add a per-iteration np.savetxt at the end of
     Gradient_Descent to produce this. The replay then matches the live behavior
     exactly.

Usage:
    python replay_early_stopping.py [path/to/checkpoints_dir]

Default checkpoints_dir is derived from parameters.py.
"""

import os
import re
import sys
import glob
import argparse
import numpy as np
import pandas as pd

try:
    from parameters import Parameters_ML
    default_ckpt = os.path.join(Parameters_ML().training_folder, "checkpoints")
except Exception:
    default_ckpt = None

parser = argparse.ArgumentParser()
parser.add_argument("ckpt_dir", nargs="?", default=default_ckpt,
                    help="checkpoints directory containing Losses_step_*.tsv")
parser.add_argument("--patience", type=int, nargs="+",
                    default=[200, 500, 1000, 2000, 5000],
                    help="patience_limit values to sweep")
parser.add_argument("--min-delta", type=float, nargs="+",
                    default=[1e-3, 5e-3, 1e-2, 5e-2],
                    help="min_delta values to sweep")
parser.add_argument("--gap-tol", type=float, default=0.10,
                    help="max acceptable (loss_at_freeze / final_loss - 1)")
parser.add_argument("--labels", type=str, default=None,
                    help="optional path to a one-per-line file of condition labels")
parser.add_argument("--out", type=str, default=None,
                    help="output directory (defaults to <ckpt_dir>/../replay)")
parser.add_argument("--per-iter", type=str, default=None,
                    help="path to a per-iteration loss TSV (rows=iterations, cols=conditions). "
                         "If given, ckpt_dir is ignored and replay is exact.")
args = parser.parse_args()

if args.per_iter is None:
    if args.ckpt_dir is None:
        sys.exit("No checkpoints directory available; pass one as the first argument.")
    if not os.path.isdir(args.ckpt_dir):
        sys.exit(f"Checkpoints directory does not exist: {args.ckpt_dir}")
    base_for_out = args.ckpt_dir
else:
    if not os.path.isfile(args.per_iter):
        sys.exit(f"Per-iteration loss file does not exist: {args.per_iter}")
    base_for_out = os.path.dirname(args.per_iter) or "."
out_dir = args.out or os.path.join(os.path.dirname(base_for_out.rstrip("/")), "replay")
os.makedirs(out_dir, exist_ok=True)

step_re = re.compile(r"Losses_step_(\d+)\.tsv$")

def load_losses(ckpt_dir):
    """Return (steps, mat) where mat[i, c] = Total_Loss at steps[i] for condition c."""
    rows = []
    for f in glob.glob(os.path.join(ckpt_dir, "Losses_step_*.tsv")):
        m = step_re.search(os.path.basename(f))
        if not m:
            continue
        step = int(m.group(1))
        df = pd.read_csv(f, sep="\t")
        rows.append((step, df["Total_Loss"].to_numpy()))
    if not rows:
        sys.exit(f"No Losses_step_*.tsv files found in {ckpt_dir}")
    rows.sort(key=lambda r: r[0])
    steps = np.array([r[0] for r in rows])
    mat = np.stack([r[1] for r in rows], axis=0)
    return steps, mat

def replay(losses_per_step, patience, min_delta):
    """For each condition, return the step-index at which it would freeze."""
    n_steps, n_cond = losses_per_step.shape
    best   = np.full(n_cond, np.inf)
    pat    = np.zeros(n_cond, dtype=int)
    frozen = np.full(n_cond, n_steps, dtype=int)  # n_steps = "never froze"
    for i in range(n_steps):
        cur = losses_per_step[i]
        denom = np.where(np.isfinite(best), best, 1.0)
        improved = np.isinf(best) | (((best - cur) / denom) > min_delta)
        best = np.where(improved, np.minimum(best, cur), best)
        pat  = np.where(improved, 0, pat + 1)
        newly = (pat >= patience) & (frozen == n_steps)
        frozen[newly] = i
    return frozen

if args.per_iter is not None:
    # Exact mode: one row per iteration
    df = pd.read_csv(args.per_iter, sep="\t")
    losses = df.to_numpy()
    n_steps, n_cond = losses.shape
    steps = np.arange(1, n_steps + 1)  # iteration index
    stride = 1
    print(f"Exact mode: {n_steps} iterations x {n_cond} conditions from {args.per_iter}")
else:
    steps, losses = load_losses(args.ckpt_dir)
    n_steps, n_cond = losses.shape
    diffs = np.diff(steps) if n_steps > 1 else np.array([1])
    stride = int(diffs[0]) if len(set(diffs)) == 1 else int(np.median(diffs))
    print(f"Coarse mode: {n_steps} checkpoints x {n_cond} conditions from {args.ckpt_dir}")
    print(f"Step range: {steps.min()} - {steps.max()}  (cadence: {stride} iterations/checkpoint)")
    print(f"Patience values will be divided by {stride} (rounded up) to translate into checkpoint units.")
    print(f"For exact replay, log per-iteration losses and pass --per-iter <file>.")

labels = None
if args.labels and os.path.exists(args.labels):
    with open(args.labels) as fh:
        labels = [ln.strip() for ln in fh if ln.strip()]
    if len(labels) != n_cond:
        print(f"Warning: labels file has {len(labels)} lines but data has {n_cond} conditions; ignoring.")
        labels = None
if labels is None:
    labels = [f"cond_{i}" for i in range(n_cond)]

final_loss = losses[-1]  # what we ultimately reached

results = []  # (patience, min_delta, max_gap, worst_label, worst_gap, max_frozen_step, n_never_froze)
for p_iter in args.patience:
    # Translate patience from iteration-units to whatever unit `losses` is sampled at.
    p_eff = max(1, int(np.ceil(p_iter / stride)))
    for d in args.min_delta:
        fr = replay(losses, p_eff, d)
        # loss_at_freeze: if a condition never froze, treat freeze as final step
        idx = np.where(fr < n_steps, fr, n_steps - 1)
        loss_at_freeze = losses[idx, np.arange(n_cond)]
        gap = loss_at_freeze / np.maximum(final_loss, 1e-30) - 1.0
        worst_c = int(np.argmax(gap))
        max_frozen_step = int(steps[idx].max())
        n_never = int(np.sum(fr == n_steps))
        results.append({
            "patience": p_iter,
            "patience_in_samples": p_eff,
            "min_delta": d,
            "max_gap_pct": float(gap.max()) * 100.0,
            "median_gap_pct": float(np.median(gap)) * 100.0,
            "worst_condition": labels[worst_c],
            "max_frozen_at_step": max_frozen_step,
            "median_frozen_at_step": int(np.median(steps[idx])),
            "n_never_froze": n_never,
        })

table = pd.DataFrame(results)
table = table.sort_values(["max_gap_pct", "max_frozen_at_step"]).reset_index(drop=True)
table_path = os.path.join(out_dir, "grid_summary.tsv")
table.to_csv(table_path, sep="\t", index=False, float_format="%.4g")

print("\n=== Replay grid summary (sorted by max_gap_pct, then by max_frozen_step) ===")
print(table.to_string(index=False))

# Pick the recommended setting: smallest max_frozen_step among rows whose max_gap_pct <= gap_tol%
acceptable = table[table["max_gap_pct"] <= args.gap_tol * 100.0]
if len(acceptable) == 0:
    print(f"\nNo grid point keeps max_gap <= {args.gap_tol*100:.0f}%; loosen gap-tol or sweep further.")
else:
    rec = acceptable.sort_values(["max_frozen_at_step", "patience"]).iloc[0]
    print(f"\n=== Recommended setting (smallest budget with max_gap <= {args.gap_tol*100:.0f}%) ===")
    print(f"  patience_limit = {int(rec['patience'])}")
    print(f"  min_delta      = {rec['min_delta']}")
    print(f"  max gap        = {rec['max_gap_pct']:.2f}%  (worst: {rec['worst_condition']})")
    print(f"  freezes all conditions by step {int(rec['max_frozen_at_step'])} of {steps.max()}")

# Per-condition table at the recommended setting (or first row if none acceptable)
chosen = (acceptable.iloc[0] if len(acceptable) else table.iloc[0])
fr = replay(losses, int(chosen["patience_in_samples"]), float(chosen["min_delta"]))
idx = np.where(fr < n_steps, fr, n_steps - 1)
loss_at_freeze = losses[idx, np.arange(n_cond)]
gap_pct = (loss_at_freeze / np.maximum(final_loss, 1e-30) - 1.0) * 100.0
per_cond = pd.DataFrame({
    "condition_idx": np.arange(n_cond),
    "label":         labels,
    "frozen_at_step": np.where(fr < n_steps, steps[idx], -1),
    "loss_at_freeze": loss_at_freeze,
    "final_loss":     final_loss,
    "gap_pct":        gap_pct,
}).sort_values("frozen_at_step")
per_cond_path = os.path.join(out_dir, "per_condition_at_recommended.tsv")
per_cond.to_csv(per_cond_path, sep="\t", index=False, float_format="%.6g")
print(f"\nWrote {table_path}")
print(f"Wrote {per_cond_path}")
