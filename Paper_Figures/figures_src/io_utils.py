"""
io_utils.py — data loaders and log parsers shared by figure scripts.

All paths default to the analysis directory two levels up from this file.
Override the data location by passing ``data_dir`` to any loader, or by
setting the ``BIOFLUX_DATA_DIR`` environment variable.

Functions
---------

- :func:`data_dir` — return the active data directory path
- :func:`load_tsv` — load a TSV from ``cross_species_analysis/``
- :func:`load_freeze_events` — return parsed freeze events (preferred)
- :func:`parse_freeze_events_from_log` — fallback parser from raw ``*_output.txt``
- :func:`parse_loss_summary` — final loss block from a run log
- :func:`load_loss_trajectory` — read all ``Losses_step_*.tsv`` from a
  checkpoints directory into a long DataFrame
- :func:`cond_to_tissue_treatment_tp` — split a condition label

If you need a different TSV that doesn't have a convenience loader yet,
just call :func:`load_tsv` with its filename.
"""

from __future__ import annotations

import os
import re
from glob import glob

import numpy as np
import pandas as pd


# ==== PATH RESOLUTION =============================================

#: Default data root (the ``Paper_Figures/`` directory). Resolved at import.
_DEFAULT_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)


def data_dir(override: str | None = None) -> str:
    """Return the path to ``Paper_Figures/`` (or the override / env var)."""
    if override is not None:
        return os.path.abspath(override)
    return os.environ.get("BIOFLUX_DATA_DIR", _DEFAULT_DATA_DIR)


# ==== EXTERNAL DATA DEPENDENCIES ==================================
#
# Two inputs come from sibling repositories rather than from this one. Both
# are resolved through an env var with a sibling-checkout default, and both
# raise a message naming the variable if they are missing, rather than a bare
# FileNotFoundError on a path that means nothing to the reader.

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)


def plantseed_roles(override: str | None = None) -> str:
    """Path to PlantSEED_Roles.json (reaction -> subsystem).

    ModelSEED/PlantSEED at tag **v2.5** (commit 212b817). PlantSEED is a data
    repository with no setup.py, so unlike ModelSEEDPy and cobrakbase it cannot
    be pip-installed -- clone it and point BIOFLUX_PLANTSEED_DIR at the
    checkout root:

        git clone --branch v2.5 https://github.com/ModelSEED/PlantSEED

    Note the file lives under Data/PlantSEED_v3/ even at tag v2.5; both
    directories exist at that tag.
    """
    if override:
        return os.path.abspath(override)
    root = os.environ.get("BIOFLUX_PLANTSEED_DIR",
                          os.path.join(_REPO_ROOT, "PlantSEED"))
    path = os.path.join(root, "Data", "PlantSEED_v3", "PlantSEED_Roles.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "PlantSEED_Roles.json not found at %s.\n"
            "Clone ModelSEED/PlantSEED at tag v2.5 and set BIOFLUX_PLANTSEED_DIR "
            "to the checkout root." % path)
    return path


def rnaseq_dir(override: str | None = None) -> str:
    """Path to the RNASeq project directory supplying transcript inputs.

    samseaver/RNASeq_Enzyme_Abundance at tag **bioflux-preprint-260813**, the
    projects/qpsi-plastidial directory inside it. Supplies the reaction molar
    fractions (gene -> reaction associations) and the per-gene TMM tables.
    Override with BIOFLUX_RNASEQ_DIR.
    """
    if override:
        return os.path.abspath(override)
    d = os.environ.get("BIOFLUX_RNASEQ_DIR",
                       os.path.join(_REPO_ROOT, "RNASeq_Enzyme_Abundance",
                                    "projects", "qpsi-plastidial"))
    if not os.path.isdir(d):
        raise FileNotFoundError(
            "RNASeq project directory not found at %s.\n"
            "Clone samseaver/RNASeq_Enzyme_Abundance at tag "
            "bioflux-preprint-260813 and set BIOFLUX_RNASEQ_DIR to its "
            "projects/qpsi-plastidial directory." % d)
    return d


#: Default sub-directory name for the cross-species analysis TSV layer.
#: Override via the ``BIOFLUX_CROSS_DIR`` env var (either an absolute path
#: or a sibling name like ``"cross_species_analysis"`` to read the legacy
#: archive). Default ``"cross_species_analysis_fresh"`` reads the LaTeX-
#: aligned per-svp outputs regenerated after the loss-reporting fix.
_DEFAULT_CROSS_SUBDIR = "analysis_cache"


def cross_species_dir(override: str | None = None) -> str:
    """Return the path to the cross-species TSV directory.

    Resolution order:
      1. explicit ``override`` argument (treated as relative to ``data_dir()``
         if not absolute);
      2. ``BIOFLUX_CROSS_DIR`` env var (same convention);
      3. ``_DEFAULT_CROSS_SUBDIR`` under ``data_dir()``.
    """
    sub = (override
           or os.environ.get("BIOFLUX_CROSS_DIR")
           or _DEFAULT_CROSS_SUBDIR)
    if os.path.isabs(sub):
        return sub
    return os.path.join(data_dir(), sub)


# ==== GENERAL TSV LOADER ==========================================

def load_tsv(name: str, override_dir: str | None = None,
             dtype: dict | None = None) -> pd.DataFrame:
    """Load a TSV from ``cross_species_analysis/`` by filename.

    The ``svp`` column is always read as a string so that "0.1" / "0.5" stay
    as labels rather than being auto-parsed to floats (which breaks sorts
    and groupbys downstream).

    Parameters
    ----------
    name : str
        Filename relative to ``cross_species_analysis/`` (e.g.
        ``"iron_timecourse.tsv"``).
    override_dir : str, optional
        If given, look in this directory instead.
    dtype : dict, optional
        Extra dtype hints to pass through to ``pd.read_csv``.
    """
    if override_dir is not None:
        path = os.path.join(override_dir, name)
    else:
        path = os.path.join(cross_species_dir(), name)
    full_dtype = {"svp": str}
    if dtype:
        full_dtype.update(dtype)
    return pd.read_csv(path, sep="\t", dtype=full_dtype)


# ==== FREEZE EVENTS ===============================================

def _attach_condition_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``label``, ``tissue``, ``treatment`` columns by reading
    ``training.npz`` per species and indexing by ``cond_idx``.
    Conditions without a label are dropped.
    """
    if df.empty:
        for c in ("label", "tissue", "treatment"):
            df[c] = []
        return df
    out_rows = []
    npz_cache: dict[str, list[str]] = {}
    for sp in df["species"].unique():
        npz = fresh_training_npz(sp)
        if npz is None:
            npz_cache[sp] = []
            continue
        data = np.load(npz, allow_pickle=True)
        names = [n.decode("utf-8") if isinstance(n, bytes) else str(n)
                 for n in data["treatments"]] if "treatments" in data.files else []
        npz_cache[sp] = names
    for _, row in df.iterrows():
        names = npz_cache.get(row["species"], [])
        ci = int(row["cond_idx"])
        if 0 <= ci < len(names):
            lbl = names[ci]
            tissue, treat, _tp = cond_to_tissue_treatment_tp(lbl)
        else:
            lbl, tissue, treat = "", "", ""
        out = dict(row)
        out["label"] = lbl
        out["tissue"] = tissue
        out["treatment"] = treat
        out_rows.append(out)
    return pd.DataFrame(out_rows)


def load_freeze_events(override_dir: str | None = None,
                       source: str = "fresh") -> pd.DataFrame:
    """Return parsed per-condition freeze events for all (species, svp) runs.

    Parameters
    ----------
    override_dir : str, optional
        For ``source="archive"``: directory holding the pre-built TSV /
        legacy log files. Defaults to ``data_dir()``.
    source : {"fresh", "archive"}
        - ``"fresh"`` (default): parse the per-svp ``run_output.txt`` logs
          under ``Biochem_*/projects/<spc>/ml/svp_X.X/`` and attach
          condition labels from each species' ``training.npz``.
        - ``"archive"``: prefer the pre-built
          ``freeze_events_both_species.tsv`` then ``freeze_events_all.tsv``
          under ``data_dir()``, falling back to parsing the legacy
          ``*_output.txt`` logs.

    Always returns columns: species, svp, freeze_step, cond_idx,
    loss_at_freeze, label, tissue, treatment.
    """
    if source == "fresh":
        rows = []
        for sp in _LEGACY_SPECIES_DIRS:      # keys are the species names
            for svp_str in ("0.1", "0.5", "1.0", "2.0"):
                log = fresh_run_log(sp, svp_str)
                if log is None:
                    continue
                sub = parse_freeze_events_from_log(log)
                if sub.empty:
                    continue
                sub["species"] = sp
                sub["svp"] = svp_str
                rows.append(sub)
        if not rows:
            return pd.DataFrame(columns=[
                "species", "svp", "freeze_step", "cond_idx",
                "loss_at_freeze", "label", "tissue", "treatment"])
        df = pd.concat(rows, ignore_index=True)
        return _attach_condition_labels(df)

    # Archive source.
    root = data_dir(override_dir)
    for candidate in ("freeze_events_both_species.tsv", "freeze_events_all.tsv"):
        path = os.path.join(root, candidate)
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t", dtype={"svp": str})
            if "species" not in df.columns:
                df["species"] = "Poplar"
            return df
    rows = []
    for sp, prefix in [("Poplar", "poplar"), ("Sorghum", "sorghum")]:
        for svp_tag, svp_str in [("01", "0.1"), ("05", "0.5"),
                                  ("1", "1.0"), ("2", "2.0")]:
            for fname in (f"{prefix}_svp{svp_tag}_output.txt",
                          f"{prefix}_svp{svp_tag}_ouput.txt"):
                full = os.path.join(root, fname)
                if os.path.exists(full):
                    sub = parse_freeze_events_from_log(full)
                    sub["species"] = sp
                    sub["svp"] = svp_str
                    rows.append(sub)
                    break
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


_FREEZE_RE = re.compile(
    r"Step (\d+): freezing conditions \[(\d+)\] \(losses \[([\d.eE+-]+)\]\)"
)


def parse_freeze_events_from_log(log_path: str) -> pd.DataFrame:
    """Regex-parse freeze events from a single ``*_output.txt`` log file.

    Returns columns: freeze_step, cond_idx, loss_at_freeze.
    """
    rows = []
    with open(log_path) as fh:
        for line in fh:
            m = _FREEZE_RE.search(line)
            if m:
                rows.append({
                    "freeze_step":     int(m.group(1)),
                    "cond_idx":        int(m.group(2)),
                    "loss_at_freeze":  float(m.group(3)),
                })
    return pd.DataFrame(rows)


# ==== LOSS SUMMARY (final block from a run log) ===================

_LOSS_FIELDS = {
    "R2":             re.compile(r"^R2\s*=\s*([\-\d.]+)"),
    "Loss_C":         re.compile(r"^Loss Constrained Targets\s+([\d.eE+-]+)"),
    "Loss_SV":        re.compile(r"^Loss SV\s+([\d.eE+-]+)"),
    "Loss_Vin":       re.compile(r"^Loss Vin bound\s+([\d.eE+-]+)"),
    "Loss_Vpos":      re.compile(r"^Loss V positive\s+([\d.eE+-]+)"),
    "Time_elapsed":   re.compile(r"^Time elapsed:\s+(.+)"),
}
_EXIT_RE = re.compile(r"All \d+ conditions plateaued at step (\d+)")


def parse_loss_summary(log_path: str) -> dict:
    """Extract the final loss block + exit step from a single run log."""
    result: dict = {"R2": None, "Loss_C": None, "Loss_SV": None,
                    "Loss_Vin": None, "Loss_Vpos": None,
                    "Time_elapsed": None, "exit_step": None}
    with open(log_path) as fh:
        for line in fh:
            # Exit step
            m = _EXIT_RE.search(line)
            if m:
                result["exit_step"] = int(m.group(1))
                continue
            # Final loss block
            for key, pat in _LOSS_FIELDS.items():
                m = pat.match(line.strip())
                if m:
                    val = m.group(1)
                    if key != "Time_elapsed":
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                    result[key] = val
    return result


#: Project root used to resolve the per-svp run logs. This module is vendored
#: at <repo>/Paper_Figures/figures_src/, so two levels up is the repository
#: root and the runs live in its projects/ directory.
_DEFAULT_FRESH_PROJECTS = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "projects")
)

#: Per-species sub-paths under the fresh projects tree.
#: CLAUDE-RAW 2026-08-03: repointed to the raw-score + loop-law arm so the
#: ML-diagnostics deck (fig2/fig3/fig4 -> fig_ml_rslt) reflects the raw model.
#: CLAUDE 2026-08-03: the -rawnocap-Bcomp scratch arms were cleaned up and the
#: raw-score + loop-law pipeline was re-run in place in the BASE project dirs,
#: so point here (uncapped objective r_s, loop-law, full svp sweep 2.0/1.0/0.5/0.1).
_LEGACY_SPECIES_DIRS = {
    "Poplar":  "qpsi-260406-plastid-poplar",
    "Sorghum": "qpsi-260406-plastid-sorghum",
}

#: Current layout: one directory per species, penalties under ml/svp_<S>/.
_SPECIES_DIRS = {
    "Poplar":  "poplar-plastidial",
    "Sorghum": "sorghum-plastidial",
}

#: CLAUDE 2026-08-18: the sweep was reorganised from one directory per
#: (species, svp) into one directory per species, with the four penalties under
#: ml/. The per-arm split existed only so the eight concurrent runs would not
#: race on the shared inputs/, integration_results/ and ml/training/ files; those
#: were byte-identical across the arms of a species and are now stored once.
#:
#: Layout: <projects>/<species>-plastidial/ml/svp_<S>/...
#:
#: Env overrides:
#:   BIOFLUX_SPECIES_DIRS is not configurable; set BIOFLUX_FRESH_PROJECTS to
#:   relocate the projects root.
#:   BIOFLUX_LEGACY_LAYOUT=1 -- fall back to the pre-260811 qpsi-* dirs.
_DEFAULT_MIN_DELTA = "1e-3"


def _legacy_layout() -> bool:
    return os.environ.get("BIOFLUX_LEGACY_LAYOUT", "") not in ("", "0")


def fresh_arm_dir(species: str, svp_str: str,
                  projects_root: str | None = None) -> str | None:
    """Absolute path to the project directory for one (species, svp) cell.

    Returns None for an unknown species. Does not check existence -- the
    callers below each test for the specific file they need.
    """
    root = projects_root or os.environ.get(
        "BIOFLUX_FRESH_PROJECTS", _DEFAULT_FRESH_PROJECTS)
    if _legacy_layout():
        spc = _LEGACY_SPECIES_DIRS.get(species)
        return os.path.join(root, spc) if spc else None
    spc = _SPECIES_DIRS.get(species)
    return os.path.join(root, spc) if spc else None


def fresh_run_log(species: str, svp_str: str,
                  projects_root: str | None = None) -> str | None:
    """Return the path to the FRESH ``run_output.txt`` for (species, svp).

    ``svp_str`` is the human-readable label ("0.1", "0.5", "1.0", "2.0").
    Returns None if the path does not exist on disk.
    """
    d = fresh_arm_dir(species, svp_str, projects_root)
    if d is None:
        return None
    path = os.path.join(d, "ml", f"svp_{svp_str}", "run_output.txt")
    return path if os.path.exists(path) else None


def fresh_checkpoints_dir(species: str, svp_str: str,
                          projects_root: str | None = None) -> str | None:
    """Return the path to the FRESH per-svp ``checkpoints/`` directory."""
    d = fresh_arm_dir(species, svp_str, projects_root)
    if d is None:
        return None
    path = os.path.join(d, "ml", f"svp_{svp_str}", "checkpoints")
    return path if os.path.isdir(path) else None


def fresh_training_npz(species: str, svp_str: str = "2.0",
                       projects_root: str | None = None) -> str | None:
    """Return the path to the FRESH ``training.npz`` for ``species``.

    Under the arms-260811 layout every cell carries its own copy. They are
    scientifically identical across svp -- verified 2026-08-12: S, Pin, Pout,
    X, Y, reactions, treatments and LB all match bit-for-bit, and the only
    differing entries are the embedded project path strings -- so ``svp_str``
    only decides which copy is read, not what it contains.
    """
    d = fresh_arm_dir(species, svp_str, projects_root)
    if d is None:
        return None
    path = os.path.join(d, "ml", "training", "training.npz")
    return path if os.path.exists(path) else None


def load_loss_summary_all(override_dir: str | None = None,
                          source: str = "fresh") -> pd.DataFrame:
    """Return a tidy DataFrame: (species, svp, Loss_SV, Loss_C, R2, exit_step, runtime).

    Parameters
    ----------
    override_dir : str, optional
        For ``source="archive"``: directory holding the archive
        ``*_svp*_output.txt`` logs. Defaults to ``data_dir()``.
    source : {"fresh", "archive"}
        - ``"fresh"`` (default): read ``Biochem_*/projects/<spc>/ml/svp_X.X/run_output.txt``
          produced by the LaTeX-aligned sweep.
        - ``"archive"``: read legacy ``{prefix}_svp{tag}_output.txt`` under
             ``data_dir()``. Unused; nothing in this repository calls it.
    """
    rows = []
    if source == "fresh":
        for sp in _LEGACY_SPECIES_DIRS:      # keys are the species names
            for svp_str in ("0.1", "0.5", "1.0", "2.0"):
                full = fresh_run_log(sp, svp_str)
                if full is not None:
                    s = parse_loss_summary(full)
                    s["species"] = sp
                    s["svp"] = svp_str
                    rows.append(s)
    else:
        root = data_dir(override_dir)
        for sp, prefix in [("Poplar", "poplar"), ("Sorghum", "sorghum")]:
            for svp_tag, svp_str in [("01", "0.1"), ("05", "0.5"),
                                      ("1", "1.0"), ("2", "2.0")]:
                for fname in (f"{prefix}_svp{svp_tag}_output.txt",
                              f"{prefix}_svp{svp_tag}_ouput.txt"):
                    full = os.path.join(root, fname)
                    if os.path.exists(full):
                        s = parse_loss_summary(full)
                        s["species"] = sp
                        s["svp"] = svp_str
                        rows.append(s)
                        break
    return pd.DataFrame(rows)


# ==== LOSS TRAJECTORY OVER ITERATIONS =============================

def load_loss_trajectory(checkpoints_dir: str,
                         sample_every: int = 10,
                         max_step: int | None = None,
                         cond_indices: list[int] | None = None) -> pd.DataFrame:
    """Concatenate per-step ``Losses_step_*.tsv`` files into a long DataFrame.

    Each per-step file has rows for each of the 24 conditions; we aggregate
    to median / p10 / p90 of Total_Loss, Data_Loss, Mass_Loss per step.

    Parameters
    ----------
    checkpoints_dir : str
        Path to a ``ml/checkpoints/`` directory containing
        ``Losses_step_*.tsv`` files.
    sample_every : int
        Read every Nth file by step number (default 10) to keep things fast.
    max_step : int, optional
        Skip files past this step number (useful when stale checkpoints
        from prior runs polluted the directory).
    cond_indices : list[int], optional
        Restrict the aggregation to these condition row indices (e.g., the
        11 Leaf rows). If None, aggregate across all rows in each TSV.
    """
    pattern = re.compile(r"Losses_step_(\d+)\.tsv$")
    candidates = []
    for fn in os.listdir(checkpoints_dir):
        m = pattern.match(fn)
        if m:
            step = int(m.group(1))
            if max_step is not None and step > max_step:
                continue
            candidates.append((step, fn))
    candidates.sort(key=lambda x: x[0])
    if sample_every > 1:
        candidates = candidates[::sample_every]

    rows = []
    for step, fn in candidates:
        df = pd.read_csv(os.path.join(checkpoints_dir, fn), sep="\t")
        if cond_indices is not None:
            valid = [i for i in cond_indices if i < len(df)]
            if not valid:
                continue
            df = df.iloc[valid]
        rows.append({
            "step":          step,
            "n_conditions":  len(df),
            "Total_median":  df["Total_Loss"].median(),
            "Total_p5":      float(np.percentile(df["Total_Loss"], 5)),
            "Total_p10":     float(np.percentile(df["Total_Loss"], 10)),
            "Total_p90":     float(np.percentile(df["Total_Loss"], 90)),
            "Total_p95":     float(np.percentile(df["Total_Loss"], 95)),
            "Data_median":   df["Data_Loss"].median(),
            "Mass_median":   df["Mass_Loss"].median(),
        })
    return pd.DataFrame(rows)


def leaf_condition_indices(species: str) -> list[int]:
    """Return row indices of Leaf conditions for ``species`` from the
    fresh ``training.npz`` (both Control and FeLim)."""
    npz_path = fresh_training_npz(species)
    if npz_path is None:
        return []
    data = np.load(npz_path, allow_pickle=True)
    if "treatments" not in data.files:
        return []
    names = [n.decode("utf-8") if isinstance(n, bytes) else str(n)
             for n in data["treatments"]]
    return [i for i, n in enumerate(names)
            if "Leaf_Control" in n or "Leaf_FeLim" in n]


# ==== CONDITION LABEL PARSING =====================================

def cond_to_tissue_treatment_tp(label: str) -> tuple[str, str, str]:
    """Split ``"Leaf_FeLim_7d"`` into ``("Leaf", "FeLim", "7d")``."""
    parts = label.split("_")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return label, "", ""
