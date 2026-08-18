#!/usr/bin/env python
from pathlib import Path
import os
import sys

project_root = str(Path(__file__).resolve()).split('src')[0]
sys.path.append(project_root)

# --- Global Species Configuration ---
# Toggle the active species here to automatically synchronize all downstream VBF and ML paths
ACTIVE_SPECIES = os.environ.get("BF_SPECIES", "Sorghum")  # edit here, or override with BF_SPECIES=Poplar

if ACTIVE_SPECIES == "Poplar":
    GLOBAL_SPC = "Poplar"
    GLOBAL_PROJECT_FOLDER = "projects/poplar-plastidial/"
    GLOBAL_BASE_MODEL = "plastidial-Ptrichocarpa-v4.1-reconstruction_fixed"
elif ACTIVE_SPECIES == "Sorghum":
    GLOBAL_SPC = "Sorghum"
    GLOBAL_PROJECT_FOLDER = "projects/sorghum-plastidial/"
    GLOBAL_BASE_MODEL = "Sbicolor-v3.1.1-plastidial-reconstruction"
else:
    raise ValueError("Unsupported species selected. Please choose 'Poplar' or 'Sorghum'.")

# Optional project directory override. Set BF_PROJECT to point at a different
# copy of a species project -- for example to run several penalties
# concurrently without them racing on the shared inputs/,
# integration_results/ and ml/training/ files. The directory must be
# self-contained (inputs/ + integration_results/).
#
# Note only sorghum-plastidial/ ships with this repository. Poplar was used in
# the paper but its results are deposited separately; BF_SPECIES=Poplar
# therefore needs a project directory you supply.
_bf_project = os.environ.get("BF_PROJECT")
if _bf_project:
    GLOBAL_PROJECT_FOLDER = _bf_project.rstrip("/") + "/"
# ------------------------------------

# --- Treatment subset filter ---
# Restrict gradient descent to vbf_* columns whose name contains any of these substrings.
# Set to None or an empty tuple to use every treatment found in vbf.tsv.
TREATMENT_FILTERS = ("Control", "FeLim")
# --------------------------------

# --- Gradient-descent sweep + budget ---
# Mass-balance penalty (p) values to run. When predict_bioinformed_flux.py is
# called without --svp it loops over this list in order; each value runs to
# convergence, with per-condition early stopping, into its own
# ml/svp_<value>/ subdirectory.
#
# The default is the single adopted operating point. Extend the list to explore
# the penalty -- the preprint swept [2.0, 1.0, 0.5, 0.1] -- but note the cost:
# every added value is another full training run, and the smaller penalties are
# the slow ones (at p = 0.1 the slowest condition needed 1.67e6 iterations
# against 1.28e6 at p = 2.0). Running four values serially in one process takes
# roughly four times as long as one; the preprint sweep instead launched one
# process per (species, p) with --svp pinned, so the runs went concurrently.
#
# What p buys is a trade-off between the two loss terms: raising it tightens
# mass balance and loosens the transcript fit. See the README.
SVP_VALUES   = [2.0]

EPOCHS       = int(2.5e6)      # full-budget epoch count (the published sweep)
TEST_EPOCHS  = 5000            # used when --test is passed; ~1-2 min per svp
LEARN_RATE   = 5e-2            # gradient-descent learn rate
DECAY_RATE   = 0.333           # momentum decay
# --------------------------------


# Sets project specific information for automatic processing of transcriptome
class Parameters_VBF: 
    def __init__(self):
        self.spc = GLOBAL_SPC
        self.project_folder = GLOBAL_PROJECT_FOLDER
        self.model_name = GLOBAL_BASE_MODEL

        self.model_path = f"{self.project_folder}inputs/{self.model_name}.json"

        self.medium = "PlantAutotrophicMedia"
        self.media_file  = f"{self.project_folder}inputs/{self.medium}"
        self.media_path = f"{self.media_file}.json"

        self.results_folder = f"{self.project_folder}integration_results/"
        # Objective reaction scores (r_s, un-normalized) per condition ship inside
        # the project's inputs/ directory alongside the model JSON / media. This is
        # the objective reaction-scores file from the RNASeq pipeline
        # ({spc}_reaction_scores.tsv), NOT the molar-fractions file.
        self.scores_folder = f"{self.project_folder}inputs/"
        self.scores_file = os.path.join(self.scores_folder, f"{self.spc}_reaction_scores.tsv")

        self.ctrl_trmt = 'Control'
        self.time_stamp = 'all'
        
        # Objective reaction score (r_s), un-normalized (no plastid-pool division).
        self.value_col = "reaction_score"
        self.trmt_column = 'condition'

        self.useRelab = False

class Parameters_ML: 
    def __init__(self):
        self.spc = GLOBAL_SPC
        self.project_folder = GLOBAL_PROJECT_FOLDER
        
        # ML models utilize the duplicated reaction format
        self.model_name = f"{GLOBAL_BASE_MODEL}_dup"
        
        self.model_path = f"{self.project_folder}inputs/{self.model_name}.xml"

        self.medium = "PlantAutotrophicMedia"
        self.media_file  = f"{self.project_folder}inputs/{self.medium}"

        self.integration_folder = f"{self.project_folder}integration_results/"
        self.vbf_file = f"{self.integration_folder}vbf.tsv"

        self.ml_folder = f"{self.project_folder}ml/"
        self.training_folder = f"{self.ml_folder}training/"

        # output_dir is set per-run by predict_bioinformed_flux.py to point
        # to the per-svp subdirectory (see svp_subdir / test_svp_subdir below).
        # Build_Model_Wrapper picks this up via getattr(..., "output_dir", ml_folder)
        # so it falls back gracefully if no svp loop is being used.
        self.output_dir = self.ml_folder

    # ---- per-svp output paths ----
    def svp_subdir(self, svp):
        """Per-svp output directory under ml/. e.g. ml/svp_1.0/"""
        return f"{self.ml_folder}svp_{svp}/"

    def test_svp_subdir(self, svp):
        """Test-mode per-svp output directory, isolated from production."""
        return f"{self.ml_folder}test/svp_{svp}/"