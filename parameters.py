#!/usr/bin/env python
from pathlib import Path
import os
import sys

project_root = str(Path(__file__).resolve()).split('src')[0]
sys.path.append(project_root)

# --- Global Species Configuration ---
# Toggle the active species here to automatically synchronize all downstream VBF and ML paths
ACTIVE_SPECIES = "Poplar"  # Options: "Poplar" or "Sorghum"

if ACTIVE_SPECIES == "Poplar":
    GLOBAL_SPC = "Poplar"
    GLOBAL_PROJECT_FOLDER = "projects/qpsi-260406-plastid-poplar/"
    GLOBAL_BASE_MODEL = "plastidial-Ptrichocarpa-v4.1-reconstruction_fixed"
elif ACTIVE_SPECIES == "Sorghum":
    GLOBAL_SPC = "Sorghum"
    GLOBAL_PROJECT_FOLDER = "projects/qpsi-260406-plastid-sorghum/"
    GLOBAL_BASE_MODEL = "Sbicolor-v3.1.1-plastidial-reconstruction"
else:
    raise ValueError("Unsupported species selected. Please choose 'Poplar' or 'Sorghum'.")
# ------------------------------------


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
        self.scores_folder = "/Users/seaver/Seaver_Lab/Git_Repos/RNASeq_Enzyme_Abundance/projects/qpsi-plastidial/integration_results/"
        self.scores_file = os.path.join(self.scores_folder, f"{self.spc}_reaction_molar_fractions.tsv")

        self.ctrl_trmt = 'Control'
        self.time_stamp = 'all'
        
        self.value_col = 'relative_reaction_score'
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