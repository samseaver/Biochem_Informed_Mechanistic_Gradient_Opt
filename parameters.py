from pathlib import Path
import os
import sys

from pathlib import Path
project_root = str(Path(__file__).resolve()).split('src')[0]
sys.path.append(project_root)

# Sets project specif information for automatic processing of transcriptome
class Parameters_VBF: 
    def __init__(self):
        self.project_folder = "projects/qpsi-sorghum-260206/"
        self.model_name = "Sbicolor-v5.1-reconstruction_fixed.json"
        self.model_path = f"{self.project_folder}inputs/{self.model_name}.json"

        self.medium = "PlantAutotrophicMedia"
        self.media_file  = f"{self.projectFolder}inputs/{self.medium}"
        self.media_path = f"{self.media_file}.json"

        pass
    
class Parameters_ML: 
    def __init__(self): 
        self.spc = "TSU"
        self.fName = 'Athaliana'
        self.mDate = '070224'
        self.error = False
        self.time_points = ["1", "5", "9", "13", "17", "21"]
        self.expFolder = "projects/cold-response/"
        self.projectFolder = "projects/cold-response-tsu-251206/"

        self.results_folder = f"{self.projectFolder}integration-results/"
        self.cobraname = "athaliana-model-251206"
        self.model_path = f"{self.projectFolder}inputs/{self.cobraname}.json"
        
        self.mediumname = "PlantAutotrophicMedia"
        self.media_path = f"{self.projectFolder}inputs/{self.mediumname}.json"
        self.mediaFile  = f"{self.projectFolder}inputs/{self.mediumname}"

        self.ctrl_trmt = 'CTL'
        self.scores_file = os.path.join(self.results_folder, f"{self.spc}_objective_abundance_{self.ctrl_trmt}.tsv")
        self.time_stamp = 'all' # 'ZT9'
        self.other_colm = 'genotype'
        self.value_col = 'mean_value'
        self.trmt_column = 'treatment'
        self.other_colm_value = 'TSU' # 'TSU', 'C24'
        self.treatments = ['CTL', 'FRZ']

        self.useRelab = False

        if self.time_stamp == 'all':
            self.treatments = [trmt+"_"+tp for trmt in self.treatments for tp in self.time_points]
        
        self.Vbfname = f"vbf-{self.other_colm}-{self.time_stamp}.tsv"
        self.VbfFile = os.path.join(self.results_folder, self.Vbfname)
        
        self.dataset_file = os.path.join(self.projectFolder, 'model', f"{self.spc}_{str(len(self.treatments))}_{self.other_colm}_{self.time_stamp}_complexFix_loopless")

        self.ml_result_folder = f"{self.projectFolder}ml_results/"
