from pathlib import Path
import os
import sys

from pathlib import Path

# Sets project specif information for automatic processing of transcriptome
class Parameters_ML_QPSI: 
    def __init__(self, project_root, spc="Poplar"): 
        self.spc = spc # "Sorghum" "Poplar"
        
        # Expreimental data specifics
        self.time_stamp = 'all'
        self.other_colm_val = 'Leaf'
        self.time_points = ["02d", "04d", "07d", "14d", "21d"]
        self.treatments = ['Control', 'FeLim', 'FeEX', 'ZnLim', 'ZnEx']

        if self.time_stamp == 'all':
            self.treatments = [trmt+"_"+tp for trmt in self.treatments for tp in self.time_points]

        self.ctrl_trmt = 'CTL'
        self.other_colm = 'tissue'
        self.value_col = 'value'
        self.useRelab = True
        self.trmt_column = 'treatment'

        # Simulation output folder
        self.expFolder = "July3_maxControl_misexRelab_modelGenesCap_loopless_QPSI/"
        self.expFolder = "Aug7_maxControl_misexRelab_modelGenesCap_loopless_QPSI/"

        self.results_folder = os.path.join(project_root, "Dataset_input", self.expFolder)
        if not os.path.exists(self.results_folder):
            print("Unable to create folder: ")
            print(self.results_folder)
            # self.error = True
  
        VbfFileName = f"{self.spc}_complexFix_{self.other_colm}_{self.time_stamp}_restrMedia_Vbf_maxCtrl.csv"
        self.VbfFile = os.path.join(self.results_folder, VbfFileName)

        ## Relative abundance
        self.relab_scores_file = os.path.join(self.results_folder, f"{self.spc}_relab_rxn_scores_tmm.csv")
        ## Objective abundance
        self.scores_file = os.path.join(self.results_folder, f"{self.spc}_objective_abundance_{self.ctrl_trmt}.tsv")

        ### ---------- Common files: models and media
        self.model_folder = os.path.join(project_root, "Dataset_input", "models_media")

        # Species specific information and model file path  
        fName = 'ptrich_4.1' if self.spc == "Poplar" else 'sbicolor_3.1.1'
        mDate = '250512' if self.spc == "Poplar" else '250617'
        modelName = f"{fName}_plastid_Thylakoid_Reconstruction_ComplexFix_RevFix3_{mDate}.json"
        self.model_path = os.path.join(self.model_folder, modelName)

        ## Media JSON file needed by the cobrakbase converter   
        self.media_path = os.path.join(self.model_folder, "PlantPlastidialAutotrophicMedia_noATP_noADP.json")

        ## Media CSV file needed by the ML simulator
        self.mediumFile = os.path.join(self.model_folder, 'plastidial_model_duplicated_restricted_media_noATP_noADP_noP')

        ## FVA fluxes
        self.fluxes_file = os.path.join(self.model_folder, "Loopless_RevFix3_FVA_Output_Poplar_plastid.tsv")

        ## ML model 
        self.dataset_file = os.path.join(project_root, 'Dataset_model', f"{self.spc}_{str(len(self.treatments))}_{self.other_colm}_{self.time_stamp}_complexFix_loopless")

class Parameters_frz: 
    def __init__(self): 
        self.spc = "athaliana" 
        self.fName = 'Athaliana'
        self.mDate = '070224'
        self.time_points = ["ZT1", "ZT5", "ZT9", "ZT13", "ZT17", "ZT21"]

        self.cobraname = f'{self.fName}_plastid_Thylakoid_Reconstruction_ComplexFix_RevFix3_{self.mDate}_duplicated'
        
        

        self.mediumname = 'plastidial_model_duplicated_restricted_media_noATP_noADP_noP'

        self.time_stamp = 'all' #'ZT9'
        self.other_colm = 'genotype'
        self.value_col = 'value'
        self.ctrl_trmt = 'CTL'
        self.trmt_column = 'treatment'
        self.other_colm_value = 'TSU' # 'TSU', 'C24'
        self.treatments = ['CTL', 'FRZ']

        self.useRelab = False

        if self.time_stamp == 'all':
            self.treatments = [trmt+"_"+tp for trmt in self.treatments for tp in self.time_points]
        
                  # athaliana_complexFix_TSU_all_noADP_Vbf_maxCtrl_fullmodel
        self.Vbfname = f"athaliana_complexFix_{self.other_colm}_{self.time_stamp}_noADP_Vbf_maxCtrl_fullmodel.csv"
        
        self.expFolder = "July3_maxControl_misexRelab_modelGenesCap_loopless_Atha/"

        self.model_path = f"{cobraname}.json"

        self.media_path = "Dataset_input/PlantPlastidialAutotrophicMedia_noATP_noADP.json"

        self.remove_med_full = ["EX_cpd00067_e0_i", "EX_cpd00007_e0_i", "EX_cpd00008_e0_i", "EX_cpd00008_e0_o", "EX_cpd11632_e0_o", "EX_cpd00048_e0_o", "EX_cpd00013_e0_o", 'EX_cpd00073_e0_o', 'EX_cpd00073_e0_i', "EX_cpd00011_e0_o", "EX_cpd00001_e0_o", "EX_cpd00002_e0_o", 'EX_cpd00005_e0_o', 'EX_cpd00006_e0_i', 'EX_cpd00009_e0_o', 'EX_cpd00254_e0_o', 'EX_cpd10515_e0_o', 'EX_cpd11624_e0_i', 'EX_cpd11624_e0_o', 'EX_cpd00098_e0_i', 'EX_cpd00098_e0_o',  'EX_cpd27368_e0_i', 'EX_cpd27368_e0_o',  'EX_cpd00204_e0_i', 'EX_cpd00075_e0_i', 'EX_cpd00075_e0_o', 'EX_cpd00076_e0_i', 'EX_cpd00076_e0_o',  'EX_cpd00209_e0_i', 'EX_cpd00209_e0_o', "EX_cpd00205_e0_o", "EX_cpd00099_e0_o"]

class Parameters_ML_ColdResponse: 
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