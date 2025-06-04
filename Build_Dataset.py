import os
import sys

DIRECTORY = './'
font = 'arial'

# printing the working directory files. One can check you see the same folders and files as in the git webpage.
print(os.listdir(DIRECTORY))

from Library.Build_Dataset import *

seed = 10
np.random.seed(seed=seed)  # seed for random number generator


spc = 'Poplar'
# cobraname = 'sbicolor_3.1.1_plastid_Thylakoid_Reconstruction_ComplexFix_070224_duplicated'
cobraname = 'ptrich_4.1_plastid_Thylakoid_Reconstruction_ComplexFix_RevFix3_250512_duplicated'
# cobraname = 'sbicolor_plastidial_model_duplicated'
mediumname = 'plastidial_model_duplicated_restricted_media_noATP_noADP_noP'


if 'atha' in spc:
    time_stamp = 'all' #'ZT9'
    time_stamp = 'ZT9'
    other_colm = 'TSU' # 'TSU', 'C24'
    if time_stamp == 'all':
        trmts = ['CTL', 'FRZ']
        time_points = ["ZT1", "ZT5", "ZT9", "ZT13", "ZT17", "ZT21"]
        treatments = [trmt+"_"+tp for trmt in trmts for tp in time_points]
        # treatments = ["Control_ZT1", "Control_ZT5", "Control_ZT9", "Control_ZT13", "Control_ZT17",
        #               "Control_ZT21", "Freeze_ZT1", "Freeze_ZT5", "Freeze_ZT9", "Freeze_ZT13",
        #               "Freeze_ZT17", "Freeze_ZT21"]
    else:
        trmts = ['CTL', 'FRZ']
        treatments = ['CTL', 'FRZ'] # ['Control', 'Freeze']

              # athaliana_complexFix_TSU_all_noADP_Vbf_maxCtrl_fullmodel
    Vbfname = f"athaliana_complexFix_{other_colm}_{time_stamp}_noADP_Vbf_maxCtrl_fullmodel.csv"
    expFolder = ""

else:
    time_stamp = 'all'
    other_colm = 'Leaf'
    time_points = ["02d", "04d", "07d", "14d", "21d"]
    trmts = ['Control', 'FeLim', 'FeEX', 'ZnLim', 'ZnEx']
    treatments = [trmt+"_"+tp for trmt in trmts for tp in time_points]
    Vbfname = f"{spc}_complexFix_{other_colm}_{time_stamp}_restrMedia_Vbf_maxCtrl.csv"
    expFolder = "May14_maxControl_misexRelab_modelGenesCap_loopless/"

mediumbound = 'UB' # Exact bound (EB) or upper bound (UB)
method = 'Vbf' #'FBA' # FBA, pFBA or EXP, Vbf, Vbf_Wt
reduce = False # Set at True if you want to reduce the model

size = len(treatments)
measure = []
# rfl = ['rxn00018_d0', 'rxn00018_d0_f', 'rxn00018_d0_r']
rfl = []
verbose = True
# End of What you can change

# Run cobra
Vbffile    = DIRECTORY+'Dataset_input/'+expFolder+Vbfname
cobrafile  = DIRECTORY+'Dataset_input/'+expFolder+cobraname
mediumfile = DIRECTORY+'Dataset_input/'+expFolder+mediumname
parameter  = TrainingSet(cobraname=cobrafile,
                        mediumname=mediumfile, mediumbound=mediumbound,
                        method=method,objective=[],
                        measure=measure, Vbfname=Vbffile,
                        restrictedFittingList = rfl, treatments=treatments, verbose=verbose)
# Note: Leaving objective and mesaure as empty lists sets the default
# objective reaction of the SBML model as the objective reaction
# and the measure (Y) as this objective reaction.
parameter.get(sample_size=size, treatments=treatments, verbose=verbose)

# np.savetxt("Result/AfterGetTempPout.tsv", parameter.Pout, delimiter='\t')
# Saving file
# print(abc)
trainingfile  = DIRECTORY+'Dataset_model/'+mediumname+'_'+parameter.mediumbound+'_'+str(size)+'_'+spc+'_'+other_colm+'_'+time_stamp+'_complexFix'
parameter.save(trainingfile, reduce=reduce, verbose=verbose)
# np.savetxt("Result/AfterSaveTempPout.tsv", parameter.Pout, delimiter='\t')
# Verifying
parameter = TrainingSet()
print("All Saved .. now loading")
parameter.load(trainingfile)
print("printing ... ")
parameter.printout()
# np.savetxt("Result/AfterLoadTempPout.tsv", parameter.Pout, delimiter='\t')

# trainingfile  = DIRECTORY+'Dataset_model/e_coli_core_UB_1000'
# parameter.load(trainingfile)
# parameter.printout()
# %% markdown
# Using iML1515, alongside an experimental file that is guiding the generation of the training set (instead of the usual 'mediumname' we have a 'expname' file which contains all experimental media compositions, in order to obtain a training sets of all biologically relevant flux distributions according to these compositions. Note that we reduce the model in this next cell.
# %% codecell
# Generate training set with E coli iML1515 with FBA simulation
# constrained by experimental file: metabolites in medium are not drawn at
# random but are the same than in the provided training experimental file
# This cell may take several hours to execute! Avoid running this in Colab
