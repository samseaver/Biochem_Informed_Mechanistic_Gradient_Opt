import sys
import os
from pathlib import Path
project_root = Path(__file__).resolve().parent

#### Replace by the path to module 
module = '/Users/selalaoui/Projects/QPSI_project/paper_repo/RNASeq_Enzyme_Abundance/src/reaction_scores'
if module not in sys.path:
    sys.path.append(module)

module = '/Users/selalaoui/Projects/QPSI_project/paper_repo/RNASeq_Enzyme_Abundance/src'
if module not in sys.path:
    sys.path.append(module)

import computeScoresAndPredictions as csp
from util.parameters import Parameters_QPSI

from parameters import *

import Library.generateBioFeasibleFlux as gvbf
import Duplicate_Model_triple_FVA as dm_FVA
import Build_Dataset as bd
import Build_Model_MM as bmm




if __name__ == '__main__':
	rs_param = Parameters_QPSI()
	ml_param = Parameters_ML_QPSI(project_root)

	# rs_param.objSaveTo   = ml_param.scores_file
	# rs_param.relabSaveTo = ml_param.relab_scores_file
	rs_param.results_folder = ml_param.results_folder

	# Compute reaction scores -- needed to compute Vbf 
	if (not os.path.exists(ml_param.scores_file)) :
		csp.generate_reactionScores(rs_param)
	

	# Duplicate model, which will also be used for Vbf computation
	remove_med_plastid = ["EX_cpd00067_e0_i", "EX_cpd00007_e0_i", "EX_cpd00008_e0_i", "EX_cpd00013_e0_o", "EX_cpd00048_e0_o", "EX_cpd11632_e0_o", "EX_cpd00011_e0_o", "EX_cpd00001_e0_o", "EX_cpd00002_e0_o", "EX_cpd00009_e0_o", "EX_cpd00002_e0_i", "EX_cpd00008_e0_o"]

	# remove_med_full = ["EX_cpd00067_e0_i", "EX_cpd00007_e0_i", "EX_cpd00008_e0_i", "EX_cpd00008_e0_o", "EX_cpd11632_e0_o", "EX_cpd00048_e0_o", "EX_cpd00013_e0_o", 'EX_cpd00073_e0_o', 'EX_cpd00073_e0_i', "EX_cpd00011_e0_o", "EX_cpd00001_e0_o", "EX_cpd00002_e0_o", 'EX_cpd00005_e0_o', 'EX_cpd00006_e0_i', 'EX_cpd00009_e0_o', 'EX_cpd00254_e0_o', 'EX_cpd10515_e0_o', 'EX_cpd11624_e0_i', 'EX_cpd11624_e0_o', 'EX_cpd00098_e0_i', 'EX_cpd00098_e0_o',  'EX_cpd27368_e0_i', 'EX_cpd27368_e0_o',  'EX_cpd00204_e0_i', 'EX_cpd00075_e0_i', 'EX_cpd00075_e0_o', 'EX_cpd00076_e0_i', 'EX_cpd00076_e0_o',  'EX_cpd00209_e0_i', 'EX_cpd00209_e0_o', "EX_cpd00205_e0_o", "EX_cpd00099_e0_o"]

	dupModel_path = ml_param.model_path.replace('.json', '_duplicated.xml')
	if not os.path.exists(dupModel_path):
		dm_FVA.run_duplicate_model(ml_param, remove_med_plastid, fluxCoupling=False)

	# Generate Vbf
	if not os.path.exists(ml_param.VbfFile):
		gvbf.generate_Vbf(ml_param)


	# Build detaset for the ML pipeline: 
	#    -> Reads Vbf to set is as the constraint 
	#    -> Reads the duplicated model to generate the matrix 
	bd.build_dataset(ml_param)

	###----- Run ML simulation
	epochs=2.5e6    
	learn_rate=1    # 
	decay_rate=.333 # 

	# initial flux for the simulation 
	# -1: set Exchange reactions to 1000, Vbf for all reaction with a value or Vbf_mean 
	# 0 or above: set starting fluxes to will be set to V0_init
	#           : if V0_init is 0, Exchange reactions are set to 1000
	V0_init=-1

	# penalty on the steady state constraint
	svp=15

	# Which hard constraint to set for modeling 
	# 0: for none
	# 1: for positive flux only 
	# 2: for both positive flux and Vbf 
	hardConst=1

	# set an upper bound constraint on the biomass reaction if 
	#   use_objective is TRUE
	use_objective = False
	biomass_max=False

	bmm.run_simulation(ml_param, epochs=epochs, learn_rate=learn_rate, decay_rate=decay_rate, V0_init=V0_init, svp=svp, hardConst=hardConst, use_objective=use_objective, biomass_max = biomass_max)




