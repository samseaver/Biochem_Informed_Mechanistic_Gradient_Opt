import warnings
warnings.simplefilter(action='ignore', category=Warning)

import sys
import os
import copy
import json
import pathlib
from datetime import datetime

sys.path.append('/scratch/seaver/Collaborations/Greenham_UMinn/ModelSEEDPy')
sys.path.append('/scratch/seaver/Collaborations/Greenham_UMinn/cobrakbase')

#### Replace by the path to module 
module = '../RNASeq_Enzyme_Abundance/src'
if module not in sys.path:
	sys.path.append(module)

module += '/reaction_scores'
if module not in sys.path:
	sys.path.append(module)

# Save the original stdout and stderr
original_stdout = sys.stdout
original_stderr = sys.stderr

try:
	# 1. Redirect stdout/stderr to the null device
	sys.stdout = open(os.devnull, 'w')
	sys.stderr = open(os.devnull, 'w')
	
	# 2. Place all the noisy imports here
	# These imports/initializations will now run silently

	# the next three lines are importaed from ../RNASeq_Enzyme_Abundance
	import computeScoresAndPredictions as csp
	from modelseedpy import FlexibleBiomassPkg
	from util.parameters import Parameters_QPSI
	from util.parameters import Parameters_ColdResponse

	from parameters import *

	from cobrakbase.core.kbasefba.fbamodel_from_cobra import CobraModelConverter as cmc
	from cobra.flux_analysis import flux_variability_analysis as fva
	from cobra.io import write_sbml_model
	import Library.generateBioFeasibleFlux as gvbf
	import prepare_model_duplication as pdm
	import flux_coupling_analysis
	import Build_Dataset as bd
	import Build_Model_MM as bmm

finally:
	# 3. ALWAYS restore the original stdout/stderr
	sys.stdout.close()
	sys.stderr.close()
	sys.stdout = original_stdout
	sys.stderr = original_stderr

if __name__ == '__main__':

	allowed_genotypes = {'TSU', 'C24'}
	genotype = None
	if len(sys.argv) == 2:
		genotype = sys.argv[1]

		if genotype not in allowed_genotypes:
			print(f"❌ Invalid value '{genotype}'. Argument must be TSU or C24.")
			sys.exit(1)

	# Project parameters
	ml_param = Parameters_ML_ColdResponse()
	if(genotype is not None):
		ml_param.spc = genotype
		ml_param.other_colm_value = genotype
		project_root = "-".join(["projects/cold-response",genotype.lower(),'251206'])+'/'
		ml_param.projectFolder=project_root
		ml_param.results_folder = f"{ml_param.projectFolder}integration-results/"
		ml_param.model_path = f"{ml_param.projectFolder}inputs/{ml_param.cobraname}.json"
		ml_param.media_path = f"{ml_param.projectFolder}inputs/{ml_param.mediumname}.json"
		ml_param.mediaFile  = f"{ml_param.projectFolder}inputs/{ml_param.mediumname}"
		ml_param.scores_file = os.path.join(ml_param.results_folder, f"{ml_param.spc}_objective_abundance_{ml_param.ctrl_trmt}.tsv")
		ml_param.VbfFile = os.path.join(ml_param.results_folder, ml_param.Vbfname)
		ml_param.dataset_file = os.path.join(ml_param.projectFolder, 'model', f"{ml_param.spc}_{str(len(ml_param.treatments))}_{ml_param.other_colm}_{ml_param.time_stamp}_complexFix_loopless")
		ml_param.ml_result_folder = f"{ml_param.projectFolder}ml_results/"

	# generate COBRA model using cobrakbase
	model = pdm.generateCobraModel(ml_param.model_path, ml_param.media_path)

	# Find and remove inactive compartments.
	# The duplication of the model means that an inactive compartment 
	# with reversible reactions can exhibit a lot of loops
	# so we remove the compartments entirely
	model = pdm.find_remove_inactive_compartments(model)
	
	# There are some dead-end transporters in the model which
	# also do not have genes associated with them so their Vbf
	# is set to a high default value but they are inherently blocked
	# so we remove them to reduce the size of the solution
	model = pdm.find_remove_blocked_transporters(model)

	# Find and remove NGAM components from biomass reaction
	# This is key because we are not optimizing biomass when we run
	# our approach, and the presence of these components (which consume ATP)
	# otherwise blocks biomass generation (energetically)
	biomass_reaction = model.reactions.get_by_id('bio1_biomass')

	# This dictionary of metabolites and the stoichiometry is taken from the
	# PlantSEED metabolic model. If you have GAM encoded in your biomass
	# you can update these values to match
	gam_stoich = 0.0
	gam_metabolite_ids = {
		'cpd00002_c0': 1.0, # ATP
		'cpd00001_c0': 1.0, # H2O
		'cpd00008_c0': -1.0, # ADP
		'cpd00009_c0': -1.0, # Pi
		'cpd00067_c0': -1.0  # Proton
	}

	# Metabolite IDs to check
	atp_id = 'cpd00002_c0'
	adp_id = 'cpd00008_c0'
	
	atp = model.metabolites.get_by_id(atp_id)
	adp = model.metabolites.get_by_id(adp_id)

	if(atp in biomass_reaction.metabolites and adp in biomass_reaction.metabolites):
		gam_stoich = biomass_reaction.metabolites[adp]
		if(gam_stoich > 0):
			print("GAM is present in the model: ",adp,gam_stoich)

	if(gam_stoich > 0):
		biomass_updates = dict()
		# This assumes that all metabolites involved in ATP hydrolysis are present
		# and at the level required for GAM to be balanced
		for met_id, gam_coeff in gam_metabolite_ids.items():
			met = model.metabolites.get_by_id(met_id)
			met_stoich = gam_coeff * gam_stoich
			biomass_updates[met] = met_stoich
		biomass_reaction.add_metabolites(biomass_updates)

		# to double-check
		for met_id in gam_metabolite_ids:
			met = model.metabolites.get_by_id(met_id)
			if(met in biomass_reaction.metabolites):
				# print("\t",met_id,biomass_reaction.get_coefficient(met))
				pass

	# The ML approach only works if all reactions go from left to right
	# We run the approach to reverse any reactions that go from right to left
	# Like-wise the reversible reactions are duplicated and the copy is reversed
	#
	# Loading the model using Cobra will add exchange reactions for metabolites
	# in the extracellular compartment, and the model may contain more than in
	# the media, that are unconstrained, so we use the media to strip away
	# unwanted exchange reactions
	dup_model = pdm.prepare_model_duplication(model,media_path=ml_param.media_path)

	# The original code for duplicating the reversible reactions in the model
	# doesn't duplicate the GPR, so we fix that here
	for r in dup_model.reactions:
		if r.id.endswith('_r'):
			r.gene_reaction_rule = dup_model.reactions.get_by_id(r.id[:-2]+'_f').gene_reaction_rule
			r.update_genes_from_gpr()

	# Compute reaction scores
	rs_param = Parameters_ColdResponse()

	# Print duplication model into new folder to be read
	# by package for generating reaction scores
	# dup_folder = os.path.dirname(ml_param.model_path)+"/duplicate/"
	# dup_file = os.path.basename(ml_param.model_path).replace(".json","_dup.json")
	# os.makedirs(dup_folder, exist_ok=True)
	# print("Writing duplicated model: ",dup_file)
	# print("To folder: ",dup_folder)
	# with open(dup_folder+dup_file,'w') as fh:
	# 	kbase_model = cmc(dup_model).build()
	# 	kbase_data = kbase_model.get_data()
	# 	json.dump(kbase_data,fh,indent=2)

	# rs_param.json_files_folder = dup_folder
	rs_param.results_folder = ml_param.results_folder
	rs_param.json_files_folder = ml_param.projectFolder+'inputs/'
	csp.generate_reactionScores(rs_param,project_species=[genotype],verbose=False)

	# Compute max fva scores with flexible biomass
	# Using flexible biomass increases the range of fluxes that are possible
	# within the core metabolic network responsible for biosynthesizing the biomass
	#
	# However, I don't want to flexible biomass (a special set of exchange reactions and constraints)
	# in the model when I run the metabolic optimization, so I make a deep copy of it first
	fb_dup_model = copy.deepcopy(dup_model)
	print(f"Adding flexible biomass ...")
	fbp = FlexibleBiomassPkg(fb_dup_model)
	fbp.build_package({"bio_rxn_id":"bio1_biomass"})
	fva_result=fva(fb_dup_model,fraction_of_optimum=0.75,pfba_factor=1.2,processes=1)

	# Need to print to project results but I leave out the exchange reactions
	# from the flexible biomass
	with open(f'{rs_param.results_folder}fva.tsv','w') as ofh:
		ofh.write("reaction\tmax\n")
		for rxn, result in fva_result.iterrows():
			if rxn in dup_model.reactions:
				ofh.write(f"{rxn}\t{result['maximum']:.6f}\n")

	gvbf.generate_Vbf(ml_param,model=dup_model)

	# prints the duplicated model to xml and updates the parameter
	ml_param.model_path = ml_param.model_path.replace(".json","_dup.xml")
	write_sbml_model(dup_model, ml_param.model_path)

	# Build detaset for the ML pipeline: 
	#    -> Reads Vbf to set is as the constraint 
	#    -> Reads the duplicated model to generate the matrix 
	# This requires a specially formatted media file for the boundary reactions in the model
	exchange_list = [r.id for r in dup_model.exchanges if "EX" in r.id]
	with open(ml_param.mediaFile+'.csv','w') as fh:
		print("writing exchanges to training media")
		fh.write(f"name,{",".join(exchange_list)}\n")
		fh.write(f"level{',2'*len(exchange_list)}\n")
		fh.write(f"max_value{',1000'*len(exchange_list)}\n")
		fh.write(f"ratio_drawing{','*len(exchange_list)}\n")
	bd.build_dataset(ml_param)

	###----- Run ML simulation
	epochs=1 #2.5e6
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
	hardConst=2

	# set an upper bound constraint on the biomass reaction if 
	#   use_objective is TRUE
	use_objective=False
	biomass_max=False

	# bmm.run_simulation(ml_param, epochs=epochs, learn_rate=learn_rate, decay_rate=decay_rate, V0_init=V0_init, svp=svp, hardConst=hardConst, use_objective=use_objective, biomass_max = biomass_max)