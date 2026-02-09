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

# Save the original stdout and stderr
original_stdout = sys.stdout
original_stderr = sys.stderr

try:
	# 1. Redirect stdout/stderr to the null device
	sys.stdout = open(os.devnull, 'w')
	sys.stderr = open(os.devnull, 'w')
	
	# 2. Place all the noisy imports here
	from modelseedpy import FlexibleBiomassPkg

	from parameters import *
	from cobrakbase.core.kbasefba.fbamodel_from_cobra import CobraModelConverter as cmc
	from cobra.flux_analysis import flux_variability_analysis as fva
	from cobra.io import write_sbml_model
	import Library.generateBioFeasibleFlux as gvbf
	import prepare_model_duplication as pdm

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
	vbf_parameters = Parameters_VBF()
	if(genotype is not None):
		vbf_parameters.spc = genotype
		vbf_parameters.other_colm_value = genotype
		project_root = "-".join(["projects/cold-response",genotype.lower(),'251206'])+'/'
		vbf_parameters.projectFolder=project_root
		vbf_parameters.results_folder = f"{vbf_parameters.projectFolder}integration-results/"
		vbf_parameters.model_path = f"{vbf_parameters.projectFolder}inputs/{vbf_parameters.cobraname}.json"
		vbf_parameters.media_path = f"{vbf_parameters.projectFolder}inputs/{vbf_parameters.mediumname}.json"
		vbf_parameters.mediaFile  = f"{vbf_parameters.projectFolder}inputs/{vbf_parameters.mediumname}"
		vbf_parameters.scores_file = os.path.join(vbf_parameters.results_folder, f"{vbf_parameters.spc}_objective_abundance_{vbf_parameters.ctrl_trmt}.tsv")
		vbf_parameters.VbfFile = os.path.join(vbf_parameters.results_folder, vbf_parameters.Vbfname)
		vbf_parameters.dataset_file = os.path.join(vbf_parameters.projectFolder, 'model', f"{vbf_parameters.spc}_{str(len(vbf_parameters.treatments))}_{vbf_parameters.other_colm}_{vbf_parameters.time_stamp}_complexFix_loopless")
		vbf_parameters.ml_result_folder = f"{vbf_parameters.projectFolder}ml_results/"

	# generate COBRA model using cobrakbase
	model = pdm.generateCobraModel(vbf_parameters.model_path, vbf_parameters.media_path)

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

	# Here we're assuming that the biomass reaction is the first reaction with an objective coefficient
	biomass_reactions = [rxn for rxn in model.reactions if rxn.objective_coefficient != 0]
	biomass_reaction = biomass_reactions[0]

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
	dup_model = pdm.prepare_model_duplication(model,media_path=vbf_parameters.media_path)

	# The original code for duplicating the reversible reactions in the model
	# doesn't duplicate the GPR, so we fix that here
	for r in dup_model.reactions:
		if r.id.endswith('_r'):
			r.gene_reaction_rule = dup_model.reactions.get_by_id(r.id[:-2]+'_f').gene_reaction_rule
			r.update_genes_from_gpr()

	# Compute max fva scores with flexible biomass
	# Using flexible biomass increases the range of fluxes that are possible
	# within the core metabolic network responsible for biosynthesizing the biomass
	#
	# However, I don't want to flexible biomass (a special set of exchange reactions and constraints)
	# in the model when I run the metabolic optimization, so I make a deep copy of it first
	fb_dup_model = copy.deepcopy(dup_model)
	print(f"Adding flexible biomass ...")
	fbp = FlexibleBiomassPkg(fb_dup_model)
	fbp.build_package({"bio_rxn_id":biomass_reaction.id})
	fva_result=fva(fb_dup_model,fraction_of_optimum=0.75,pfba_factor=1.2,processes=1)

	# Need to print to project results but I leave out the exchange reactions
	# from the flexible biomass
	with open(f'{vbf_parameters.results_folder}fva.tsv','w') as ofh:
		ofh.write("reaction\tmax\n")
		for rxn, result in fva_result.iterrows():
			if rxn in dup_model.reactions:
				ofh.write(f"{rxn}\t{result['maximum']:.6f}\n")

	gvbf.generate_Vbf(vbf_parameters)
