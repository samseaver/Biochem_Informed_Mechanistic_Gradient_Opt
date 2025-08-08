import warnings
warnings.simplefilter(action='ignore', category=Warning)
import os
import sys
import time
import datetime


from modelseedpy import FlexibleBiomassPkg
from cobra.flux_analysis import flux_variability_analysis as fva
from cobrakbase.core.kbase_object_factory import KBaseObjectFactory

from cobra.io import read_sbml_model

from Library.Duplicate_Model import *

def runFluxCoupling(model, dup_co_model, model_path):
	models = {'org': model, 'dup': dup_co_model}
	for type, mdl in models.items():
		start = time.time()
		print(f"Running flux coupling for {type} model ...")
		model_type = 'plastid' if 'plastid' in model_path.lower() else 'full'
		model_type = model_type+"_"+type

		mdl.reactions.get_by_id("bio1_biomass").lower_bound=0.5
		fva_rxns_explore = list(mdl.reactions)

		flux_dict = dict()
		fva_dict = dict()
		for reaction in mdl.reactions:
			if('bio1' in reaction.id):
				continue

			reaction.objective_coefficient=1.0

			flux_df = mdl.optimize().fluxes
			flux_dict[reaction.id]=flux_df

			fva_df = fva(mdl, fva_rxns_explore, processes=1, fraction_of_optimum=0.8)
			fva_dict[reaction.id]=fva_df

			reaction.objective_coefficient=0.0

			#if(reaction.id == mdl.reactions[10].id):
			#	break

		ofh = open(f'Dataset_input/FVA_Output_{spc}_{model_type}.tsv','w')
		ofh.write("\t".join(["reaction"]+list(fva_dict.keys())+["avg"])+'\n')
		for reaction in mdl.reactions:
			if('bio1' in reaction.id or reaction.id == 'protein_flex'):
				continue

			max_sum=0.0
			count=0.0
			max_list=list()
			for max_flux_rxn in fva_dict:
				if(max_flux_rxn == reaction.id):
					continue

				rxn_fva = fva_dict[max_flux_rxn].loc[fva_dict[max_flux_rxn].index == reaction.id]
				max = rxn_fva.iloc[0]['minimum']

				max_list.append("{:.6f}".format(max))
				max_sum+=max
				count+=1.0

			mean_max = "0.0"
			if(count>0):
				mean_max = "{:.6f}".format(max_sum/count)
			ofh.write("\t".join([reaction.id]+max_list+[mean_max])+'\n')

		# print("new model media ", mdl.medium)
		end = time.time()
		timeD = end - start
		
		print("Time elapsed: ", str(datetime.timedelta(seconds=timeD)))

def generateCobraModel(model_path, media_path):
	print('abc2')
	KBOF = KBaseObjectFactory()
	model = KBOF.build_object_from_file(model_path, "KBaseFBA.FBAModel")
	co_media = KBOF.build_object_from_file(media_path, "KBaseBiochem.Media")
	model.medium = co_media
	return model

def run_duplicate_model(param, remove_med, fluxCoupling=False): 
	spc = param.spc 
	expFolder = param.expFolder
	model_path = param.model_path
	media_path = param.media_path

	# generate COBRA model using cobrakbase
	model = generateCobraModel(model_path, media_path)
	
	# Print model opt solution 
	solution = model.optimize()
	print(solution)

	# Print the cobra model to file
	from cobra.io import write_sbml_model
	write_sbml_model(model, model_path.replace('json', 'xml'))
	print("SBML model has been saved to: ", model_path.replace('json', 'xml'))


	# ## Checking the objective of the model
	# Optional step for exploring how the biomass is encoded. In some models, several biomass reactions are available and one has to make sure using the right one.
	for reac in model.reactions:
	     if "biomass" in reac.id or "BIOMASS" in reac.id:
	        print(reac)


	# ## Screen outflowing and inflowing reactions
	# For each reaction that has different compartments in reactants and products (we call it "transfer reactions"), we annotate the reaction with a suffix "i" for inflowing (None --> e --> p --> c --> m) and "o" for outflowing (m --> c --> p --> e --> None). When the compartment-changing of metabolites is balanced, or not present, we use different suffix: "for" as in forward, designating the default way of the reaction (positive flux) and "rev" as in reverse, designating the opposite way (negative flux). We reverse the products and reactants so that the same reactions happen, ensuring that we have a positive flux for all reactions.
	#
	# To do so, we first define a dictionary for mapping which (reactant compartment, product compartment) pair is matching which suffix: "io_dict".
	#
	# Some reactions are problematic because they are showing both inflow and outflow simultaneously. To tackle this, we ignore the small molecules listed in "unsignificant_mols".
	#
	# For each reaction, we count the number of "inflowing" and "outflowing" pairs, and the way the reaction happens (forward, backward, reversible or other).

	# "i" for inflowing (None --> e --> c --> d) and "o" for outflowing (d --> c --> e --> None)

	io_dict = {"_i": [(None, "e0"), (None, "c0"), ("e0","c0"), ("c0", "d0"), ("e0", "d0")],
	           "_o": [("c0", None), ("e0", None), ("d0", "e0"), ("d0", "c0"), ("c0", "e0")]}

	unsignificant_mols = ["h_p", "h_c", "pi_c", "pi_p", "adp_c", "h2o_c", "atp_c"]

	# Will print a dictionary counting the reactions in reversible, forward, backward
	reac_id_to_io_count_and_way = screen_out_in(model, io_dict, unsignificant_mols)

	# We duplicate all exchange reactions (excepted sink reactions) and reversible 
	# internal reactions (not unidirectional ones). We get the suffix '_i' for compartment 
	# changing reaction from the exterior to the cytoplasm, and '_o' for the other way. 
	# We also use the suffix "_f" and "_r" for forward and reverse duplicated reactions 
	# that do not change compartment, or show equal compartment exchanges.
	dup_co_model = duplicate_model(model, reac_id_to_io_count_and_way)


	# Correct the medium of the duplicated model, all non-default medium exchange 
	# reactions put at 1e-300, so they are at a value very close to 0 but still appear
	#  in the medium object.
	default_med = model.medium
	new_med = dup_co_model.medium
	correct_med =  correct_duplicated_med(default_med, new_med)
	dup_co_model.medium = correct_med
	print(dup_co_model.medium)

	# ## Medium check-up (default model V.S. duplicated-reaction model)
	# Here we compare the results with randomized medium objects for both models, 
	# reporting the absolute difference between the two.
	for i in range(10):
	    # print('_'*50)
	    s, new_s = change_medium(model, dup_co_model, i*3)
	    if s != None and new_s != None:
	        print(s, new_s, "diff = ", abs(s-new_s))
	    elif s != None:
	        print("infeasible duplicated medium")
	    elif new_s != None:
	        print("infeasible default medium")
	    elif s == None and new_s == None:
	        print("Both medium are impossible")


	# ## Saving the duplicated-reactions model
	dup_co_model.repair() # rebuild indices and pointers in the model if necessary


	## restrict media
	solution = dup_co_model.optimize()
	print(solution)

	for med in remove_med:
	    print(med)
	    dup_co_model.remove_reactions(med)
	    solution = dup_co_model.optimize()
	    
	    print(solution)
	    print("After restricting media, model has ", len(dup_co_model.reactions), " reactions")

	new_name = model_path[:-5] + "_duplicated" + model_path[-5:]
	new_name = new_name.replace('json', 'xml')
	cobra.io.write_sbml_model(dup_co_model, new_name)

	solution = model.optimize()
	## -- Full model
	# cpds = ['cpd03091_c0', 'cpd03091_d0', 'cpd03091_m0', 'cpd02701_m0', 'cpd02701_c0']
	# for cpd_id in cpds:
	# 	if cpd_id in model.metabolites:
	# 	    cpd = model.metabolites.get_by_id(cpd_id)
	# 	    print(cpd.summary())
	# print(solution)

	solution = dup_co_model.optimize()
	print(solution)
	print("Duplicated model's location: " + new_name)
	print("Total number of reactions: ", len(dup_co_model.reactions))

	for reaction in dup_co_model.reactions:
	    if('SK' in reaction.id):
	        print(reaction.id)
	        

	if fluxCoupling: 
		runFluxCoupling(model, dup_co_model, model_path)


