import warnings
warnings.simplefilter(action='ignore', category=Warning)
import os
import sys
import json

from cobrakbase.core.kbase_object_factory import KBaseObjectFactory
from Library.Duplicate_Model import screen_out_in,duplicate_model
from cobra.flux_analysis import flux_variability_analysis as fva

def stripExchange(model, media):
	
	media_constraints = media.get_media_constraints()
	reactions_to_keep = list()
	for cpd in media_constraints:
		exc = 'EX_'+cpd
		try:
			reaction = model.reactions.get_by_id(exc)
			reactions_to_keep.append(reaction)
		except:
			bounds=media_constraints[cpd]
			if(bounds[0] == 0 and bounds[1] > 0):
				exc+='_o'
				try:
					reaction = model.reactions.get_by_id(exc)
					reactions_to_keep.append(reaction)
				except:
					pass
			elif(bounds[1] == 0 and bounds[0] < 0):
				exc+='_i'
				try:
					reaction = model.reactions.get_by_id(exc)
					reactions_to_keep.append(reaction)
				except:
					pass
			pass
				
	reactions_to_remove = list()
	for exchange in model.exchanges:
		if(exchange not in reactions_to_keep):
			reactions_to_remove.append(exchange)
	
	if(len(reactions_to_remove)>0):
		model.remove_reactions(reactions_to_remove)

def find_remove_inactive_compartments(model,tolerance=1e-6):
		
	fva_result=fva(model,fraction_of_optimum=0.75,processes=1)

	# column 0 is minimum flux
	min_flux_at_max = abs(fva_result.iloc[:,0] + 1000)<tolerance
	min_flux_at_zero = fva_result.iloc[:,0].abs()<tolerance
	min_flux_lt_zero = fva_result.iloc[:,0].abs()>tolerance
	# column 1 is maximum flux
	max_flux_at_max = abs(fva_result.iloc[:,1] - 1000)<tolerance
	max_flux_at_zero = fva_result.iloc[:,1].abs()<tolerance
	max_flux_gt_zero = fva_result.iloc[:,1].abs()>tolerance

	# 1. Reactions with [Min, Max] close to [-1000, 1000] (Full dynamic range)
	rs_rev = min_flux_at_max & max_flux_at_max

	# 2. Reactions with [Min, Max] close to [-1000, 0] (Full negative range)
	rs_irev_bck = min_flux_at_max & max_flux_at_zero
	
	# 3. Reactions with [Min, Max] close to [0, 1000] (Full positive range)
	rs_irev_fwd = max_flux_at_max & min_flux_at_zero

	# 4. Combined selector: Any reaction matching the above three criteria
	pir_selector = rs_rev | rs_irev_bck | rs_irev_fwd
	ar_selector = min_flux_lt_zero.values | max_flux_gt_zero.values
	ir_selector = min_flux_at_zero & max_flux_at_zero

	active_reactions = fva_result.loc[ar_selector,:].index.tolist()
	inactive_reactions = fva_result.loc[ir_selector,:].index.tolist()
	# thermodynamic loops means that reactions that appear to be active are in fact
	# useless, so we look for these too
	possible_inactive_reactions = fva_result.loc[pir_selector,:].index.tolist()

	active_compartment_dict = dict()
	for reaction in model.reactions:
		compartment_list = list(set(metabolite.compartment for metabolite in reaction.metabolites))
		entry = fva_result.loc[reaction.id]
		for cpt in compartment_list:
			if(cpt not in active_compartment_dict):
				active_compartment_dict[cpt]={'active':0,'inactive':0,'ambig':0}
			if(reaction.id in possible_inactive_reactions):
				active_compartment_dict[cpt]['ambig']+=1
			elif(reaction.id in active_reactions):
				active_compartment_dict[cpt]['active']+=1
			elif(reaction.id in inactive_reactions):
				active_compartment_dict[cpt]['inactive']+=1

	# We consider a compartment for removal if it has zero active reactions
	# this excludes reactions that are ambiguous (i.e. part of a loop)
	compartments_to_remove = list()
	for cpt,activity in active_compartment_dict.items():
		if(activity['active']==0):
			compartments_to_remove.append(cpt)

	reactions_to_remove = list()
	for reaction in model.reactions:
		for metabolite in reaction.metabolites:
			if(metabolite.compartment in compartments_to_remove):
				reactions_to_remove.append(reaction.id)

	model.remove_reactions(reactions_to_remove)
	return(model)

def find_remove_blocked_transporters(model):

	reactions_to_remove = list()
	for reaction in model.reactions:
		# skip boundary
		if reaction in model.boundary:
			continue

		# skip non-transport
		if len(reaction.compartments) == 1:
			continue

		for metabolite in reaction.metabolites:
			if len(metabolite.reactions)==1:
				rxn = list(metabolite.reactions)[0]
				reactions_to_remove.append(rxn.id)
				# print(r.id+"\t"+r.reaction+"\t"+",".join([g.id for g in r.genes]))

	model.remove_reactions(reactions_to_remove)
	return(model)

def generateCobraModel(model_path, media_path):
	KBOF = KBaseObjectFactory()
	model = KBOF.build_object_from_file(model_path, "KBaseFBA.FBAModel")
	media = KBOF.build_object_from_file(media_path, "KBaseBiochem.Media")
	model.medium = media
	stripExchange(model, media)

	return model
		
def prepare_model_duplication(model,media_path=None):

	# ## Screen outflowing and inflowing reactions
	# For each reaction that has different compartments in reactants and products 
	# (we call it "transfer reactions"), we annotate the reaction with a suffix "i" for inflowing 
	# (None --> e --> c --> d/m --> y/j) and "o" for outflowing (y/j --> d/m --> c --> e --> None).
	#
	# When the compartment-changing of metabolites is balanced, or not present, 
	# we use different suffix: "for" as in forward, designating the default way of the reaction 
	# (positive flux) and "rev" as in reverse, designating the opposite way (negative flux). 
	# We reverse the products and reactants so that the same reactions happen, ensuring that we 
	# have a positive flux for all reactions.
	#
	# To do so, we first define a dictionary for mapping which (reactant compartment, 
	# product compartment) pair is matching which suffix: "io_dict".
	#
	# Some reactions are problematic because they are showing both inflow and outflow simultaneously.
	# To tackle this, we ignore the small molecules listed in "unsignificant_mols".
	#
	# For each reaction, we count the number of "inflowing" and "outflowing" pairs, and the way the 
	# reaction happens (forward, backward, reversible or other).

	io_dict = {"_i": [(None,"e0"), (None,"c0"), ("e0","c0"), ("c0","d0"), ("e0","d0"), ("c0","m0"), ("d0","y0"), ("m0","j0")],
			   "_o": [("c0",None), ("e0",None), ("y0","d0"), ("j0","m0"), ("m0","c0"), ("d0","e0"), ("d0","c0"), ("c0","e0")]}

	unsignificant_mols = []

	# Will print a dictionary counting the reactions in reversible and backward
	# The dictionary will not count reactions that are determined to be forward
	reac_id_to_io_count_and_way = screen_out_in(model, io_dict, unsignificant_mols)
	
	# We duplicate all exchange reactions (excepted sink reactions) and reversible 
	# internal reactions (not unidirectional ones). We get the suffix '_i' for compartment 
	# changing reaction from the exterior to the cytoplasm, and '_o' for the other way. 
	# We also use the suffix "_f" and "_r" for forward and reverse duplicated reactions 
	# that do not change compartment, or show equal compartment exchanges.
	dup_model = duplicate_model(model, reac_id_to_io_count_and_way)

	if(media_path is not None):
		KBOF = KBaseObjectFactory()
		media = KBOF.build_object_from_file(media_path, "KBaseBiochem.Media")
		print("Stripping media exchanges")
		stripExchange(dup_model,media)
		
	return dup_model
