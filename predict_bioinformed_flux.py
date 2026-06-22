#!/usr/bin/env python
import warnings
warnings.simplefilter(action='ignore', category=Warning)

import sys
import os
import time
import random
import numpy as np
import pandas as pd

# Generate a random seed
# Using time ensures it changes every second. 
# We wrap it in int() to get a clean integer.
seed_value = int(time.time())

# Print the seed so you can reproduce this run later if needed!
print(f"--- INITIALIZING RANDOM SEED: {seed_value} ---")

# Set the seed for all libraries
random.seed(seed_value)
np.random.seed(seed_value)

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
	# These imports/initializations will now run silently
	from parameters import *
	import cobra
	from Library.Build_Dataset import *
	import Build_Model_Wrapper as bmw

finally:
	# 3. ALWAYS restore the original stdout/stderr
	sys.stdout.close()
	sys.stderr.close()
	sys.stdout = original_stdout
	sys.stderr = original_stderr

if __name__ == '__main__':

	ml_parameters = Parameters_ML()

	dup_model = cobra.io.read_sbml_model(ml_parameters.model_path)

	print(f"Loaded model '{dup_model.id}' with {len(dup_model.reactions)} reactions.")

	# 2. Print Exchange Reactions
	# print("\n--- Exchange Reactions ---")
	# print(f"{'ID':<25} {'Equation'}")
	# print("-" * 60)

	# for rxn in dup_model.exchanges:
	# 	print(f"{rxn.id:<25} {rxn.reaction}")
	
	print(f"--- Scraping treatments from: {ml_parameters.vbf_file} ---")
	try:
        # Read only the header (nrows=0) to be fast/efficient
		vbf_df = pd.read_csv(ml_parameters.vbf_file, sep='\t', nrows=0)
	except FileNotFoundError:
		sys.exit(f"Error: Could not find VBF file at {ml_parameters.vbf_path}")

    # List comprehension to find and clean the names
    # requires Python 3.9+ for .removeprefix()
	treatments = [col.removeprefix('vbf_') for col in vbf_df.columns
	              if col.startswith('vbf_')
	              and (not TREATMENT_FILTERS or any(k in col for k in TREATMENT_FILTERS))]
	if TREATMENT_FILTERS:
		print(f"Applying TREATMENT_FILTERS={TREATMENT_FILTERS!r}")
	print(f"Found {len(treatments)} treatments:",treatments[:5])

	# Build detaset for the ML pipeline: 
	#    -> Reads Vbf to set is as the constraint 
	#    -> Reads the duplicated model to generate the matrix 
	# This requires a specially formatted media file for the boundary reactions in the model
	exchange_list = [r.id for r in dup_model.exchanges if "EX" in r.id]
	with open(ml_parameters.media_file+'.csv','w') as fh:
		print("writing exchanges to training media")
		fh.write(f"name,{','.join(exchange_list)}\n")
		fh.write(f"level{',2'*len(exchange_list)}\n")
		fh.write(f"max_value{',1000'*len(exchange_list)}\n")
		fh.write(f"ratio_drawing{','*len(exchange_list)}\n")

	mediumbound = 'UB' # Exact bound (EB) or upper bound (UB)
	method = 'Vbf' #'FBA' # FBA, pFBA or EXP, Vbf, Vbf_Wt
	cobra_file  = ml_parameters.model_path.replace('.xml', '')
	medium_file = ml_parameters.media_file
	verbose=False

	if not os.path.exists(ml_parameters.training_folder): os.makedirs(ml_parameters.training_folder)

	training_set  = TrainingSet(cobra_name=cobra_file,
                            	medium_name=medium_file,
                            	medium_bound=mediumbound,
                            	treatments=treatments,
                            	verbose=verbose,
                            	method=method,
                            	vbf_name=ml_parameters.vbf_file,
								output_dir=ml_parameters.training_folder,
                            	objective=[],
                            	measure=[],
                            	restrictedFittingList = [])

    # Note: Leaving objective and mesaure as empty lists sets the default
    # objective reaction of the SBML model as the objective reaction
    # and the measure (Y) as this objective reaction.
	training_set.get(sample_size=len(treatments), treatments=treatments, verbose=verbose)

    # Saving file
	training_file  = ml_parameters.training_folder+'training'
	print("Saving training file: ",training_file)
	reduce = False # Set at True if you want to reduce the model
	training_set.save(training_file, reduce=reduce, verbose=verbose)

    # Verifying by re-loading
    # training_set = TrainingSet()
    # print("All Saved .. now loading")
    # training_set.load(training_file)
    # print("printing ... ")
    # training_set.printout()

	###----- Run ML simulation
	epochs=1e6
	learn_rate=5e-2
	decay_rate=.333

	# initial flux for the simulation 
	# -1: set Exchange reactions to 1000, Vbf for all reaction with a value or Vbf_mean 
	# 0 or above: set starting fluxes to will be set to V0_init
	#           : if V0_init is 0, Exchange reactions are set to 1000
	# the biomass reaction is always set to zero but it is hardcoded to find 'bio1'
	V0_init=-1

	# penalty on the steady state constraint
	svp=1.0

	# Which hard constraint to set for modeling 
	# 0: for none
	# 1: for positive flux only 
	# 2: for both positive flux and Vbf
	# 02/24/26: code has been updated to implement different approaches for setting floors and ceilings
	# for fluxes accordingly to Vbf so we not using this parameter
	hardConst = 0

	# ======================================================================
	# --- DYNAMIC FVA WARM START PARSING ---
	# ======================================================================
	fva_file = f"{ml_parameters.integration_folder}/fva.tsv"
	exchange_fluxes = {}
	
	print(f"Parsing FVA file for Warm Start values from: {fva_file}")
	if os.path.exists(fva_file):
		with open(fva_file, 'r') as f:
			for line in f:
				# Split strictly by tabs
				parts = line.strip().split('\t') 
				
				# Check for EX_ prefix and ensure there's a second column
				if len(parts) == 2 and parts[0].startswith('EX_'):
					rxn_id = parts[0]
					flux_val = float(parts[1])  # Directly grab the 2nd column
					exchange_fluxes[rxn_id] = flux_val

		print(f"-> Successfully extracted {len(exchange_fluxes)-1} exchange ceilings.")
	else:
		print(f"Warning: FVA file not found! Defaulting to standard initialization.")
	# ======================================================================

	# Now we pass it down the chain!
	bmw.run_simulation(
	    ml_parameters, 
	    epochs=epochs, 
	    learn_rate=learn_rate, 
	    decay_rate=decay_rate, 
	    V0_init=V0_init, 
	    svp=svp, 
	    hardConst=hardConst,
	    exchanges = exchange_fluxes
	)
