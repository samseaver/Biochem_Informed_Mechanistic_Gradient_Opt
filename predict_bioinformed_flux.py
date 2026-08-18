#!/usr/bin/env python
import warnings
warnings.simplefilter(action='ignore', category=Warning)

import argparse
import shutil
import sys
import os
import time
import random
import numpy as np
import pandas as pd

# ---- CLI ----
# --test       run a short sweep at TEST_EPOCHS (writes under ml/test/svp_X.X/)
# --epochs N   override epoch count explicitly
# --svp X      run only this svp value (otherwise loops over SVP_VALUES,
#              which defaults to the single operating point 2.0)
# --seed N     explicit seed (default: int(time.time()))
_argp = argparse.ArgumentParser(description=__doc__)
_argp.add_argument("--test",   action="store_true",
                   help="Short test sweep (TEST_EPOCHS), output under ml/test/")
_argp.add_argument("--epochs", type=int, default=None,
                   help="Override the epoch count (else: TEST_EPOCHS if --test, else EPOCHS)")
_argp.add_argument("--svp",    type=float, default=None,
                   help="Run only this svp value (else: loop over parameters.SVP_VALUES)")
_argp.add_argument("--seed",   type=int, default=None,
                   help="Explicit random seed (default: int(time.time()))")
_args = _argp.parse_args()

# Generate a random seed
# Using time ensures it changes every second.
# We wrap it in int() to get a clean integer.
#
# PREPRINT PROVENANCE: the arms-260812 sweep behind the manuscript was run
# with the fixed seed 1786429390, passed explicitly as --seed. Reproducing
# those runs requires passing it; the wall-clock default below will not
# reproduce them.
seed_value = _args.seed if _args.seed is not None else int(time.time())

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

	###----- Run ML simulation (pull constants from parameters.py)
	# EPOCHS / TEST_EPOCHS / LEARN_RATE / DECAY_RATE / SVP_VALUES are
	# defined in parameters.py — edit them there to change the sweep.
	if _args.epochs is not None:
		epochs = _args.epochs
	elif _args.test:
		epochs = TEST_EPOCHS
	else:
		epochs = EPOCHS
	learn_rate = LEARN_RATE
	decay_rate = DECAY_RATE
	svp_list   = [_args.svp] if _args.svp is not None else list(SVP_VALUES)

	# Initial flux vector. Set here rather than on the command line: the method
	# uses one initialization and changing it is a deliberate edit, not a runtime
	# option. Full description in get_V0() in Library/Build_Model.py; in brief:
	#   -2  THE PUBLISHED SETTING. Scored reactions start at V_bf; unscored
	#       reactions stay at 0 rather than being imputed at true_vbf_mean/2, so
	#       they are recruited only where mass balance demands it; each media
	#       chain is seeded from integration_results/media_chain_init.tsv
	#       (build it with make_media_chain_init.py).
	#   -1  historical. Scored reactions start at V_bf; unscored ones imputed by
	#       the capacity-aware ~true_vbf_mean/2 rule; no media-chain seeding.
	#    0  no V_bf seeding, no imputation; exchange reactions set to 1000.
	#   >0  flat fill: every non-negative entry set to V0_init.
	# 'bio1' is always forced to 0.0 (hardcoded by id), and the exchange ceilings
	# from fva.tsv are injected as a warm start, whichever value is used.
	# Output files are named for this choice: startVbfandZero / startVbfandMean /
	# startFlat<N>.
	V0_init = -2

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

	# When --test is set, wipe the entire ml/test/ tree first so each test
	# invocation starts from scratch. Production svp dirs (ml/svp_X.X/) are
	# never touched here — those have their own per-step cleanup in
	# Library/Build_Model.py Gradient_Descent.
	if _args.test:
		test_root = os.path.join(ml_parameters.ml_folder, "test")
		if os.path.isdir(test_root):
			print(f"--test: clearing {test_root}/ (stale contents from previous test run)")
			shutil.rmtree(test_root)

	# Per-svp loop. Each iteration writes to its own ml/svp_<value>/ (or
	# ml/test/svp_<value>/ under --test) so successive runs don't clobber.
	# stdout is tee'd to a per-svp run_output.txt for reproducibility.
	class _Tee:
		def __init__(self, *streams): self.streams = streams
		def write(self, s):
			for st in self.streams: st.write(s)
		def flush(self):
			for st in self.streams: st.flush()

	for svp in svp_list:
		svp_dir = (ml_parameters.test_svp_subdir(svp)
		           if _args.test else ml_parameters.svp_subdir(svp))
		os.makedirs(svp_dir, exist_ok=True)
		ml_parameters.output_dir = svp_dir

		log_path = os.path.join(svp_dir, "run_output.txt")
		print(f"\n===== svp = {svp}  (epochs={epochs}, output={svp_dir}) =====")

		with open(log_path, "w") as logf:
			_original_stdout = sys.stdout
			sys.stdout = _Tee(_original_stdout, logf)
			try:
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
			finally:
				sys.stdout = _original_stdout
