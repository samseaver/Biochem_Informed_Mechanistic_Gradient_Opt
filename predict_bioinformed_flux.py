import warnings
warnings.simplefilter(action='ignore', category=Warning)

import sys
import os
import copy

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
	import Build_Dataset as bd
	import Build_Model_MM as bmm

finally:
	# 3. ALWAYS restore the original stdout/stderr
	sys.stdout.close()
	sys.stderr.close()
	sys.stdout = original_stdout
	sys.stderr = original_stderr

if __name__ == '__main__':

	model = cobra.io.read_sbml_model('your_model.xml')

	print(f"Loaded model '{model.id}' with {len(model.reactions)} reactions.")

	# 2. Print Exchange Reactions
	print("\n--- Exchange Reactions ---")
	print(f"{'ID':<25} {'Equation'}")
	print("-" * 60)

	for rxn in model.exchanges:
		print(f"{rxn.id:<25} {rxn.reaction}")
	
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
	# bd.build_dataset(ml_param)

	###----- Run ML simulation
	epochs=10 #2.5e6
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
