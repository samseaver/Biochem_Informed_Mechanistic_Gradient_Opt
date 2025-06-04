import os
import sys
import numpy as np
from time import time
import datetime
from Library.Build_Model import *

DIRECTORY = './'
font = 'arial'

# printing the working directory files. One can check you see the same folders and files as in the git webpage.
print(os.listdir(DIRECTORY))

# We declare this function here and not in the
# function-storing python file to modify it easily
# as it can change the printouts of the methods
def printout(V, Stats, model, id='all'):
    # printing Stats
    print("R2 = %.2f (+/- %.2f) Constraint = %.2f (+/- %.2f)" % \
          (Stats.train_objective[0], Stats.train_objective[1],
           Stats.train_loss[0], Stats.train_loss[1]))
    Vout = tf.convert_to_tensor(np.float32(model.Y))
    Loss_norm, dLoss = Loss_Vout(V, model.Pout, Vout)
    print('Loss Targets', np.mean(Loss_norm))
    Loss_norm, dLoss = Loss_SV(V, model.S)
    print('Loss SV', np.mean(Loss_norm))
    Vin = tf.convert_to_tensor(np.float32(model.X))
    Pin = tf.convert_to_tensor(np.float32(model.Pin))
    Vlb = tf.convert_to_tensor(np.float32(model.LB))
    if Vin.shape[1] == model.S.shape[1]: # special case
        Vin  = tf.linalg.matmul(Vin, tf.transpose(Pin), b_is_sparse=True)
    Loss_norm, dLoss = Loss_Vin(V, model.Pin, Vin, model.mediumbound)
    print('Loss Vin bound', np.mean(Loss_norm))
    Loss_norm, dLoss = Loss_Vpos(V, Vlb, model)
    print('Loss V positive', np.mean(Loss_norm))
    # print(V)

    processResults(model.reactions, V, Vin, Pin, model.Pout, id, model.treatments)

def processResults(reactions, V, Vin, Pin, Pout, id, treatments=[]):
    import pandas
    temp_df = pandas.DataFrame(data=V, columns=reactions)
    temp_df.to_csv(f"Result/{id}_V_rxn.tsv", sep='\t')
    temp_df = pandas.DataFrame(data=V, columns=reactions, index=treatments)
    temp_df.to_csv(f"Result/{id}_V_rxn_trmt.tsv", sep='\t')

    np.savetxt(f"Result/{id}_V.tsv", V, delimiter='\t')
    np.savetxt(f"Result/{id}_X.tsv", Vin, delimiter='\t')
    np.savetxt(f"Result/{id}_Pin.tsv", Pin, delimiter='\t')
    np.savetxt(f"Result/{id}_Pout.tsv", Pout, delimiter='\t')


# What you can change
seed = 10
np.random.seed(seed=seed)


spc = 'Sorghum'
day = "all"
tissue = "Leaf"
trainname = f"plastidial_model_duplicated_restricted_media_noATP_noADP_noP_UB_25_{spc}_{tissue}_{day}_complexFix"

loss_outfile="Result/"+trainname+"_loss"
targets_outfile= "Result/"+trainname+"_targets"
size = 26 # number of runs must be lower than the number of element in trainname
# timestep = int(1.0e5) # LP 1.0e4 QP 1.0e5
timestep = int(2.5e6) # 3.5e6
learn_rate = 1 # LP 0.3 QP 1.0
decay_rate = .333 # only in QP, UB 0.333 EB 0.9

use_objective = False
objective = [use_objective, 'bio1_biomass'] #[] # ['bio1_biomass'] 

biomass_max = 200.0
# End of What you can change


start = time.time()

id = f'{spc}_{tissue}_{day}_complexFix_plastid_startVbf_custRelu'

# Create model and run GD for X and Y randomly drawn from trainingfile
trainingfile = DIRECTORY+'Dataset_model/'+trainname
model = Neural_Model(trainingfile = trainingfile,
              objective=objective,
              model_type = 'MM_QP',
              timestep = timestep,
              learn_rate = learn_rate,
              decay_rate = decay_rate,
              biomass_max = biomass_max,
              verbose=True)

model.printout()
np.savetxt(f"Y_{id}.csv", model.Y, delimiter=',')

# Prints a summary of the model before running
model.printout()

# Runs the appropriate method
if model.model_type == 'MM_QP':
    Ypred, Stats = MM_QP(model, loss_outfile=loss_outfile, targets_outfile=targets_outfile, verbose=True)

# Printing results
printout(Ypred, Stats, model, id)

end = time.time()
timeD = end - start
print("Time elapsed: ", str(datetime.timedelta(seconds=timeD)))
