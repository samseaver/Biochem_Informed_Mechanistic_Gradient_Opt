import os
import sys
import numpy as np
from time import time
import datetime
from Library.Build_Model import *


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

def run_simulation(param, epochs=2.5e6, learn_rate=1, decay_rate=.333, V0_init=-1, svp=15, hardConst=1, use_objective=False, biomass_max=100):
    # What you can change
    seed = 10
    np.random.seed(seed=seed)

    ## Model variables: 
    spc = param.spc
    day = param.time_stamp
    tissue = param.other_colm_val
    trainingfile = param.dataset_file
    trainname = trainingfile.split('/')[-1]

    loss_outfile="Result/"+trainname+"_loss"
    targets_outfile= "Result/"+trainname+"_targets"

    # timestep = int(1.0e5) # LP 1.0e4 QP 1.0e5
    timestep = int(epochs) # 3.5e6
    learn_rate = learn_rate # LP 0.3 QP 1.0
    decay_rate = decay_rate # only in QP, UB 0.333 EB 0.9

    # to set a constraint on the biomass reaction
    ##  sets 'biomass_max' as the maximum value for biomass 
    use_objective = False
    objective = [use_objective, 'bio1_biomass'] #[] # ['bio1_biomass'] 

    ## For experiment specific file names
    hardConst_dict = {0: "noRelu", 
                      1: "VposRelu", 
                      2: "VposVbfRelu"}
    V0_initVal = "startVbfandMean" if V0_init < 0 else "start"+str(V0_init)

    id = f'{spc}_{tissue}_{day}_complexFix_plastid_{V0_initVal}_{hardConst_dict[hardConst]}_loopless'

    # End of What you can change


    start = time.time()
    # Create model and run GD for X and Y in trainingfile
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
        Ypred, Stats = MM_QP(model, loss_outfile=loss_outfile, targets_outfile=targets_outfile, V0_init=V0_init, svp=svp, hardConst=hardConst, verbose=True)

    # Printing results
    printout(Ypred, Stats, model, id)

    end = time.time()
    timeD = end - start
    print("Time elapsed: ", str(datetime.timedelta(seconds=timeD)))
