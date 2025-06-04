import os
import sys
import numpy as np
from time import time

DIRECTORY = './'
font = 'arial'

# printing the working directory files. One can check you see the same folders and files as in the git webpage.
print(os.listdir(DIRECTORY))

# from pathlib import Path
# project_root = str(Path(__file__).resolve())
# sys.path.append(project_root)
# print(project_root)
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

    # temp_df = pandas.DataFrame(data=Vin, columns=reactions)
    # temp_df.to_csv("Result/tempVin.tsv", sep='\t')
    #
    # temp_df = pandas.DataFrame(data=Pin, columns=reactions)
    # temp_df.to_csv("Result/tempPin.tsv", sep='\t')
    #
    # temp_df = pandas.DataFrame(data=Pout, columns=reactions)
    # temp_df.to_csv("Result/tempPout.tsv", sep='\t')

    np.savetxt(f"Result/{id}_V.tsv", V, delimiter='\t')
    np.savetxt(f"Result/{id}_X.tsv", Vin, delimiter='\t')
    np.savetxt(f"Result/{id}_Pin.tsv", Pin, delimiter='\t')
    np.savetxt(f"Result/{id}_Pout.tsv", Pout, delimiter='\t')


# What you can change
seed = 10

np.random.seed(seed=seed)
spc = 'athaliana'
day = "all"
tissue = "C24"
            # plant_autotrophic_media_restricted_full_UB_12_athaliana_C24_all_complexFix.npz
trainname = f"plant_autotrophic_media_restricted_full_UB_12_{spc}_{tissue}_{day}_complexFix"

loss_outfile="Result/"+trainname+"_loss"
targets_outfile= "Result/"+trainname+"_targets"
size = 13 # number of runs must be lower than the number of element in trainname
timestep = int(1.0e5) # LP 1.0e4 QP 1.0e5
timestep = int(2.5e6) # 3.5e6
learn_rate = 1 # LP 0.3 QP 1.0
decay_rate = .33 # only in QP, UB 0.333 EB 0.9

use_objective = False
objective = [use_objective, 'bio1_biomass'] #[] # ['bio1_biomass'] # ['BIOMASS_Ecoli_core_w_GAM']

biomass_max = 200.0
# End of What you can change


sTime = time.time()
id = f'{spc}_{tissue}_{day}_complexFix_full'
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

# Select a random subset of the training set (of specified size)
if size < model.X.shape[0]:
    ID = np.random.choice(model.X.shape[0], size, replace=False)
    # print(abc)
    # model.X, model.Y= model.X[ID:ID+1,:], model.Y[ID:ID+1,:]
    model.X, model.Y, model.LB= model.X[ID,:], model.Y[ID,:], model.LB[ID,:]
    if len(objective):
        model.objY = model.objY[ID,:]

# print(model.Y)
np.savetxt(f"Y_{id}.csv", model.Y, delimiter=',')
# print(abc)

# Prints a summary of the model before running
model.printout()

# Runs the appropriate method
if model.model_type == 'MM_QP':
    Ypred, Stats = MM_QP(model, loss_outfile=loss_outfile, targets_outfile=targets_outfile, verbose=True)

# Printing results
printout(Ypred, Stats, model, id)
print('Done after ', (time.time()- sTime))


print(abc)
for i in range(5):
    id = i
    sTime = time.time()
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

    # Select a random subset of the training set (of specified size)
    # ID = np.random.choice(model.X.shape[0], size, replace=False)
    ID = [i]
    # print(abc)
    # model.X, model.Y= model.X[ID:ID+1,:], model.Y[ID:ID+1,:]
    model.X, model.Y, model.LB= model.X[ID,:], model.Y[ID,:], model.LB[ID,:]
    if len(objective):
        model.objY = model.objY[ID,:]

    # print(model.Y)
    np.savetxt(f"Y_{id}.csv", model.Y, delimiter=',')
    # print(abc)

    # Prints a summary of the model before running
    model.printout()

    # Runs the appropriate method
    if model.model_type == 'MM_QP':
        Ypred, Stats = MM_QP(model, loss_outfile=loss_outfile, targets_outfile=targets_outfile, verbose=True)

    # Printing results
    printout(Ypred, Stats, model, id)
    print('Done after ', (time.time()- sTime))
    os.replace("Result/sandbox_model_restritected_media_UB_5_loss",\
     f"Result/sandbox_model_restritected_media_UB_5_loss_{id}")
