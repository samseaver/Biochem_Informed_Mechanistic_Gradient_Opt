import numpy as np
import time
import datetime
from Library.Build_Model import *

# We declare this function here and not in the
# function-storing python file to modify it easily
# as it can change the printouts of the methods
def printout(V, Stats, model, param, id='all'):
    # printing Stats
    print("R2 = %.2f (+/- %.2f) Constraint = %.2f (+/- %.2f)" % \
          (Stats.train_objective[0], Stats.train_objective[1],
           Stats.train_loss[0], Stats.train_loss[1]))
    Vout = tf.convert_to_tensor(np.float32(model.Y))
    Loss_norm, dLoss = Loss_Vout_constraint(V, model.Pout, Vout)
    print('Loss Constrained Targets', np.mean(Loss_norm))
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

    processResults(model.reactions, V, Vin, Pin, model.Pout, id, param, model.treatments)

def processResults(reactions, V, Vin, Pin, Pout, id, param, treatments=[]):
    import pandas
    temp_df = pandas.DataFrame(data=V, columns=reactions, index=treatments)
    temp_df.to_csv(f"{param.ml_folder}results/{id}_V_headers.tsv", sep='\t')

    np.savetxt(f"{param.ml_folder}results/{id}_X.tsv", Vin, delimiter='\t')
    np.savetxt(f"{param.ml_folder}results/{id}_Pin.tsv", Pin, delimiter='\t')
    np.savetxt(f"{param.ml_folder}results/{id}_Pout.tsv", Pout, delimiter='\t')

def run_simulation(param, epochs=2.5e6, learn_rate=1, decay_rate=.333, V0_init=-1, svp=15, hardConst=1, exchanges = None, use_objective=False, biomass_max=100):
    
    # timestep = int(1.0e5) # LP 1.0e4 QP 1.0e5
    timestep = int(epochs) # 3.5e6
    learn_rate = learn_rate # LP 0.3 QP 1.0
    decay_rate = decay_rate # only in QP, UB 0.333 EB 0.9
    
    ## For experiment specific file names
    hardConst_dict = {0: "noRelu", 
                      1: "VposRelu", 
                      2: "VposVbfRelu"}
    V0_initVal = "startVbfandMean" if V0_init < 0 else "start"+str(V0_init)

    id = f'{V0_initVal}_{hardConst_dict[hardConst]}'

    # End of What you can change

    start = time.time()
    # Create model and run GD for X and Y in trainingfile
    model = Neural_Model(trainingfile = param.training_folder+'training',
                         model_type = 'MM_QP',
                         timestep = timestep,
                         learn_rate = learn_rate,
                         decay_rate = decay_rate,
                         exchanges = exchanges,
                         output_dir = param.ml_folder,
                         verbose=True)

    if not os.path.exists(f"{param.ml_folder}initialize"): os.makedirs(f"{param.ml_folder}initialize")
    np.savetxt(f"{param.ml_folder}initialize/Y_{id}.tsv", model.Y, delimiter='\t')

    # Runs the appropriate method
    if not os.path.exists(f"{param.ml_folder}finalize"): os.makedirs(f"{param.ml_folder}finalize")
    loss_outfile=param.ml_folder+"finalize/loss"
    targets_outfile=param.ml_folder+"finalize/targets"

    # Make sure a 'checkpoints' folder exists
    ckpt_dir = os.path.join(param.ml_folder, "checkpoints")
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)

    if model.model_type == 'MM_QP':
        Ypred, Stats = MM_QP(model, loss_outfile=loss_outfile, targets_outfile=targets_outfile, V0_init=V0_init, svp=svp, hardConst=hardConst, verbose=True)

    # Printing results
    if not os.path.exists(f"{param.ml_folder}results"): os.makedirs(f"{param.ml_folder}results")
    printout(Ypred, Stats, model, param, id)

    end = time.time()
    timeD = end - start
    print("Time elapsed: ", str(datetime.timedelta(seconds=timeD)))
