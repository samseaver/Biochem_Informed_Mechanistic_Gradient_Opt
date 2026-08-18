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
    # Use per-run output_dir if set (per-svp loop); fall back to ml_folder.
    out = getattr(param, "output_dir", param.ml_folder)
    temp_df = pandas.DataFrame(data=V, columns=reactions, index=treatments)
    temp_df.to_csv(f"{out}results/{id}_V_headers.tsv", sep='\t')

    np.savetxt(f"{out}results/{id}_X.tsv", Vin, delimiter='\t')
    np.savetxt(f"{out}results/{id}_Pin.tsv", Pin, delimiter='\t')
    np.savetxt(f"{out}results/{id}_Pout.tsv", Pout, delimiter='\t')

def run_simulation(param, epochs=2.5e6, learn_rate=1, decay_rate=.333, V0_init=-2, svp=2.0, hardConst=0, exchanges = None, use_objective=False, biomass_max=100):
    """Run one gradient-descent simulation. Defaults are the published settings.

    V0_init selects how the starting flux vector is built. Full description in
    Library/Build_Model.py::get_V0; in brief:
      -2  DEFAULT, "evidence-only". Scored reactions start at V_bf; UNSCORED
          reactions stay at 0 rather than being imputed at ~true_vbf_mean/2;
          media chains are seeded from integration_results/media_chain_init.tsv.
      -1  historical. Scored reactions start at V_bf; unscored ones are imputed
          by the capacity-aware ~true_vbf_mean/2 rule; no media-chain seeding.
       0  no V_bf seeding, no imputation; exchange reactions set to 1000.
      >0  flat fill at the constant V0_init.

    Output files are named after the initialization that produced them:
    startVbfandZero (-2), startVbfandMean (-1), startFlat<N> (>=0). Before
    2026-08-18 every negative V0_init was labelled "startVbfandMean", so any
    result file predating that carries the -1 label whatever init it used;
    the run log's "[Init] V0_init=-2" line is the authority for those.

    hardConst is retained only for that filename; Build_Model ignores its value.
    """

    # timestep = int(1.0e5) # LP 1.0e4 QP 1.0e5
    timestep = int(epochs) # 3.5e6
    learn_rate = learn_rate # LP 0.3 QP 1.0
    decay_rate = decay_rate # only in QP, UB 0.333 EB 0.9
    
    ## For experiment specific file names
    hardConst_dict = {0: "noRelu", 
                      1: "VposRelu", 
                      2: "VposVbfRelu"}
    # Name the outputs after what the initialization actually did, so a run's
    # files identify themselves. Until 2026-08-18 every negative V0_init was
    # labelled "startVbfandMean", which was correct for -1 and wrong for -2.
    if V0_init == -2:
        V0_initVal = "startVbfandZero"   # scored -> Vbf, unscored -> 0
    elif V0_init < 0:
        V0_initVal = "startVbfandMean"   # scored -> Vbf, unscored -> ~mean/2
    else:
        V0_initVal = "startFlat" + str(V0_init)

    id = f'{V0_initVal}_{hardConst_dict[hardConst]}'

    # End of What you can change

    start = time.time()
    # Per-run output goes to param.output_dir (set per-svp by the driver
    # script). Falls back to ml_folder for backward compatibility.
    out = getattr(param, "output_dir", param.ml_folder)

    # Create model and run GD for X and Y in trainingfile.
    # training_folder stays on `param` directly — training data is
    # invariant to svp and is shared across all per-svp runs.
    model = Neural_Model(trainingfile = param.training_folder+'training',
                         model_type = 'MM_QP',
                         timestep = timestep,
                         learn_rate = learn_rate,
                         decay_rate = decay_rate,
                         exchanges = exchanges,
                         output_dir = out,
                         verbose=True)

    if not os.path.exists(f"{out}initialize"): os.makedirs(f"{out}initialize")
    np.savetxt(f"{out}initialize/Y_{id}.tsv", model.Y, delimiter='\t')

    # Runs the appropriate method
    if not os.path.exists(f"{out}finalize"): os.makedirs(f"{out}finalize")
    loss_outfile=out+"finalize/loss"
    targets_outfile=out+"finalize/targets"

    # Make sure a 'checkpoints' folder exists
    ckpt_dir = os.path.join(out, "checkpoints")
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)

    if model.model_type == 'MM_QP':
        Ypred, Stats = MM_QP(model, loss_outfile=loss_outfile, targets_outfile=targets_outfile, V0_init=V0_init, svp=svp, hardConst=hardConst, verbose=True)

    # Printing results
    if not os.path.exists(f"{out}results"): os.makedirs(f"{out}results")
    printout(Ypred, Stats, model, param, id)

    end = time.time()
    timeD = end - start
    print("Time elapsed: ", str(datetime.timedelta(seconds=timeD)))
