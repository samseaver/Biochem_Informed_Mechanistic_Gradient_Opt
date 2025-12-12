import os
import sys

from Library.Build_Dataset import *

seed = 10
np.random.seed(seed=seed)  # seed for random number generator

def build_dataset(param):
    spc        = param.spc
    time_stamp = param.time_stamp
    other_colm = param.other_colm_value
    treatments = param.treatments
    expFolder  = param.expFolder

    mediumbound = 'UB' # Exact bound (EB) or upper bound (UB)
    method = 'Vbf' #'FBA' # FBA, pFBA or EXP, Vbf, Vbf_Wt
    reduce = False # Set at True if you want to reduce the model

    size = len(treatments)
    measure = []
    rfl = []
    verbose = True
    # End of What you can change

    # Run cobra
    Vbffile    = param.VbfFile
    print(param.VbfFile)
    cobrafile  = param.model_path.replace('.xml', '')
    mediumfile = param.mediaFile
    parameter  = TrainingSet(cobraname=cobrafile,
                            mediumname=mediumfile, mediumbound=mediumbound,
                            method=method,objective=[],
                            measure=measure, Vbfname=Vbffile,
                            restrictedFittingList = rfl, treatments=treatments, verbose=verbose)

    # Note: Leaving objective and mesaure as empty lists sets the default
    # objective reaction of the SBML model as the objective reaction
    # and the measure (Y) as this objective reaction.
    parameter.get(sample_size=size, treatments=treatments, verbose=verbose)

    # Saving file
    trainingfile  = param.dataset_file
    print("Saving training file: ",trainingfile)
    parameter.save(trainingfile, reduce=reduce, verbose=verbose)

    # Verifying
    parameter = TrainingSet()
    print("All Saved .. now loading")
    parameter.load(trainingfile)
    print("printing ... ")
    parameter.printout()

