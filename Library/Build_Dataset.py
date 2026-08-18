###############################################################################
# This library create training sets for AMN
# Trainig sets are either based on experimental datasets
# or FBA (Cobrapy) simulations
# Authors: Jean-loup Faulon jfaulon@gmail.com and Bastien Mollet
###############################################################################

from __future__ import print_function
import os
import sys
import math
import numpy as np
import pandas
import cobra
from sklearn.utils import shuffle
sys.setrecursionlimit(10000) # for row_echelon function

###############################################################################
# New methods for Vbf
###############################################################################


def _s_matrix_path():
    """Destination for the S-matrix export, or None if there is nowhere to put it.

    This used to be written as a bare 's_matrix.csv' relative to the working
    directory, so the eight concurrent runs of a sweep all wrote the same file
    and the survivor was whichever finished last. Follow BF_PROJECT -- the
    per-run override the sweep already sets -- so each run writes into its own
    integration_results/ instead.
    """
    proj = os.environ.get("BF_PROJECT", "").rstrip("/")
    d = os.path.join(proj, "integration_results") if proj else "integration_results"
    return os.path.join(d, "s_matrix.csv") if os.path.isdir(d) else None


def _write_s_matrix(model):
    path = _s_matrix_path()
    if path is None:
        return
    cobra.util.array.create_stoichiometric_matrix(
        model, array_type="DataFrame").to_csv(path)

def get_matrices_vbf(model, medium, vbf, measure, reactions):
    # Build the matrices consumed by the QP gradient-descent solver.
    # vbf is the list of
    # Return
    # - S [mxn]: stochiometric matrix
    # - V2M [mxn]: to compute metabolite
    #        production fluxes from reaction fluxes
    # - M2V [mxn]: to compute reaction fluxes
    #        from substrate production fluxes
    # - Pin [n_in x n]: to go from reactions to medium fluxes
    # - Pout [n_out x n]: to go from reactions to measured fluxes

    # m = metabolite, n = reaction/v/flux, p = medium
    S = np.asarray(cobra.util.array.create_stoichiometric_matrix(model))
    _write_s_matrix(model)
    n, m, n_in, n_out = S.shape[1], S.shape[0], len(medium), len(measure)

    # Get V2M and M2V from S
    V2M, M2V = S.copy(), S.copy()
    for i in range(m):
        for j in range(n):
            if S[i][j] < 0:
                V2M[i][j] = 0
                M2V[i][j] = -1/S[i][j]
            else:
                V2M[i][j] = S[i][j]
                M2V[i][j] = 0
    M2V = np.transpose(M2V)

    # Boundary matrices from reaction to medium fluxes
    Pin, i = np.zeros((n_in,n)), 0
    for rid in medium:
        j = get_index_from_id(rid,reactions)
        Pin[i][j] = 1
        i = i+1

    # Experimental measurements matrix from reaction to measured fluxes
    Pout, i = np.zeros((n_out,n)), 0

    for rid in measure:
        j = get_index_from_id(rid,reactions)
        Pout[i][j] = 1
        i = i+1

    return S, Pin, Pout, V2M, M2V

###############################################################################
# IOs with pandas
###############################################################################

def read_csv(filename):
    # Reading datafile with pandas
    # Return HEADER and DATA
    filename += '.csv'
    dataframe = pandas.read_csv(filename, header=0)
    HEADER = dataframe.columns.tolist()
    dataset = dataframe.values
    DATA = np.asarray(dataset[:,:])
    return HEADER, DATA


###############################################################################
# Cobra's model utilities and matrices (written by Bastien Mollet)
###############################################################################

# Cobra utilities and stoichiometric derived matrices
def get_index_from_id(name,L):
    # Return index in L of id name
    for i in range(len(L)):
        if L[i].id == name:
            return i
    return -1

def get_objective(model):
    # Get the reaction carring the objective
    # Someone please tell me if there is
    # a clearner way in Cobra to get
    # the objective reaction

    r = str(model.objective.expression)
    r = r.split()
    r = r[0].split('*')
    obj_id = r[1]

    # line below crash if does not exist
    r = model.reactions.get_by_id(obj_id)

    return obj_id

def get_matrices(model, medium, measure, reactions):
    # Build the matrices consumed by the QP gradient-descent solver.
    # Return
    # - S [mxn]: stochiometric matrix
    # - V2M [mxn]: to compute metabolite
    #        production fluxes from reaction fluxes
    # - M2V [mxn]: to compute reaction fluxes
    #        from substrate production fluxes
    # - Pin [n_in x n]: to go from reactions to medium fluxes
    # - Pout [n_out x n]: to go from reactions to measured fluxes

    # m = metabolite, n = reaction/v/flux, p = medium
    S = np.asarray(cobra.util.array.create_stoichiometric_matrix(model))
    _write_s_matrix(model)

    n, m, n_in, n_out = S.shape[1], S.shape[0], len(medium), len(measure)

    # Get V2M and M2V from S
    V2M, M2V = S.copy(), S.copy()
    for i in range(m):
        for j in range(n):
            if S[i][j] < 0:
                V2M[i][j] = 0
                M2V[i][j] = -1/S[i][j]
            else:
                V2M[i][j] = S[i][j]
                M2V[i][j] = 0
    M2V = np.transpose(M2V)

    # Boundary matrices from reaction to medium fluxes
    Pin, i = np.zeros((n_in,n)), 0
    for rid in medium:
        j = get_index_from_id(rid,reactions)
        Pin[i][j] = 1
        i = i+1

    # # Experimental measurements matrix from reaction to measured fluxes
    # Pout, i = np.zeros((n_out,n)), 0
    # for rid in measure:
    #     j = get_index_from_id(rid,reactions)
    #     Pout[i][j] = 1
    #     i = i+1
    # Experimental measurements matrix from reaction to measured fluxes
    Pout, i = np.zeros((n,n)), 0
    for rid in measure:
        j = get_index_from_id(rid,reactions)
        Pout[j][j] = 1

    # print(Pout)
    # print(abc)
    return S, Pin, Pout, V2M, M2V

def get_matrices_original(model, medium, measure, reactions):
    # Build the matrices consumed by the QP gradient-descent solver.
    # Return
    # - S [mxn]: stochiometric matrix
    # - V2M [mxn]: to compute metabolite
    #        production fluxes from reaction fluxes
    # - M2V [mxn]: to compute reaction fluxes
    #        from substrate production fluxes
    # - Pin [n_in x n]: to go from reactions to medium fluxes
    # - Pout [n_out x n]: to go from reactions to measured fluxes

    # m = metabolite, n = reaction/v/flux, p = medium
    S = np.asarray(cobra.util.array.create_stoichiometric_matrix(model))
    _write_s_matrix(model)
    n, m, n_in, n_out = S.shape[1], S.shape[0], len(medium), len(measure)

    # Get V2M and M2V from S
    V2M, M2V = S.copy(), S.copy()
    for i in range(m):
        for j in range(n):
            if S[i][j] < 0:
                V2M[i][j] = 0
                M2V[i][j] = -1/S[i][j]
            else:
                V2M[i][j] = S[i][j]
                M2V[i][j] = 0
    M2V = np.transpose(M2V)

    # Boundary matrices from reaction to medium fluxes
    Pin, i = np.zeros((n_in,n)), 0
    for rid in medium:
        j = get_index_from_id(rid,reactions)
        Pin[i][j] = 1
        i = i+1

    # Experimental measurements matrix from reaction to measured fluxes
    Pout, i = np.zeros((n_out,n)), 0
    for rid in measure:
        j = get_index_from_id(rid,reactions)
        Pout[i][j] = 1
        i = i+1

    return S, Pin, Pout, V2M, M2V


###############################################################################
# Running Cobra
###############################################################################

def run_cobra(model, objective, IN, method='FBA', verbose=False, objective_fraction=0.75, cobra_min_flux=1.0e-8):
    # Inputs:
    # - model
    # - objective: a list of reactions (first two only are considered)
    # - IN: Initial values for all reaction fluxes
    # - method: FBA or pFBA
    # run FBA optimization to compute recation fluxes on the provided model
    # set the medium using values in dictionary IN.
    # When 2 objectives are given one first maximize the first objective (obj1).
    # then one set the upper and lower bounds for that objective to
    # objective_fraction * obj1 (e.g. objective_fraction = 0.75) and maximize
    # for the second objective
    # Outputs:
    # - FLUX, the reaction fluxes compyted by FBA for all reactions
    # - The value for the objective

    # set the medium and objective

    medium = model.medium # This is the model medium
    medini = medium.copy()
    for k in medium.keys(): # Reset the medium
        medium[k] = 0
    for k in IN.keys(): # Additional cmpds added to medium
        if k in medium.keys():
            medium[k] = float(IN[k])
    model.medium = medium

    # run FBA for primal objective
    model.objective = objective[0]
    solution = cobra.flux_analysis.pfba(model) \
    if method == 'pFBA' else model.optimize()


    solution_val = solution.fluxes[objective[0]]
    if solution_val <= 0:
        print(objective)
        print(solution)
        print(model.medium)

    if verbose:
        print('primal objectif =', objective, method, solution_val)

    # run FBA for second objective
    # primal objectif is set to a fraction of its value
    if len(objective) > 1:
        obj = model.reactions.get_by_id(objective[0])
        obj_lb, obj_ub = obj.lower_bound, obj.upper_bound
        obj.lower_bound = objective_fraction * solution_val
        obj.upper_bound = objective_fraction * solution_val
        model.objective = objective[1]
        solution = cobra.flux_analysis.pfba(model) \
        if method == 'pFBA' else model.optimize()
        solution_val = solution.fluxes[objective[1]]
        if verbose:
            print('second objectif =', objective, method, solution_val)

        # reset bounds and objective to intial values
        obj.lower_bound, obj.upper_bound = obj_lb, obj_ub
        model.objective = objective[0]

    # get the fluxes for all model reactions
    FLUX = IN.copy()
    for x in model.reactions:
        if x.id in FLUX.keys():
            FLUX[x.id] = solution.fluxes[x.id]
            if math.fabs(float(FLUX[x.id])) < cobra_min_flux: # !!!
                FLUX[x.id] = 0

    # Reset medium
    model.medium = medini

    return FLUX, solution_val

###############################################################################
# Generating random medium runing Cobra
###############################################################################
def create_fiexd_medium_vbf(model, medium, valmed, verbose=False):
    # Generate a fixed flux matrix using model media
    # Input:
    # - model
    # - medium: list of reaction fluxes in medium
    # Ouput:
    # - Intial reaction fluxes set to medium values

    INFLUX = dict()
    
    for r in model.reactions:
        INFLUX[r.id] = 0

    for ind in range(len(medium)):
        INFLUX[medium[ind]] = valmed[ind]
    
    return INFLUX

def getBioFluxes(model, Pout, medium, valmed, bioFluxes, treatment, augmntXY=False, inf={}):
    """
    Extracts biological fluxes from the VBF dataframe for a specific treatment.
    Expects columns in the format: 'vbf_{treatment}'
    """
    # Initialize default medium/fluxes if not provided
    if not inf:
        inf = create_fiexd_medium_vbf(model, medium, valmed)

    FLUX = inf.copy()
    LB_vals = dict()
    
    # Define the specific column name based on the new header format
    col_name = f"vbf_{treatment}"
    
    # Check if the treatment exists in the dataframe
    if col_name not in bioFluxes.columns:
        print(f"Warning: Column '{col_name}' not found in VBF data. Defaulting to 0.")

    found = 0
    
    for x in model.reactions:
        # 1. Determine the flux value
        flux = FLUX.get(x.id, 0.0) 
        
        # We add a boolean flag to track if the reaction actually exists in the VBF dataset
        is_measured = False 
        
        if (x.id in bioFluxes.index) and (col_name in bioFluxes.columns):
            try:
                val = bioFluxes.loc[x.id, col_name]
                if not np.isnan(val):
                    flux = val
                    is_measured = True # The data exists, even if it is 0.0!
                    found += 1
            except Exception:
                pass 
        
        # 2. Update the FLUX dictionary
        FLUX[x.id] = flux

        # 3. Update Pout (Measurement Matrix)
        # THE FIX: If it is in the dataset, enforce it (Pout = 1). 
        # If it is missing from the dataset, leave it free (Pout = 0).
        j = get_index_from_id(x.id, model.reactions)
        if is_measured:
            Pout[j][j] = 1
        else:
            Pout[j][j] = 0

        # 4. Store Lower Bounds
        LB_vals[x.id] = x.lower_bound

    # if verbose:
    #     print(f"Assigned VBF values to {found}/{len(model.reactions)} reactions from '{col_name}'")

    # Format outputs
    Y = np.asarray(list(FLUX.values()))
    LB = np.asarray(list(LB_vals.values()))
    X = np.asarray([ inf[medium[i]] for i in range(len(medium)) ])
    
    if augmntXY:
        X = np.concatenate((X, Y), axis=0)

    return X, Y, LB, Pout

def create_random_medium_cobra(model, objective, medium, mediumbound, in_varmed, levmed, valmed, ratmed, method='FBA', verbose=False, cobra_min_objective=1.0e-3):
    # Generate a random input and get Cobra's output
    # Input:
    # - model
    # - objective: the reaction fluxes to optimize
    # - medium: list of reaction fluxes in medium
    # - in_varmed: the medium reaction fluxes allowed to change
    #              (can be empty then varmed are drawn at random)
    # - levmed: teh number of level a flux can take
    # - valmed: the maximum value the flux can take
    # - ratmed: the ration of fluxes turned on
    # - method: the method used by Cobra
    # Make sure the medium does not kill the objective
    # i.e. objective > cobra_min_objective
    # Ouput:
    # - Intial reaction fluxes set to medium values

    MAX_iteration = 5 # max numbrer of Cobra's failaure allowed

    medini = model.medium.copy()
    INFLUX = {}
    for r in model.reactions:
        INFLUX[r.id] = 0

    # X = actual number of variable medium turned ON
    L_in_varmed = len(in_varmed)
    if L_in_varmed > 0:
        X = len(in_varmed)
    else:
        X = sum(map(lambda x : x>1, levmed)) # total number of variable medium
        X = np.random.binomial(X, ratmed, 1)[0] if ratmed < 1 else int(ratmed)
        X = 1 if X == 0 else X

    X = len(medini)
    print(f"can change {X} media")

    # Indices for minmed varmed
    minmed, varmed = [], []
    for i in range(len(medium)):
        if levmed[i] <= 1: # mimimum medium indices
            minmed.append(i)
        else:
            if len(in_varmed) > 0:
                if medium[i] not in in_varmed:
                    continue
            varmed.append(i) # variable medium indices

    modmed = minmed + varmed  if mediumbound == 'EB' else varmed

    for iteration in range(MAX_iteration):
        # create random medium choosing X fluxes in varmed at random
        INFLUX = {k: 0 for k in INFLUX.keys()} # reset
        model.medium = medini # reset
        varmed = shuffle(varmed) # that's where random choice occur
        for i in range(len(minmed)):
            j = minmed[i]
            k = medium[j]
            INFLUX[k], model.medium[k] = valmed[j], valmed[j]
        for i in range(X):
            j = varmed[i]
            k = medium[j]
            v = (L_in_varmed+1) * np.random.randint(1,high=levmed[j]) * valmed[j]/(levmed[j]-1)
            INFLUX[k], model.medium[k] = v, v

        # check with cobra
        try:
            _, obj = run_cobra(model, objective, INFLUX,
                               method=method, verbose=False)
        except:
            print('Cobra cannot be run start again')
            # treshold, iteration, up, valmed = \
            # init_constrained_objective(objective_value, in_treshold,
            #                 modmed, valmed, verbose=verbose)
            continue

        if obj < cobra_min_objective:
            continue # must have some objective

        # We have a solution
        if verbose:
            p = [ medium[varmed[i]] for i in range(X)]
            print('pass (varmed, obj):', p, obj)
        break

    model.medium = medini # reset medium

    return INFLUX

def get_io_cobra(model, objective, medium, mediumbound, varmed, levmed, valmed, ratmed, E, method='FBA', inf={}, verbose=False):
    # Generate a random input and get Cobra's output
    # Input:
    # - model: the cobra model
    # - objective: the list of objectiev fluxes to maximize
    # - medium: list of reaction fluxes in medium
    # - varmed: the medium reaction fluxes allowed to change
    #            (can be empty then varmed are drawn at random)
    # - levmed: the number of level an uptake flux can take
    # - valmed: the maximum value the flux can take
    # - ratmed: the ration of fluxes turned on
    # - method: the method used by Cobra
    # Output:
    # - X=medium , Y=fluxes for reactions in E
    
    if inf == {}:
        inf = create_random_medium_cobra(model, objective,
                                         medium, mediumbound,
                                         varmed, levmed, valmed.copy(), ratmed,
                                         method=method,verbose=verbose)
    out,obj = run_cobra(model,objective,inf,method=method,verbose=verbose)

    Y = np.asarray(list(out.values()))
    X = np.asarray([ inf[medium[i]] for i in range(len(medium)) ])

    return X, Y

###############################################################################
# Creating, saving and loading training set object
# Training set object used in all modules
###############################################################################

class TrainingSet:
    # All element necessary to run AMN
    # cf. save for definition of parameters
    def __init__(self,  cobra_name='', medium_name='', medium_bound='EB', medium_size=-1, objective=[], method='FBA', measure=[], vbf_name='', restrictedFittingList=[], treatments=[], output_dir = "./", verbose=False):

        # Store output directory
        self.output_dir = output_dir
        
        # Create directory if it doesn't exist
        if self.output_dir and not os.path.exists(self.output_dir):
            try:
                os.makedirs(self.output_dir)
                if verbose: print(f"Created output directory: {self.output_dir}")
            except OSError as e:
                sys.exit(f"Error creating output directory {self.output_dir}: {e}")

        if cobra_name == '':
            return # create an empty object
        if not os.path.isfile(cobra_name+'.xml'):
            print(cobra_name)
            sys.exit('xml cobra file not found')
        if not os.path.isfile(medium_name+'.csv'):
            print(medium_name)
            sys.exit('medium or experimental file not found')
        if method.lower() == 'vbf':
            if not os.path.isfile(vbf_name):
                sys.exit(f"Vbf file {vbf_name} not found")

        self.cobraname = cobra_name # model cobra file
        self.mediumname = medium_name # medium file
        self.mediumbound = medium_bound # EB or UB
        self.method = method
        self.Vbfname = vbf_name
        self.model = cobra.io.read_sbml_model(cobra_name+'.xml')
        ## remove --------------------
        # self.model.reactions.get_by_id("bio1_biomass").lower_bound = 1
        #### -------------------------

        self.reduce = False
        self.allmatrices = True
        self.treatments = treatments

        # Read V_bf dataframe
        self.Vbf_df = pandas.read_csv(vbf_name, sep='\t')
        if restrictedFittingList:
            self.Vbf_df = self.Vbf_df[self.Vbf_df['reaction_id'].isin(restrictedFittingList)]

        self.Vbf_df = self.Vbf_df.set_index('reaction_id')

        if self.method.lower() == 'vbf':
            print("Vbf method selected")
            vbf_rxns = list(self.Vbf_df.index)
            # to keep the sorting correct
            # self.measure = [r.id for r in self.model.reactions if r.id.rsplit("_", 1)[0] in vbf_rxns]
            self.measure = [r.id for r in self.model.reactions if r.id in vbf_rxns]
            # print("Measure: ", len(self.measure), len(vbf_rxns))

        else:
            self.measure = [r.id for r in self.model.reactions] \
            if measure == [] else measure

        # set medium
        H, M = read_csv(medium_name)
        if 'EXP' in self.method : # Reading X, Y
            if medium_size < 1:
                sys.exit('must indicate medium size with experimental dataset')
            medium = []
            for i in range(medium_size):
                medium.append(H[i])
            self.medium = medium
            self.levmed, self.valmed, self.ratmed = [], [], 0
            self.X = M[:,0:len(medium)]
            self.Y = M[:,len(medium):]
            self.size = self.Y.shape[0]
            self.LB = None
        else:
            self.medium = H[1:]
            self.levmed = [float(i) for i in M[0,1:]]
            self.valmed = [float(i) for i in M[1,1:]]
            self.ratmed = float(M[2,1])
            self.X, self.Y, self.LB = np.asarray([]).reshape(0,0), \
            np.asarray([]).reshape(0,0), np.asarray([]).reshape(0,0)

        if verbose:
            print(self.Vbf_df.head())
            print(self.measure[:3])
            print('medium:',self.medium)
            print('levmed:',self.levmed)
            print('valmed:',self.valmed)
            print('ratmed:',self.ratmed)

        # set objectve and measured reactions lists
        self.objective = [get_objective(self.model)] \
        if objective == [] else objective
        if verbose:
            print('objective: ',self.objective)
            print('measurements size: ',len(self.measure))

        # compute matrices and objective vector for AMN
        self.S, self.Pin, self.Pout, self.V2M, self.M2V = \
        get_matrices(self.model, self.medium, self.measure,
                     self.model.reactions)

    def save(self, filename, reduce=False, verbose=False):
        # save cobra model in xml and parameter in npz (compressed npy)
        self.reduce = reduce
        # remove
        # if self.reduce:
        #     self.reduce_and_run(verbose=verbose)

        # Recompute matrices
        self.S, self.Pin, _, self.V2M, self.M2V = \
        get_matrices(self.model, self.medium, self.measure,
                         self.model.reactions)
        self.S_int, self.S_ext, self.Q, self.P, \
        self.b_int, self.b_ext, self.Sb, self.c = \
        [], [], [], [], [], [], [], []
        # get_matrices_LP(self.model, self.mediumbound, self.X, self.S,
                             # self.Pin, self.medium, self.objective)
        # save cobra file
        cobra.io.write_sbml_model(self.model, filename+'.xml')
        # save parameters
        np.savez_compressed(filename,
                            cobraname = filename,
                            reduce = self.reduce,
                            mediumname = self.mediumname,
                            mediumbound = self.mediumbound,
                            objective =self.objective,
                            method = self.method,
                            size = self.size,
                            medium = self.medium,
                            levmed = self.levmed,
                            valmed = self.valmed,
                            ratmed = self.ratmed,
                            measure = self.measure,
                            S = self.S,
                            Pin = self.Pin,
                            Pout = self.Pout,
                            V2M = self.V2M,
                            M2V = self.M2V,
                            X = self.X,
                            Y = self.Y,
                            S_int = self.S_int,
                            S_ext = self.S_ext,
                            Q = self.Q,
                            P = self.P,
                            b_int = self.b_int,
                            b_ext = self.b_ext,
                            Sb = self.Sb,
                            c = self.c,
                            reactions = [r.id for r in self.model.reactions],
                            LB = self.LB,
                            treatments = self.treatments)

    def load(self, filename):
        # load parameters (npz format)
        if not os.path.isfile(filename+'.npz'):
            print(filename+'.npz')
            sys.exit('file not found')

        loaded = np.load(filename+'.npz')
        self.cobraname = str(loaded['cobraname'])
        self.reduce = str(loaded['reduce'])
        self.reduce = True if self.reduce == 'True' else False
        self.mediumname = str(loaded['mediumname'])
        self.mediumbound = str(loaded['mediumbound'])
        self.objective = loaded['objective']
        self.method = str(loaded['method'])
        self.size = loaded['size']
        self.medium = loaded['medium']
        self.levmed = loaded['levmed']
        self.valmed = loaded['valmed']
        self.ratmed = loaded['ratmed']
        self.measure = loaded['measure']
        self.S = loaded['S']
        self.Pin = loaded['Pin']
        self.Pout = loaded['Pout']
        self.V2M = loaded['V2M']
        self.M2V = loaded['M2V']
        self.X = loaded['X']
        self.Y = loaded['Y']
        self.S_int = loaded['S_int']
        self.S_ext = loaded['S_ext']
        self.Q = loaded['Q']
        self.P = loaded['P']
        self.b_int = loaded['b_int']
        self.b_ext = loaded['b_ext']
        self.Sb = loaded['Sb']
        self.c = loaded['c']
        self.allmatrices = True
        self.model = cobra.io.read_sbml_model(self.cobraname+'.xml')

        if 'LB' in loaded:
            self.LB = loaded['LB']
        else:
            self.LB = {}
            print(self.size)
            for i in range(self.size):
                LB_vals = dict()
                for x in self.model.reactions:
                    LB_vals[x.id] = x.lower_bound
                self.LB[i] = np.asarray(list(LB_vals.values()))
            self.LB = np.asarray(list(self.LB.values()))

        if 'reactions' in loaded:
            self.reactions = loaded['reactions']
        else:
            self.reactions = [r.id for r in self.model.reactions]

        self.treatments = loaded['treatments'] if 'treatments' in loaded \
                                                else []


    def printout(self,filename=''):
        self.additionalPrinting()
        print(f"model reactions: {len(self.reactions)} ")
        if filename != '':
            sys.stdout = open(filename, 'wb')
        print('model file name:',self.cobraname)
        print('reduced model:',self.reduce)
        print('medium file name:',self.mediumname)
        print('medium bound:',self.mediumbound)
        print('list of reactions in objective:',self.objective)
        print('method:',self.method)
        print('trainingsize:',self.size)
        print('list of medium reactions:',len(self.medium))
        print('list of medium levels:',len(self.levmed))
        print('list of medium values:', self.valmed)
        print('ratio of variable medium turned on:',self.ratmed)
        print('list of measured reactions:',len(self.measure))
        print('Stoichiometric matrix',self.S.shape)
        print('Boundary matrix from reactions to medium:',self.Pin.shape)
        print('Measurement matrix from reaction to measures:',self.Pout.shape)
        print('Reaction to metabolite matrix:',self.V2M.shape)
        print('Metabolite to reaction matrix:',self.M2V.shape)
        print('Training set X:',self.X.shape)
        print('Training set Y:',self.Y.shape)
        print('Training set LB:',self.LB.shape)
        print('Training set treatments: ', self.treatments)
        if self.allmatrices:
            print('S_int matrix', self.S_int.shape)
            print('S_ext matrix', self.S_ext.shape)
            print('Q matrix', self.Q.shape)
            print('P matrix', self.P.shape)
            print('b_int vector', self.b_int.shape)
            print('b_ext vector', self.b_ext.shape)
            print('Sb matrix', self.Sb.shape)
            print('c vector', self.c.shape)
        if filename != '':
            sys.stdout.close()


    def additionalPrinting(self):
        import pandas
        temp_df = pandas.DataFrame(self.Pout, columns=self.reactions)
        temp_df.index = self.reactions
        temp_df.to_csv(os.path.join(self.output_dir,"dataSetPout.tsv"), sep='\t')

        temp_df = pandas.DataFrame(self.Y, columns=self.reactions)
        temp_df.index = self.treatments
        temp_df.to_csv(os.path.join(self.output_dir,"dataSetY.tsv"), sep='\t')

        temp_df = pandas.DataFrame(self.X, columns=self.medium)
        temp_df.index = self.treatments
        temp_df.to_csv(os.path.join(self.output_dir,"dataSetX.tsv"), sep='\t')
        # temp_df = pandas.DataFrame(data=Vin, columns=reactions)
        # temp_df.to_csv(os.path.join(self.output_dir,"tempVin.tsv"), sep='\t')
        #
        # temp_df = pandas.DataFrame(data=Pin, columns=reactions)
        # temp_df.to_csv(os.path.join(self.output_dir,"tempPin.tsv"), sep='\t')
        #
        # temp_df = pandas.DataFrame(data=Pout, columns=reactions)
        # temp_df.to_csv(os.path.join(self.output_dir,"tempPout.tsv"), sep='\t')

        # np.savetxt(os.path.join(self.output_dir,"dataSetPout.tsv"), self.Pout, delimiter='\t')


    def get(self, sample_size=100, varmed=[], reduce=False, treatments=[], verbose=False):
        # Generate a training set for AMN
        # Input: sample size
        # objective_value and variable medium
        # (optional when experimental datafile)
        # Output: X,Y (medium and reaction flux values)

        X, Y, LB, inf = {}, {}, {}, {}
        if 'vbf' not in self.method.lower():
            for i in range(sample_size):
                if verbose: print('sample:',i)

                # Cobra is run on reduce model where X is already know
                if reduce:
                    inf = {r.id: 0 for r in self.model.reactions}
                    for j in range(len(self.medium)):
                        inf[self.medium[j]] = self.X[i,j]
                LB_vals = dict()
                for x in model.reactions:
                    LB_vals[x.id] = x.lower_bound
                LB[i] = np.asarray(list(LB_vals.values()))

                X[i], Y[i] = \
                get_io_cobra(self.model, self.objective,
                             self.medium, self.mediumbound, varmed,
                             self.levmed, self.valmed, self.ratmed,
                             self.Pout, inf=inf, method=self.method,
                             verbose=verbose)


        else:
            # append Vbf values to X if method is 'vbf_wt'
            augmntXY = True if self.method.lower() == 'vbf_wt' else False
            for i in range(len(treatments)):
                trmt = treatments[i]

                X[i], Y[i], LB[i], self.Pout = \
                getBioFluxes(self.model, self.Pout, self.medium, self.valmed,
                                    self.Vbf_df, trmt, augmntXY=augmntXY)

        # Y[len(treatments)]=rxns
        X = np.asarray(list(X.values()))
        Y = np.asarray(list(Y.values()))
        LB = np.asarray(list(LB.values()))
        rxns = [r.id for r in self.model.reactions]
        temp_df = pandas.DataFrame(data=Y, columns=rxns)
        temp_df.index = self.treatments
        temp_df.to_csv(os.path.join(self.output_dir,"temp_Y.tsv"),sep='\t')

        # rxns = [r.id for r in self.model.reactions]
        # temp_df = pandas.DataFrame(data=LB, columns=rxns)
        # temp_df.index = self.treatments
        # temp_df.to_csv(os.path.join(self.output_dir,"temp_LB.csv"))
 
        # In case mediumbound is 'EB' replace X[i] by Y[i] for i in medium
        if self.mediumbound == 'EB':
            i = 0
            for rid in self.medium:
                j = get_index_from_id(rid, self.model.reactions)
                X[:,i] = Y[:,j]
                i += 1

        # In case 'get' is called several times
        if self.X.shape[0] > 0 and reduce == False:
            self.X = np.concatenate((self.X, X), axis=0)
            self.Y = np.concatenate((self.Y, Y), axis=0)
            self.LB = np.concatenate((self.LB, LB), axis=0)
        else:
            self.X, self.Y, self.LB = X, Y, LB
        self.size = self.X.shape[0]

        np.savetxt(os.path.join(self.output_dir,"temp_Pout.tsv"), self.Pout, delimiter='\t')

    def filter_measure(self, measure=[], verbose=False):
        # Keep only reaction fluxes in measure
        # Input:
        # - measure: a list of measured reaction fluxes
        # - reduce: when True the matrices are reduced considering
        #   the training set, all reactions not in the medium and
        #   having zero flux for all instances in the trainig set
        #   are removed
        # Output:
        # - updated self.Y (reduced to reaction fluxes in measure)
        # - self.Yall all reactions

        self.measure = measure if len(measure) > 0 else self.measure
        _, _, self.Pout, _, _ = \
        get_matrices_original(self.model, self.medium, self.measure, self.model.reactions)
        self.Yall = self.Y.copy()
        print(self.Y.shape)
        print(self.Pout.shape)
        print(self.measure)
        if self.measure != []:
            # Y = only the reaction fluxes that are in Vout
            Y = np.matmul(self.Y,np.transpose(self.Pout)) \
            if ('EXP') not in self.method else self.Y
            self.Y = Y
        if verbose:
            print('number of reactions: ', self.S.shape[1], self.Yall.shape[1])
            print('number of metabolites: ', self.S.shape[0])
            print('filtered measurements size: ',self.Y.shape[1])


    def filter_measure_return(self, measure=[], biomass_max=4, verbose=False):
        # Keep only reaction fluxes in measure
        # Input:
        # - measure: a list of measured reaction fluxes
        # - reduce: when True the matrices are reduced considering
        #   the training set, all reactions not in the medium and
        #   having zero flux for all instances in the trainig set
        #   are removed
        # Output:
        # - updated self.Y (reduced to reaction fluxes in measure)
        # - self.Yall all reactions

        self.measure = measure if len(measure) > 0 else self.measure
        # _, _, Pout, _, _ = \
        # get_matrices(self.model, self.medium, self.measure, self.model.reactions)


        self.Yall = self.Y.copy()

        Y = self.Y.copy()
        ObjY = np.zeros((self.Y.shape[0], self.Y.shape[1]))
        Pout = self.Pout.copy()
        for msr in measure:
            j = get_index_from_id(msr, self.model.reactions)
            Pout[j][j] = 1
            for i in range(self.Y.shape[0]):
                Y[i][j] = biomass_max
                ObjY[i][j] = biomass_max


        print("filter measure - Y   : ", self.Y.shape)
        print("filter measure - Pout: ", Pout.shape)
        print("filter measure - msr : ", self.measure)
        # if self.measure != []:
        #     # Y = only the reaction fluxes that are in Vout
        #     Y = np.matmul(self.Y,np.transpose(self.Pout)) \
        #     if ('EXP') not in self.method else self.Y

        if verbose:
            print('number of reactions: ', self.S.shape[1], self.Yall.shape[1])
            print('number of metabolites: ', self.S.shape[0])
            print('filtered measurements size: ',self.Y.shape[1])

        return Pout, Y, ObjY, self.Yall

