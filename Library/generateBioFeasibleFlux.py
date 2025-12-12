import sys
import pandas as pa
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px

from cobra.io import read_sbml_model, write_sbml_model
avogadro = 6.02214076e+23

from pathlib import Path
project_root = str(Path(__file__).resolve()).split('Library')[0]
print(project_root)

sys.path.append('/scratch/seaver/Collaborations/ElAlaoui_SFSU/Biochem_Informed_Mechanistic_Gradient_Opt')
from parameters import *

def get_rxn_ID(clm_name, row):
    if any(row[clm_name].endswith(y) for y in ['_f', '_r', '_i', '_o']):
        id_only = row[clm_name].rsplit("_", 1)[0]
    else:
        id_only = row[clm_name]

    return id_only

def load_fluxes(fluxes_file, verbose=True):
    fva_flux_df = pa.read_csv(fluxes_file, sep='\t')
    print("=============== load_fluxes =============")
    if verbose: print(fva_flux_df.describe())
    
    # remove reactions added by the flexible biomass package
    # columns_to_drop = list(fva_flux_df.filter(regex="FLEX_*|protein*|avg").columns)
    rows_to_drop = fva_flux_df.filter(regex="(?i).*flex.*", axis=0).index
    
    # fva_flux_df = fva_flux_df[fva_flux_df.columns.drop(columns_to_drop)]
    fva_flux_df = fva_flux_df.drop(index=rows_to_drop, errors='ignore')
    
    # Fill NA values as zero
    fva_flux_df[fva_flux_df.columns[1:]] = fva_flux_df[fva_flux_df.columns[1:]].fillna(0)
    # fva_flux_df[fva_flux_df.columns[1:]] = float(fva_flux_df[fva_flux_df.columns[1:]])

    # Old code for averaging FCA results
    # fva_flux_df['mean_flux'] = np.abs(fva_flux_df[fva_flux_df.columns[1:]]).sum(axis=1)
    # fva_flux_df['positive_count'] = (fva_flux_df[fva_flux_df.columns[1:]] != 0).sum(axis=1)
    # fva_flux_df['mean_flux'] = fva_flux_df['mean_flux']/fva_flux_df['positive_count']
    # fva_flux_df.drop(['positive_count'], axis=1, inplace=True)
    # fva_flux_df = fva_flux_df[['reaction', 'mean_flux']]

    if verbose: print(fva_flux_df.describe())

    if verbose:
        print(fva_flux_df.columns)
        print(fva_flux_df.head())

    fva_flux_df.rename(columns={'reaction':'rxn_ID'}, inplace=True)
    fva_flux_df.rename(columns={'max':'mean_flux'}, inplace=True)
    if verbose: print(fva_flux_df.head())
    # print(abc)

    print("=============== load_fluxes =============")
    return fva_flux_df

def load_scores_objKapp_relabVbf(scores_file, ctrl_trmt= 'Control', useRelab=True, day='all', value_col='value', trmt_column='treatment', verbose=False):
    
    sep = '\t' if '.tsv' in scores_file else ','
    scores_df = pa.read_csv(scores_file, sep = sep)
    if '00h' in scores_df['time_stamp'].unique():
        scores_df = scores_df[~scores_df['time_stamp'].isin(['01h', '00h'])]

    if verbose: print(scores_df.head())

    if useRelab:
        sep = '\t' if '.tsv' in relab_scores_file else ','
        relab_scores_df = pa.read_csv(relab_scores_file, sep = sep)
        # Convert units using avogadro number
        print(relab_scores_df.head())
        
        if '00h' in relab_scores_df['time_stamp'].unique():
            relab_scores_df = relab_scores_df[~relab_scores_df['time_stamp'].isin(['01h', '00h'])]
        relab_scores_df[value_col] = relab_scores_df[value_col].astype('float') / avogadro

    else:
        relab_scores_df = scores_df.copy()

    # Use score DF for K_app computation 
    scores_df[value_col] = scores_df[value_col].astype('float')
     # --> Use control treatment only
    control = scores_df[(scores_df[trmt_column] == ctrl_trmt) &
                            (scores_df[other_colm] == grpr3)]


    # drop all columns except score and reaction IDs
    control = control[[value_col, 'rxn_ID']]
    control = control.groupby('rxn_ID').mean()
    if verbose: print(control.head())

    # Keep value of interest
    # Compute Vbf for all time points and treatments 
    if day.lower() == 'all':
        relab_scores_df = relab_scores_df[(relab_scores_df[other_colm] == grpr3)]
        relab_scores_df[trmt_column] =\
                            relab_scores_df[trmt_column]+"_"+relab_scores_df['time_stamp']
    # Compute Vbf for one specific time point `day` and all treatments                        
    else:
        relab_scores_df = relab_scores_df[(relab_scores_df[other_colm] == grpr3) &
                                        (relab_scores_df['time_stamp'] == day)]
    
    # Keep only important columns
    relab_scores_df = relab_scores_df[[value_col, trmt_column, 'rxn_ID']]


    if verbose:
        print(relab_scores_df.head())
        print(control.head())

    return relab_scores_df, control

def load_scores_relabKapp_objVbf(scores_file, relab_scores_file='', ctrl_trmt= 'Control', useRelab=True, day='all', value_col='value', trmt_column='treatment', other_colm='tissue', grpr3='Leaf', verbose=False):
    
    sep = '\t' if '.tsv' in scores_file else ','
    print("Loading scores from ",scores_file)
    scores_df = pa.read_csv(scores_file, sep = sep)
    if '00h' in scores_df['time_stamp'].unique():
        scores_df = scores_df[~scores_df['time_stamp'].isin(['01h', '00h'])]

    if verbose: print(scores_df.head())

    if useRelab:
        sep = '\t' if '.tsv' in relab_scores_file else ','
        relab_scores_df = pa.read_csv(relab_scores_file, sep = sep)
        # Convert units using avogadro number
        print(relab_scores_df.head())
        
        if '00h' in relab_scores_df['time_stamp'].unique():
            relab_scores_df = relab_scores_df[~relab_scores_df['time_stamp'].isin(['01h', '00h'])]
        relab_scores_df[value_col] = relab_scores_df[value_col].astype('float') / avogadro

    else:
        relab_scores_df = scores_df.copy()
        
    relab_scores_df[value_col] = relab_scores_df[value_col].astype('float')

    apply_additional_filter =  (other_colm.lower() is not None) and (other_colm in relab_scores_df.columns)

    control_selector = (relab_scores_df[trmt_column] == ctrl_trmt)
    
    if(apply_additional_filter):
        control_selector = column_selector & (relab_scores_df[other_colm] == grpr3)

    #  --> Use control treatment only
    print(relab_scores_df.columns)
    print(relab_scores_df.head())

    control = relab_scores_df[control_selector]

    # drop all columns except score and reaction IDs
    control = control[[value_col, 'rxn_ID']]
    control = control.groupby('rxn_ID').mean()
    if verbose: print(control.head())

    # Keep value of interest
    # Compute Vbf for all time points and treatments

    if day.lower() == 'all':

        if(apply_additional_filter):
            scores_df = scores_df[(scores_df[other_colm] == grpr3)]

        scores_df[trmt_column] = scores_df[trmt_column].astype(str)+"_"+scores_df['time_stamp'].astype(str)

    # Compute Vbf for one specific time point `day` and all treatments                        
    else:
        scores_filter = (scores_df['time_stamp'] == day)
        if(apply_additional_filter):
            scores_filter = scores_filter & (scores_df[other_colm] == grpr3)

        scores_df = scores_df[scores_filter]
    
    # Keep only important columns
    scores_df = scores_df[[value_col, trmt_column, 'rxn_ID']]

    if verbose:
        print(scores_df.head())
        print(control.head())

    return scores_df, control

def duplicate_Vbf_values(co_model, Vbf_df, verbose=False):
    ## find all duplicated model reactions 
    reactions = [r.id for r in co_model.reactions]
    dup_reactions_df = pa.DataFrame(reactions, columns=['rxn_ID_dup'])
    dup_reactions_df['rxn_ID'] = dup_reactions_df.apply(lambda row: get_rxn_ID('rxn_ID_dup', row), axis=1)
    print(dup_reactions_df.groupby('rxn_ID'))

    Vbf_df = Vbf_df.merge(dup_reactions_df, on='rxn_ID', how='left')
    Vbf_df.drop('rxn_ID', axis=1, inplace=True)
    Vbf_df.rename(columns={'rxn_ID_dup': 'rxn_ID'}, inplace=True)

    # Make rxn_ID colmn the first colm of the DF
    clms = list(Vbf_df.columns)
    clms.remove('rxn_ID')
    clms = ['rxn_ID'] + clms
    if verbose: print(Vbf_df.head())

    return Vbf_df[clms]

def generate_all_df(day, fluxes_file, scores_file, relab_scores_file="", value_col='value', trmt_column='treatment', ctrl_trmt='Control', other_colm='tissue', grpr3='Leaf', useRelab=True, verbose=False):
    # Reaction score for non-duplicated model
    scores_df, control = load_scores_relabKapp_objVbf(scores_file, relab_scores_file, ctrl_trmt, useRelab, day, value_col, trmt_column,other_colm=other_colm, grpr3=grpr3, verbose=verbose)
    # Fluxes for duplicated model
    fluxes_dup = load_fluxes(fluxes_file)
    verbose is True
    if verbose:
        print("Control DF: \n", control.head())
        print("Scores DF: \n", scores_df.head())
        print("Fluxes DF: \n", fluxes_dup.head())

    test_ids = ['rxn20617_c0','rxn20617_c0_f','rxn20617_c0_r',
                'rxn20617_d0','rxn20617_d0_f','rxn20617_d0_r']

    fluxes_dup['short_rxn_ID'] = fluxes_dup['rxn_ID'].str.replace('(_f|_r)$', '', regex=True)

    Vbf_df = fluxes_dup.merge(
        control,
        left_on='short_rxn_ID',  # Match the stripped ID from the fluxes DataFrame
        right_on='rxn_ID',     # Match the original ID from the control DataFrame
        how='left',             # Keep all rows from fluxes_dup
        suffixes=('_flx', '_ctrl') # Use suffixes to differentiate columns if needed
    )

    print("Columns: ",Vbf_df.columns)
    # Compute K_app
    # Vbf_df = control.merge(fluxes_dup, on='rxn_ID', how='left')
    print("Vbf_df: ", Vbf_df.head())

    for rxn in test_ids:
        if((Vbf_df['rxn_ID'] == rxn).any()):
            # if(rxn in Vbf_df.index):
            single_reaction_row = Vbf_df[Vbf_df['rxn_ID'] == rxn]
            mean_flux_value = single_reaction_row['mean_flux'].iloc[0]
            print("Vbf 1: ",rxn,mean_flux_value)
        if((fluxes_dup['rxn_ID'] == rxn).any()):
            single_reaction_row = fluxes_dup[fluxes_dup['rxn_ID'] == rxn]
            mean_flux_value = single_reaction_row['mean_flux'].iloc[0]
            print("F 1: ",rxn,mean_flux_value)

    Vbf_df['kapp'] = np.abs(Vbf_df['mean_flux'])/Vbf_df[value_col]

    Vbf_df.rename(columns={value_col:'ctrl_score'}, inplace=True)
    if verbose:
        print("----- K_app describe: ")
        print(Vbf_df['kapp'].describe())

    # Compute V_bf for every treatment
    scores_df['rxn_ID'] = scores_df['rxn_ID'].astype(str)
    Vbf_df['rxn_ID'] = Vbf_df['rxn_ID'].astype(str)
    if verbose:
        print('There are ', Vbf_df['rxn_ID'].nunique(), ' reactions in Vbf_df DF')
        print('There are ', scores_df['rxn_ID'].nunique(), ' reactions in scores_df DF')

    # align K_app values with the reactions' scores for all <treatments_timePoint>
    # Vbf_df = pa.merge(Vbf_df, scores_df, on='rxn_ID', how='left')
    Vbf_df = Vbf_df.merge(
        scores_df,
        left_on='short_rxn_ID',  # The non-directional key in the Vbf_df
        right_on='rxn_ID',      # The non-directional key in the other_df
        how='left',             # Keep all rows from Vbf_df
        suffixes=('', '_sco') # Use suffixes to manage column naming conflicts
    )
    print(Vbf_df.head())
    print("Columns: ",Vbf_df.columns)
    Vbf_df['v'] = Vbf_df[value_col] * Vbf_df['kapp']
    Vbf_df.rename(columns={value_col:"score"}, inplace=True)
    
    # pivot the dataframe to create a column for each treatment
    ind = ['rxn_ID', 'mean_flux', 'kapp']
    col = [trmt_column]
    val = ['score', 'v']
    Vbf_df = Vbf_df.pivot(index=ind, columns=col, values=val)
    Vbf_df.columns = Vbf_df.columns.map('{0[0]}_{0[1]}'.format)
    Vbf_df = Vbf_df.reset_index()

    # Clean values 
    Vbf_df.replace([np.inf, -np.inf], 0, inplace=True)
    Vbf_df = Vbf_df.fillna(0)
    Vbf_df = Vbf_df.drop(columns=['v_nan'])
    Vbf_df = Vbf_df.drop(columns=['score_nan'])

    print(Vbf_df.head())
    # print(Vbf_df.columns)
    # print(abc)
    
    # plot V_bf to compare
    # v_colms = [x for x in Vbf_df.columns if 'v_' in x]
    # v_colms = ['rxn_ID'] + v_colms
    # Vbf_df[v_colms].set_index('rxn_ID').plot()
    # plt.show()

    return Vbf_df

def vbf_reset(Vbf_df, treatments, threshold): 
    for trmt in treatments:
        trmt_mean = Vbf_df['v_'+trmt].mean()
        Vbf_df.loc[(Vbf_df['v_'+trmt] <= threshold) & (Vbf_df['score_'+trmt] > 0), 'v_'+trmt] = trmt_mean
    print(Vbf_df.head(10))
    return Vbf_df


def vbf_histo1(Vbf_df, treatments): 
    Vbf_df_long = pa.melt(Vbf_df, id_vars=['rxn_ID', 'mean_flux'], value_vars=list(Vbf_df.filter(regex="v_*|score_*")))
    Vbf_df_long[['type', 'treatment', 'day']] = Vbf_df_long['variable'].str.split('_', expand=True)

    Vbf_df_long.drop(columns=['variable'], inplace=True)

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots


    
    fig = make_subplots(rows=5, cols=5, 
                        subplot_titles=treatments)
    i, j = 1, 1 
    for condition in treatments:
        Vbf_df_temp = Vbf_df[['rxn_ID', 'mean_flux', 'v_'+condition, 'score_'+condition]]
        Vbf_df_temp = Vbf_df_temp[(Vbf_df_temp['score_'+condition] > 0) & (Vbf_df_temp['mean_flux'] > 0)]
        trmt, day = condition.split('_')[0], condition.split('_')[1]

        

        trace = go.Histogram(x=Vbf_df_temp['v_'+condition],
                                name = condition
                              # xbins=dict(
                              # start='1969-11-15',
                              # end='1972-03-31',
                              # size='M18'), # M18 stands for 18 months
                              # autobinx=False
                             )
        fig.add_trace(trace, i, j)

        print(i, j)

        j += 1
        if (j==6):
            i+=1 
            j = 1
        if (i==6): 
            i = 1
        
        
          
    fig.show()

    print(abc)

def vbf_histo2(Vbf_df, treatments): 
    Vbf_df_long = pa.melt(Vbf_df, id_vars=['rxn_ID', 'mean_flux'], value_vars=list(Vbf_df.filter(regex="v_*|score_*")))
    Vbf_df_long[['type', 'treatment', 'day']] = Vbf_df_long['variable'].str.split('_', expand=True)

    Vbf_df_long.drop(columns=['variable'], inplace=True)

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots



    fig = make_subplots(rows=5, cols=5, 
                        subplot_titles=treatments)
    i, j = 1, 1 

    print(Vbf_df.shape)
    for condition in treatments:
        Vbf_df_temp = Vbf_df[['rxn_ID', 'mean_flux', 'v_'+condition, 'score_'+condition]]
        Vbf_df_temp = Vbf_df_temp[(Vbf_df_temp['score_'+condition] > 0) & (Vbf_df_temp['mean_flux'] >= 0) & (Vbf_df_temp['v_'+condition] <= 0)]

        print(len(Vbf_df_temp['rxn_ID'].unique()))
        # print(abc)

        trmt, day = condition.split('_')[0], condition.split('_')[1]

        print("before Vbf ", Vbf_df_temp.shape)
        print(Vbf_df_temp['v_'+condition][(Vbf_df_temp['v_'+condition] <= 10)].shape)
       
        trace = go.Histogram(x=Vbf_df_temp['v_'+condition][(Vbf_df_temp['v_'+condition] <= 10) ],
                                name = condition
                              # xbins=dict(
                              # start='1969-11-15',
                              # end='1972-03-31',
                              # size='M18'), # M18 stands for 18 months
                              # autobinx=False
                             )
        fig.add_trace(trace, i, j)

        print(i, j)

        j += 1
        if (j==6):
            i+=1 
            j = 1
        if (i==6): 
            i = 1
        
        
          
    fig.show()

    # print(abc)

def vbf_hists(Vbf_df, treatments): 
    Vbf_df_long = pa.melt(Vbf_df, id_vars=['rxn_ID', 'mean_flux'], value_vars=list(Vbf_df.filter(regex="v_*|score_*")))
    Vbf_df_long[['type', 'treatment', 'day']] = Vbf_df_long['variable'].str.split('_', expand=True)

    Vbf_df_long.drop(columns=['variable'], inplace=True)

    print('-'*30)
    print(Vbf_df_long.head(5))

    fig = px.histogram(Vbf_df_long[(Vbf_df_long['type']=='v') & (Vbf_df_long['value']>0) & (Vbf_df_long['mean_flux']>0)], 
                        x="value", 
                        color="type", 
                        facet_col = 'day', 
                        facet_row = 'treatment')
    fig.show()

    # import plotly.graph_objects as go
    # from plotly.subplots import make_subplots

    # x = ['1970-01-01', '1970-01-01', '1970-02-01', '1970-04-01', '1970-01-02',
    #      '1972-01-31', '1970-02-13', '1971-04-19']

    # fig = make_subplots(rows=5, cols=5)

    # trace0 = go.Histogram(x=x, nbinsx=4)
    # trace1 = go.Histogram(x=x, nbinsx = 8)
    # trace2 = go.Histogram(x=x, nbinsx=10)
    # trace3 = go.Histogram(x=x,
    #                       xbins=dict(
    #                       start='1969-11-15',
    #                       end='1972-03-31',
    #                       size='M18'), # M18 stands for 18 months
    #                       autobinx=False
    #                      )
    # trace4 = go.Histogram(x=x,
    #                       xbins=dict(
    #                       start='1969-11-15',
    #                       end='1972-03-31',
    #                       size='M4'), # 4 months bin size
    #                       autobinx=False
    #                       )
    # trace5 = go.Histogram(x=x,
    #                       xbins=dict(
    #                       start='1969-11-15',
    #                       end='1972-03-31',
    #                       size= 'M2'), # 2 months
    #                       autobinx = False
    #                       )

    # fig.add_trace(trace0, 1, 1)
    # fig.add_trace(trace1, 1, 2)
    # fig.add_trace(trace2, 2, 1)
    # fig.add_trace(trace3, 2, 2)
    # fig.add_trace(trace4, 3, 1)
    # fig.add_trace(trace5, 3, 2)

    # fig.show()

    print(abc)

def save_to_file(Vbf_df, spc, grpr3, day, saveTo, useRelab=False):
    # Write to file
    v_colms = [x for x in Vbf_df.columns if 'v_' in x]
    v_colms = ['rxn_ID'] + v_colms
    name = [spc, 'complexFix', grpr3, day, 'restrMedia', 'Vbf', 'maxCtrl']
    if useRelab:
        name = name + ['mixedRelab']
    if 'atha' in spc:
        name = name + ['fullmodel']

    print("Saving Vbf to",saveTo)
    print("Warning, dropping empty rows")
    Vbf_df = Vbf_df.dropna(subset=['rxn_ID'])
    Vbf_df[v_colms].to_csv(saveTo, index=False,sep='\t')
    saveTo = saveTo.replace(".tsv", "_kapp.tsv")
    Vbf_df.to_csv(saveTo, index=False,sep='\t')

def generate_Vbf(param,model=None):
    print(param)
    spc = param.spc
    day = param.time_stamp
    grpr3 = param.other_colm_value
    treatments = param.treatments
    expFolder = param.expFolder
    verbose=True

    ctrl_trmt = param.ctrl_trmt
    other_colm = param.other_colm
    value_col = param.value_col
    useRelab = param.useRelab
    trmt_column = param.trmt_column

    ## FVA fluxes
    fluxes_file = param.results_folder+'fva.tsv'

    ## Relative abundance
    if(hasattr(param,'relab_scores_file')):
        relab_scores_file = param.relab_scores_file
    else:
        relab_scores_file = None

    ## Objective abundance
    scores_file = param.scores_file
    
    Vbf_df = generate_all_df(day, fluxes_file, scores_file,relab_scores_file, value_col, trmt_column, ctrl_trmt, other_colm, grpr3, useRelab, verbose)

    # Vbf_df = duplicate_Vbf_values(model, Vbf_df, verbose)

    save_to_file(Vbf_df, spc, grpr3, day, param.VbfFile, useRelab)

if __name__ == '__main__':
    param = Parameters_ML_ColdResponse()
    generate_Vbf(param)

    # vbf_histo2(Vbf_df, treatments)
    # vbf_hist(Vbf_df, treatments)
    # Vbf_df_new = vbf_reset(Vbf_df, treatments, 0.004)
    # save_to_file(Vbf_df, spc, grpr3, day, expFolder)

