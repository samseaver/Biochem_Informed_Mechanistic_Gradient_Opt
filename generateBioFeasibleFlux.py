import sys
import pandas as pa
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px

avogadro = 6.02214076e+23

from pathlib import Path
project_root = str(Path(__file__).resolve()).split('Library')[0]

sys.path.append(project_root)
from parameters import *

# This function is used because in FVA, even with a pfba_factor
# reactions in a thermodynamic loop will display a consistently high
# and articifical flux. Here we attempt to find the value of this 
# by assuming that all thermodynamic loops would exhibit the same high
# flux and return the most common ('mode') high value
def determine_ceiling_threshold(df, col='max'):
    # Detects the "solver wall" artifact
    top_half = df[df[col] > df[col].median()].copy()
    top_half['rounded'] = top_half[col].round(1)
    if top_half.empty: return df[col].max()
    common_artifact_value = top_half['rounded'].mode()[0]
    return common_artifact_value * 0.99

def load_fluxes(parameters, scores_df, verbose=False):
    fluxes_file = f"{parameters.results_folder}fva.tsv"
    print("Loading fluxes from "+fluxes_file)

    fva_df = pa.read_csv(fluxes_file, sep='\t')
    
    # because of rounding, there's potentially values that are -0.000
    fva_df['max'] = fva_df['max'].abs()

    # establish base reaction id without direction
    fva_df['base_id'] = fva_df['reaction'].str.replace(r'_[rfio]$', '', regex=True)
    
    # Identify most common FVA value (from loops)
    # Use cutoff to find median flux for imputation later
    # Median should to be calculated now before net flux calculations
    cutoff = determine_ceiling_threshold(fva_df, col='max')
    # Find set of reactions that would be used to computed median flux (from non-zero fluxes)
    # But calculate median after calculating net flux
    mask_real = (fva_df['max'] > 1e-6) & (fva_df['max'] < cutoff)

    # ---------------------------------------------------------
    # Calculate Net Flux for Reversible Pairs
    # ---------------------------------------------------------

    # 1. NEW: Force expand=False so it returns a clean Series, not a DataFrame
    fva_df['base_rxn_cpt_id'] = fva_df['reaction'].str.replace(r'_([frio])$', '', regex=True)
    fva_df['base_rxn_direction'] = fva_df['reaction'].str.extract(r'_([frio])$', expand=False)

    # 2. Split into Forward/Outward ('f', 'o') and Reverse/Inward ('r', 'i')
    fo_df = fva_df[fva_df['base_rxn_direction'].isin(['f', 'o'])].set_index('base_rxn_cpt_id')
    ri_df = fva_df[fva_df['base_rxn_direction'].isin(['r', 'i'])].set_index('base_rxn_cpt_id')

    # 3. Find Reversible Pairs
    reversible_ids = fo_df.index.intersection(ri_df.index)
    print(f"Calculating net Flux calculation for {len(reversible_ids)} reversible/transport reaction pairs.")

    # 4. Calculate Net Flux
    fo_vals = fo_df.loc[reversible_ids, 'max']
    ri_vals = ri_df.loc[reversible_ids, 'max']
    net_flux = (fo_vals - ri_vals).abs()

    # 5. Determine Dominant Direction
    forward_dominant = fo_vals > ri_vals

    # 6. Build an exact mapping dictionary (Bypasses all Pandas index bugs)
    id_fo = fo_df.loc[reversible_ids, 'reaction'].values
    id_ri = ri_df.loc[reversible_ids, 'reaction'].values
    
    update_dict = {}
    for i in range(len(reversible_ids)):
        if forward_dominant.iloc[i]:
            update_dict[id_fo[i]] = net_flux.iloc[i]
            update_dict[id_ri[i]] = 0.0
        else:
            update_dict[id_fo[i]] = 0.0
            update_dict[id_ri[i]] = net_flux.iloc[i]

    # 7. Apply Updates forcefully using the dictionary
    fva_df['max'] = fva_df.apply(lambda row: update_dict.get(row['reaction'], row['max']), axis=1)

    # Calculate median_flux to use for imputation
    median_flux = fva_df.loc[mask_real, 'max'].median()
    print(f"Median Flux For Imputation: {median_flux:.2f}")
    
    # =========================================================
    # Conditional Imputation For Active Enzymes
    # Median is computed in load_fluxes
    # =========================================================
    
    # Map the scores, but DO NOT use fillna(0) yet! Leave missing scores as NaN.
    # Collapse the 57 conditions into a single maximum active score per enzyme
    unique_score_map = scores_df.groupby('base_id')[parameters.value_col].max()
    fva_df['active_score'] = fva_df['base_id'].map(unique_score_map)
    
    # Identify Irreversible Reactions that are blocked AND ACTIVE
    irreversible_mask = ~fva_df['base_rxn_cpt_id'].isin(reversible_ids)
    
    # Check that active_score > 0
    blocked_irrev = irreversible_mask & (fva_df['max'] <= 1e-6) & (fva_df['active_score'] > 0)
    print(f"Imputing {blocked_irrev.sum()} active, blocked irreversible reactions.")
    fva_df.loc[blocked_irrev, 'max'] = median_flux

    # Identify Reversible Pairs where BOTH directions are 0 AND ACTIVE
    fva_df.set_index('reaction', inplace=True, drop=False)

    # Grab current values for all pairs
    current_fo = fva_df.loc[id_fo, 'max'].values
    current_ri = fva_df.loc[id_ri, 'max'].values
    
    # Grab the enzyme activity scores for these pairs
    pair_scores = fva_df.loc[id_fo, 'active_score'].values

    # Check where BOTH are effectively zero AND the enzyme is active
    both_zero_mask = (current_fo <= 1e-6) & (current_ri <= 1e-6) & (pair_scores > 0)
    both_zero_count = both_zero_mask.sum()

    print(f"Imputing {both_zero_count} active reversible pairs")

    # =========================================================
    # Master Imputation Block (Active Pairs + Scored NAs)
    # =========================================================

    # Get IDs for the reversible pairs where BOTH directions are 0 (and active)
    ids_to_impute_fo = id_fo[both_zero_mask]
    ids_to_impute_ri = id_ri[both_zero_mask]

    # Get IDs for reactions where FVA is NA, but we HAVE a reaction score
    fva_na_but_scored = fva_df['max'].isna() & fva_df['active_score'].notna()
    ids_na_scored = fva_df.index[fva_na_but_scored].tolist()

    # Combine all targets into a single list
    all_impute_targets = list(ids_to_impute_fo) + list(ids_to_impute_ri) + ids_na_scored

    # Impute the FULL median capacity in one single operation
    fva_df.loc[all_impute_targets, 'max'] = median_flux

    print(f"Imputed median flux to {len(all_impute_targets)} total reaction tracks "
          f"({both_zero_mask.sum()} reversible pairs + {fva_na_but_scored.sum()} FVA NAs).")
    
    # Mask out reactions with no score for the Vbf calculation later
    # You can save this boolean mask to use when you build your Vbf matrix
    fva_df['calculate_vbf'] = fva_df['active_score'].notna()

    # Clean up helper columns
    fva_df.drop(columns=['base_rxn_cpt_id', 'base_rxn_direction'], inplace=True)

    if verbose: print(fva_df.head())
    fva_df.rename(columns={'reaction':'rxn_ID'}, inplace=True)

    return fva_df

def load_scores(parameters, verbose=False):
    
    sep = '\t' if '.tsv' in parameters.scores_file else ','
    print("Loading scores from ",parameters.scores_file)
    scores_df = pa.read_csv(parameters.scores_file, sep = sep)

    if verbose: print(scores_df.head())

    # if parameters.useRelab:
    #     sep = '\t' if '.tsv' in parameters.relab_scores_file else ','
    #     relab_scores_df = pa.read_csv(parameters.relab_scores_file, sep = sep)
    #     relab_scores_df[parameters.value_col] = relab_scores_df[parameters.value_col].astype('float') / avogadro

    relab_scores_df = scores_df.copy()
    relab_scores_df[parameters.value_col] = relab_scores_df[parameters.value_col].astype('float')
    
    control_selector = relab_scores_df[parameters.trmt_column].str.contains(parameters.ctrl_trmt, regex=False)
    control = relab_scores_df[control_selector]

    if(control.empty):
        print(f"ctrl_trmt parameter '{parameters.ctrl_trmt}' cannot be found in '{parameters.trmt_column}' column in data")
        sys.exit(1)

    # drop all columns except score and reaction IDs
    control = control[[parameters.value_col, 'rxn_ID']]
    control = control.groupby('rxn_ID').mean()
    control = control.reset_index()
    if verbose: print(control.head())
    
    # Keep only important columns
    scores_df = scores_df[[parameters.value_col, parameters.trmt_column, 'rxn_ID']]
    if verbose:
        print(scores_df.head())
        print(control.head())

    return scores_df, control

def generate_kapp_vbf(parameters, verbose=False):
    # Reaction score for non-duplicated model
    scores_df, control_df = load_scores(parameters, verbose=verbose)
    
    scores_df.rename(columns={'rxn_ID': 'base_id'}, inplace=True)
    scores_df.rename(columns={'mean_value': parameters.value_col}, inplace=True)
    scores_df = scores_df[scores_df['base_id'].str.contains(r'rxn\d{5}', regex=True)]

    control_df.rename(columns={'rxn_ID': 'base_id'}, inplace=True)
    control_df.rename(columns={parameters.value_col: 'average_rs'}, inplace=True)
    control_df = control_df[control_df['base_id'].str.contains(r'rxn\d{5}', regex=True)]

    # Fluxes for duplicated model
    # reaction scores are used for imputation of fluxes in active enzymes
    fva_df = load_fluxes(parameters, scores_df)
    
    if verbose:
        print("Control DF: \n", control_df.head())
        print("Scores DF: \n", scores_df.head())
        print("Fluxes DF: \n", fva_df.head())

    kapp_df = pa.merge(fva_df, control_df, on='base_id', how='inner', indicator=True)
    # Summarize the results
    # print(kapp_df['_merge'].value_counts())

    # Calculate Kapp = Max / Average RS
    # This will generate np.inf where average_rs is 0 (due to zero abundance)
    kapp_df['kapp'] = kapp_df['max'] / kapp_df['average_rs']

    # Find rows where Kapp is specifically Infinite (positive or negative)
    inf_mask = np.isinf(kapp_df['kapp'])
    inf_rows = kapp_df[inf_mask]

    print(f"Found {len(inf_rows)} rows where Kapp is Infinite (Max > 0, Avg_RS = 0).")
    if not inf_rows.empty:
        print("Sample rows with Infinite Kapp:")
        print(inf_rows[['rxn_ID', 'max', 'average_rs', 'kapp']].head(10))

        # force Kapp to 0.0 where it is currently np.inf 
        # Logic: "If not expressed in control, Kapp is 0."
        kapp_df.replace([np.inf, -np.inf], 0.0, inplace=True)
        kapp_df['kapp'] = kapp_df['kapp'].fillna(0.0)

    # It remains that the only other reason kapp is zero after this
    # is if the max flux of a single direction for a reversible reaction
    # was set to zero (see load_fluxes)

    if verbose:
        print(kapp_df['kapp'].describe())

    kapp_df = kapp_df[['rxn_ID', 'base_id', 'max', 'kapp', 'average_rs']].drop_duplicates()
    vbf_df = pa.merge(kapp_df, scores_df, on='base_id', how='inner',indicator=True)
    vbf_df['vbf'] = vbf_df['kapp'] * vbf_df[parameters.value_col]
    
    # pivot the dataframe to create a column for each treatment
    vbf_df = vbf_df.rename(columns={parameters.value_col: 'rs'})
    ind = ['rxn_ID', 'max', 'average_rs', 'kapp']
    col = [parameters.trmt_column]
    val = ['rs', 'vbf']
    vbf_df = vbf_df.pivot(index=ind, columns=col, values=val)
    vbf_df.columns = vbf_df.columns.map('{0[0]}_{0[1]}'.format)
    vbf_df = vbf_df.reset_index()

    return vbf_df

def generate_Vbf(parameters):
    verbose=False

    Vbf_df = generate_kapp_vbf(parameters, verbose)

    Vbf_df.to_csv(f"{parameters.results_folder}vbf.tsv", index=False,sep='\t')

if __name__ == '__main__':
    vbf_parameters = Parameters_VBF()
    generate_Vbf(vbf_parameters)