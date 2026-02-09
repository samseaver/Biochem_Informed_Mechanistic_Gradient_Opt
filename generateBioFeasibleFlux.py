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

def load_fluxes(fluxes_file, verbose=False):
    print("Loading fluxes from "+fluxes_file)

    fva_df = pa.read_csv(fluxes_file, sep='\t')
    
    # remove reactions added by the flexible biomass package
    rows_to_drop = fva_df.filter(regex="(?i).*flex.*", axis=0).index
    fva_df = fva_df.drop(index=rows_to_drop, errors='ignore')
    
    # because of rounding, there's potentially values that are -0.000
    fva_df['max'] = fva_df['max'].abs()
    
    # Fill NA values as zero
    fva_df[fva_df.columns[1:]] = fva_df[fva_df.columns[1:]].fillna(0)

    # Identify most common FVA value (from loops)
    # Use cutoff to find median flux for imputation later
    # Median should to be calculated now before net flux calculations
    cutoff = determine_ceiling_threshold(fva_df, col='max')
    # Calculate Median Threshold (from non-zero fluxes)
    mask_real = (fva_df['max'] > 1e-6) & (fva_df['max'] < cutoff)
    median_flux = fva_df.loc[mask_real, 'max'].median()
    print(f"Median Flux For Imputation: {median_flux:.2f}")

    # ---------------------------------------------------------
    # Calculate Net Flux for Reversible Pairs
    # ---------------------------------------------------------

    # 1. Identify the "Base ID" (e.g., 'rxn00001_c0') and Direction ('f' or 'r')
    # We temporarily add helper columns
    fva_df['base_rxn_cpt_id'] = fva_df['reaction'].str.replace(r'_[fr]$', '', regex=True)
    fva_df['base_rxn_direction'] = fva_df['reaction'].str.extract(r'_([fr])$')

    # 2. Split into Forward and Reverse Series for alignment
    # We index by 'base_match' to line them up perfectly
    f_fluxes = fva_df[fva_df['base_rxn_direction'] == 'f'].set_index('base_rxn_cpt_id')['max']
    r_fluxes = fva_df[fva_df['base_rxn_direction'] == 'r'].set_index('base_rxn_cpt_id')['max']

    # 3. Find Reversible Pairs (IDs that exist in BOTH forward and reverse)
    reversible_ids = f_fluxes.index.intersection(r_fluxes.index)
    print(f"Calculating net Flux calculation for {len(reversible_ids)} reversible reaction pairs.")

    # 4. Calculate Net Flux
    # Net = |Forward - Reverse|
    f_vals = f_fluxes.loc[reversible_ids]
    r_vals = r_fluxes.loc[reversible_ids]
    net_flux = (f_vals - r_vals).abs()

    # 5. Determine Dominant Direction
    # True if Forward > Reverse, False if Reverse > Forward
    forward_dominant = f_vals > r_vals

    # 6. Build the Update Series
    # We create two Series: one for Forward IDs, one for Reverse IDs
    id_f = reversible_ids + '_f'
    id_r = reversible_ids + '_r'

    # Prepare values: 
    # If Fwd Dominant: Fwd=Net, Rev=0
    # If Rev Dominant: Fwd=0, Rev=Net
    new_f_vals = pa.Series(np.where(forward_dominant, net_flux, 0.0), index=id_f)
    new_r_vals = pa.Series(np.where(forward_dominant, 0.0, net_flux), index=id_r)

    # Combine into one update packet
    updates = pa.concat([new_f_vals, new_r_vals])
    updates.name = 'max'

    # 7. Apply Updates Safely
    # We set the index to 'reaction' so .update() can match IDs automatically
    fva_df.set_index('reaction', inplace=True)
    fva_df.update(updates)
    fva_df.reset_index(inplace=True)

    # =========================================================
    # Conditional Imputation
    # =========================================================

    # Identify Irreversible Reactions that are blocked
    irreversible_mask = ~fva_df['base_rxn_cpt_id'].isin(reversible_ids)
    blocked_irrev = irreversible_mask & (fva_df['max'] <= 1e-6)
    print(f"Imputing {blocked_irrev.sum()} blocked irreversible reactions.")
    fva_df.loc[blocked_irrev, 'max'] = median_flux

    # Identify Reversible Pairs where BOTH directions are 0
    # (These are the 'Net Zero Loops' or 'Blocked Pairs')
    # We must check the current values in fva_df
    fva_df.set_index('reaction', inplace=True, drop=False)

    # Grab current values for all pairs
    current_f = fva_df.loc[id_f, 'max'].values
    current_r = fva_df.loc[id_r, 'max'].values

    # Check where BOTH are effectively zero
    both_zero_mask = (current_f <= 1e-6) & (current_r <= 1e-6)
    both_zero_count = both_zero_mask.sum()

    print(f"Imputing {both_zero_count} reversible pairs")

    # Impute ONLY the Forward direction for these pairs
    # We leave Reverse as 0 to avoid creating a new loop
    ids_to_impute_f = id_f[both_zero_mask]
    fva_df.loc[ids_to_impute_f, 'max'] = median_flux
    ids_to_impute_r = id_r[both_zero_mask]
    fva_df.loc[ids_to_impute_r, 'max'] = median_flux

    # Clean up helper columns
    fva_df.drop(columns=['base_rxn_cpt_id', 'base_rxn_direction'], inplace=True)
    fva_df.rename(columns={'reaction':'rxn_ID'}, inplace=True)
    
    if verbose: print(fva_df.head())
    return fva_df

def load_scores(parameters, verbose=False):
    
    sep = '\t' if '.tsv' in parameters.scores_file else ','
    print("Loading scores from ",parameters.scores_file)
    scores_df = pa.read_csv(parameters.scores_file, sep = sep)

    if verbose: print(scores_df.head())

    if parameters.useRelab:
        sep = '\t' if '.tsv' in parameters.relab_scores_file else ','
        relab_scores_df = pa.read_csv(parameters.relab_scores_file, sep = sep)
        relab_scores_df[parameters.value_col] = relab_scores_df[parameters.value_col].astype('float') / avogadro
    else:
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
    scores_df.rename(columns={'mean_value': 'reaction_score'}, inplace=True)
    scores_df = scores_df[scores_df['base_id'].str.contains(r'rxn\d{5}', regex=True)]

    control_df.rename(columns={'rxn_ID': 'base_id'}, inplace=True)
    control_df.rename(columns={'reaction_score': 'average_rs'}, inplace=True)
    control_df = control_df[control_df['base_id'].str.contains(r'rxn\d{5}', regex=True)]

    # Fluxes for duplicated model
    fluxes_file = f"{parameters.results_folder}fva.tsv"
    fva_df = load_fluxes(fluxes_file)
    fva_df['base_id'] = fva_df['rxn_ID'].str.replace(r'_[rfio]$', '', regex=True)
    
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
    vbf_df['vbf'] = vbf_df['kapp'] * vbf_df['reaction_score']
    
    # pivot the dataframe to create a column for each treatment
    vbf_df = vbf_df.rename(columns={'reaction_score': 'rs'})
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