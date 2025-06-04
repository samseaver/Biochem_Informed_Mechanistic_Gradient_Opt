import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pa

import plotly.express as px

import plotly.io as pio
pio.templates.default = "plotly_white" #"none"

sv = 15
data_file = f"/Users/selalaoui/Projects/AMN/omic_amn_mm/Result/sv{sv}/processed_sv{sv}.csv"
data = pa.read_csv(data_file)
species = ["Poplar", "Sorghum"]
treatments = ["FeEX", "FeLim", "ZnEx", "ZnLim"]
treatments = ["FeLim"]

# color_map = {"FeLim": "rgba(181, 148, 182, 1)", "FeEX": "rgba(99, 160, 136, 1)", "Control": "rgba(75, 74, 103, 1)", 
# "ZnEx": "rgba(190, 110, 70, 1)", "ZnLim": "rgba(27, 64, 121, 1)"}

# color_map = {"FeLim": "rgba(254, 95, 85, 1)", "FeEX": "rgba(0, 168, 150, 1)", "Control": "rgba(53, 82, 74, 1)", 
# "ZnEx": "rgba(255, 210, 117, 1)", "ZnLim": "rgba(81, 187, 254, 1)"}

color_map = {'Control': 'rgba(64, 61, 88, 1)', 'FeEX': 'rgba(0, 175, 181, 1)', 'FeLim': 'rgba(170, 189, 140, 1)',
 'ZnEx': 'rgba(206, 83, 116, 1)', 'ZnLim': 'rgba(243, 155, 109, 1)'}

plotEach, plotSep = False, True

# ---------------------------- Plots each value and each treatment in a separate subplot
if plotSep: 
    titles = [f'{treat}' for spec in species[:1] for treat in treatments]
    specs = []
    for spc in species:
        specs.append([{"secondary_y": True} for _ in treatments])


    fig = make_subplots(rows=len(species), cols=len(treatments),
                        specs=specs, 
                        subplot_titles=titles, 
                        shared_xaxes=True,
                        shared_yaxes=True,
                        vertical_spacing=0.02,
                        horizontal_spacing=0.02)

    trmtLegend = True
    carbLegend = False
    ctrCarbLegend = False
    ctrLegend = True
    row = 0
    for spc in species:
        row += 1
        col = 0 
        for trmt in treatments: 
            col += 1
            temp_data = data[(data["treatment"].isin([trmt, "Control"])) & (data["spc"]==spc)]

            fig.add_trace(
                        go.Scatter(x=temp_data["day"][temp_data["treatment"] == "Control"], 
                                y=temp_data["pred_flux"][temp_data["treatment"] == "Control"], 
                                name="Control - Predicted Flux", 
                                marker=dict(
                                    color=color_map["Control"],  # Marker color
                                    size=7,       # Marker size
                                    symbol='circle' # Marker shape
                                ),
                                line=dict(
                                    color=color_map["Control"],  # Line color
                                    width=2       # Line width
                                ), showlegend=ctrLegend
                            ), row=row, col=col, secondary_y=False#, showlegend=True  
                    )
            fig.add_trace(
                        go.Scatter(x=temp_data["day"][temp_data["treatment"] == trmt], 
                                y=temp_data["pred_flux"][temp_data["treatment"] == trmt], 
                                name=f"{trmt} - Predicted Flux", 
                                marker=dict(
                                    color=color_map[trmt],  # Marker color
                                    size=7,       # Marker size
                                    symbol='circle' # Marker shape
                                ),
                                line=dict(
                                    color=color_map[trmt],  # Line color
                                    width=2       # Line width
                                ), showlegend=trmtLegend
                            ), row=row, col=col, secondary_y=False,
                    )
            
            fig.add_trace(
                        go.Scatter(x=temp_data["day"][temp_data["treatment"] == "Control"], 
                                y=temp_data["carbon_flux"][temp_data["treatment"] == "Control"], 
                                name=f"Conrol - Carbon Imbalance",
                                marker=dict(
                                    color=color_map["Control"],  # Marker color
                                    size=7,       # Marker size
                                    symbol='circle' # Marker shape
                                ),
                                line=dict(
                                    color=color_map["Control"],  # Line color
                                    width=2,       # Line width
                                    dash='dot'
                                ), showlegend=ctrCarbLegend

                            ), row=row, col=col, secondary_y=True#, showlegend=True   
                    )
            fig.add_trace(
                        go.Scatter(x=temp_data["day"][temp_data["treatment"] == trmt], 
                                y=temp_data["carbon_flux"][temp_data["treatment"] == trmt],
                                name=f"{trmt} - Carbon Imbalance",
                                marker=dict(
                                    color=color_map[trmt],  # Marker color
                                    size=7,       # Marker size
                                    symbol='circle' # Marker shape
                                ),
                                line=dict(
                                    color=color_map[trmt],  # Line color
                                    width=2,       # Line width
                                    dash='dot'
                                ), showlegend=carbLegend
                            ), row=row, col=col, secondary_y=True#, showlegend=True        
                    )
            
            ctrLegend = False
            ctrCarbLegend = False
        carbLegend = True
        ctrCarbLegend = True 
        trmtLegend = False 
    # for i in range(len(species)):
    #     fig.update_yaxes(title_text=f'{species[i]} <br> Metabolite Export', row=i+1, col=1, secondary_y=False)
    #     fig.update_yaxes(title_text='Carbon Imbalance', row=i+1, col=1, secondary_y=True)
    # for i in range(len(species)):
    #     for j in range(len(treatments)):
    #         if j == 0:
    #             # visible=False
    #             fig.update_yaxes(showticklabels=False, row=i + 1, col=j + 1, secondary_y=True)
    #             fig.update_yaxes(title_text=f'{species[i]} <br> Metabolite Export', row=i+1, col=j+1, secondary_y=False)
    #         if j == len(treatments)-1:
    #             fig.update_yaxes(showticklabels=False, row=i + 1, col=j + 1, secondary_y=False)
    #             fig.update_yaxes(title_text='Carbon Imbalance', row=i+1, col=j+1, secondary_y=True)
    #         else:
    #             fig.update_yaxes(showticklabels=False, row=i + 1, col=j + 1, secondary_y=True)
    #             fig.update_yaxes(showticklabels=False, row=i + 1, col=j + 1, secondary_y=False)
    fig.update_yaxes(range=[-105, 105], secondary_y=True)
    fig.update_yaxes(range=[-0.2, 7.5], secondary_y=False)
    fig.update_layout(height=500, width=380, font=dict(family='Times New Roman'), showlegend=True)
    plot_path = f"export_cBalance_{trmt}.png"
    pio.write_image(fig, plot_path, scale=5, width=380, height=500)
    fig.show()

# ---------------------------- Plots each value in a separate subplot
if plotEach:
    specs = []
    for spc in species:
        specs.append([{"secondary_y": True} for _ in ["pred_flux", "carbon_flux"]])
    fig = make_subplots(rows=len(species), cols=len(["pred_flux", "carbon_flux"]),
                        specs=specs, subplot_titles=species,
                        shared_xaxes=True,
                        shared_yaxes=True,
                        vertical_spacing=0.02,
                        horizontal_spacing=0.02)

    row = 0
    legend = True
    for val in ["pred_flux", "carbon_flux"]:
        row += 1
        col = 0 
        for spc in species:
            col += 1
            dash = 'solid' if val == "pred_flux" else 'dot'
            # Add traces 
            for trmt in ["Control"]+treatments: 
                temp_data = data[(data["treatment"]==trmt) & (data["spc"]==spc)]

                fig.add_trace(
                            go.Scatter(x=temp_data["day"], 
                                    y=temp_data[val],  
                                    marker=dict(
                                        color=color_map[trmt],  # Marker color
                                        size=7,       # Marker size
                                        symbol='circle' # Marker shape
                                    ),
                                    line=dict(
                                        color=color_map[trmt],  # Line color
                                        width=2,       # Line width
                                        dash = dash

                                    ), name = f"{trmt} - {val}", 
                                    showlegend=legend
                                ), row=row, col=col, secondary_y=False
                        ) 
            
            legend = False 

    fig.update_yaxes(title_text=f'Metabolite Export', row=1, col=1)
    fig.update_yaxes(title_text='Carbon Imbalance', row=2, col=1)
    fig.update_layout(height=450, width=900, font=dict(family='Times New Roman'))
    fig.show()

# ---------------------------- scatter plots by size

import json
from urllib.request import urlopen

def get_subsysClass(subsystems, verbose=False):
    pathways_class_dict = dict()
    PS_url  = "https://raw.githubusercontent.com/ModelSEED/PlantSEED/"
    PS_tag  = "8cf60046e4af68912f7a7d3eeff16880a07f56bd"
    PS_json = "/Data/PlantSEED_v3/PlantSEED_Roles.json"
    PS_json_data = json.load(urlopen(PS_url+PS_tag+PS_json))
    if verbose: print("LOADED ROLES")
    notProcessed = subsystems.copy()

    for item in PS_json_data:
        all_classes = item["classes"].keys()
        for cls in all_classes:
            for pathway in item["classes"][cls]:
                if pathway in notProcessed:
                    new_pathway = pathway.replace('_in_plants','')
                    new_pathway = new_pathway.replace('_',' ')
                    pathways_class_dict[new_pathway] = cls
                    pathways_class_dict[f"Z{cls}"] = cls
                    notProcessed.remove(pathway)
                if not notProcessed: break

    return pathways_class_dict

def apply_literal_eval(row, col='subsystems'):
    if isinstance(row[col], str):
        import ast
        if "Calvin-Benson-Bassham_cycle_in_plants" in row[col]:
            # print('---------------------> Here')
            return ["Calvin-Benson-Bassham_cycle_in_plants"]
        else:
            return ast.literal_eval(row[col])
    else:
        return []


sorghum_path = "/Users/selalaoui/Projects/QPSI_project/Enzyme_Abundance_all/src/util/Sorghum_leaf_res_flux.tsv"
sorghum_df = pa.read_csv(sorghum_path, sep = '\t')
sorghum_df['spc'] = 'Sorghum'
print("sorghum df ", sorghum_df.shape)

poplar_path = "/Users/selalaoui/Projects/QPSI_project/Enzyme_Abundance_all/src/util/Poplar_leaf_res_flux.tsv"
poplar_df = pa.read_csv(poplar_path, sep = '\t')
poplar_df['spc'] = 'Poplar'
print("poplar df ", poplar_df.shape)

subsys = pa.concat([sorghum_df, poplar_df])
print("all df ", subsys.shape)
subsys = subsys[subsys["treatment"].isin(['Control', 'FeLim'])]

subsys['subsystems'] = subsys.apply(lambda row: apply_literal_eval(row), axis=1)
subsys = subsys.explode('subsystems')
# print(subsys['subsystems'].unique())

sys_class = get_subsysClass(list(subsys['subsystems'].unique()))
sys_class_ls = sorted([v+'_'+k for k,v in sys_class.items()])
sys_class_dict = {k:v+'_'+k for k,v in sys_class.items()}

remove_sub = ['Photorespiration (oxidative C2 cycle)', 'Plastoquinone biosynthesis', 'Starch biosynthesis', 'Starch degradation', 
'Galactose degradation', 'Folate biosynthesis', 'Histidine Biosynthesis', 
'GDP-sugars biosynthesis and interconversions', 'Rubisco shunt', 'Fatty acid biosynthesis (mitochondrial)',
'Acetyl-CoA carboxylase complexes', 'Tyrosine and phenylalanine metabolism', 'TCA cycle',
'Arginine metabolism and urea cycle', 'Alanine, serine, glycine metabolism', 
'Branched-chain amino acid metabolism']
#, 'Purine de novo biosynthesis', 'UDP-glucose and UDP-galactose biosynthesis'

subsys['subsystems'] = subsys['subsystems'].str.replace('_in_plants','')
subsys['subsystems'] = subsys['subsystems'].str.replace('_',' ')
subsys = subsys[~subsys['subsystems'].isin(remove_sub)]


subsys= subsys.groupby(['spc', 'day', 'treatment', 'subsystems'])['pred_flux'].sum().reset_index(name='flux_sums')
subsys['class'] = subsys['subsystems'].map(sys_class)
subsys = subsys[subsys['class'].isin(["Central Carbon", "Amino acids", "Fatty acids"])]

## names include class
subsys.replace({"subsystems": sys_class_dict},inplace=True)



print(subsys.columns)
# Index(['spc', 'day', 'treatment', 'subsystems', 'flux_sums', 'class']

# Get the control values for each day
control_pred = subsys[subsys['treatment'] == 'Control'][['spc', 'day', 'subsystems', 'class', 'flux_sums']]
felim_pred = subsys[subsys['treatment'] == 'FeLim'][['spc', 'day', 'subsystems', 'class', 'flux_sums']]
# Merge the control predictions back to the original DataFrame
control_pred = control_pred.merge(felim_pred, how='left', on=['day', 'spc', 'subsystems', 'class'], suffixes=('', '_felim'))
felim_pred = felim_pred.merge(control_pred[['spc', 'day', 'subsystems', 'class', 'flux_sums']], 
    how='left', on=['day', 'spc', 'subsystems', 'class'], 
    suffixes=('', '_control'))

print(control_pred.head())
print(felim_pred.head())

control_pred['size'] = control_pred['flux_sums'] - control_pred['flux_sums_felim'] 
control_pred['size'] = (control_pred['size'] + abs(control_pred['size']))/2
control_pred.drop(columns=['flux_sums_felim'], inplace=True)
control_pred['treatment'] = 'Control'
print(control_pred.head())

felim_pred['size'] = felim_pred['flux_sums'] - felim_pred['flux_sums_control'] 
felim_pred['size'] = (felim_pred['size'] + abs(felim_pred['size']))/2
felim_pred.drop(columns=['flux_sums_control'], inplace=True)
felim_pred['treatment'] = 'FeLim'
print(felim_pred.head())

subsys = pa.concat([control_pred, felim_pred])
print(subsys.head())
# print(abc)
# print(subsys.columns)
# print(control_pred.head())
# subsys = subsys.merge(control_pred, how='left', on=['day', 'spc', 'subsystems'], suffixes=('', '_control'))
# print(subsys.tail())
# # Add the size column based on the condition
# # subsys['size'] = ((subsys['flux_sums'] > subsys['flux_sums_control']) 
# #     & (subsys['treatment'].isin(['FeLim']))).astype(int)
# subsys['size'] = (abs(subsys['flux_sums'] - subsys['flux_sums_control']))

# print(subsys.tail())
# # Drop the extra control column
# subsys.drop(columns=['flux_sums_control'], inplace=True)
# print(subsys['subsystems'].unique())
print(subsys.tail())
# print(subsys)

# subsys['ds'] = subsys['day']+'_'+subsys['spc']
# dss = ['02d_Poplar', '02d_Sorghum', '03d ', '04d_Poplar', '04d_Sorghum', '04d', '07d_Poplar', '07d_Sorghum', '07d', '14d_Poplar', '14d_Sorghum', '14d', '21d_Poplar', '21d_Sorghum']

subsys['ds'] = subsys['day']+'_'+subsys['treatment']
dss = ['02d_Control', '02d_FeLim', '03d ', '04d_Control', '04d_FeLim', '04d', '07d_Control', '07d_FeLim', '07d', '14d_Control', '14d_FeLim', '14d', '21d_Control', '21d_FeLim']


sys_class_ls = set(subsys[(subsys['flux_sums']>30)]['subsystems'].tolist())
print(sys_class_ls)

# sys_class_ls.update(["Amino acids_ZAmino",
    # "Fatty acids_ZFatty",
#     "Energy_ZEnergy",
#     "Cell wall_ZCell",
#     "Carbohydrates_ZCarbohydrates",
#     "Central Carbon_ZCentral",
#     "Nucleic acids_ZNucleic",
#     "Cofactors_ZCofactors",
#     "Lipids_ZLipids"])
sys_class_ls.update(["Amino acids_ZAmino",
    "Fatty acids_ZFatty",
    "Central Carbon_ZCentral"])

sys_class_ls = sorted(list(sys_class_ls))
title = "some title"
ht, wt = 400, 800
y_labels = {'Central Carbon_Pentose phosphate pathway': "Pentose phosphate pathway", 
'Fatty acids_Unsaturated Fatty Acid biosynthesis': "Unsaturated FA biosynthesis", 
'Amino acids_Glutamine, glutamate, aspartate, asparagine metabolism': "Glu, Gln, Asp, Asn metabolism", 
'Central Carbon_Calvin-Benson-Bassham cycle': "Calvin-Benson-Bassham Cycle", 
'Fatty acids_Fatty acid biosynthesis (plastidial)': "FA biosynthesis",  
'Amino acids_Lysine and threonine metabolism': "Lys, and Thr metabolism", 
'Central Carbon_Glycolysis and Gluconeogenesis': "Glycolysis and Gluconeogenesis"}

# color_map = {'Poplar': 'rgba(85, 55, 57, 1)', 'Sorghum': 'rgba(116, 142, 84, 1)'}
color_map = {'Control': 'rgba(187, 68, 48, 1)', 'FeLim': 'rgba(8, 76, 97, 1)'}
# color_map = {'Control': 'rgba(187, 68, 48, 1)', "FeLim": 'rgba(34, 49, 39, 1)'}
fig = px.scatter(subsys[(subsys['flux_sums']>30)], 
    x='ds', y="subsystems", 
    size="flux_sums", title=title, 
    facet_col='spc', 
    category_orders={"ds": dss, 'subsystems':sys_class_ls}, 
    color='treatment', symbol='treatment', 
    symbol_sequence=['circle', 'circle'],
    color_discrete_map=color_map, 
    facet_col_spacing=0.01,
    # labels = y_labels, 
    height=ht, width=wt) # , facet_row='tissue'

spc_df = subsys[(subsys['flux_sums']>30) & (subsys['spc']=='Poplar') & (subsys['size']>5)]
fig = fig.add_trace(go.Scatter(x=spc_df['ds'],
                            y=spc_df['subsystems'],
                            # x=sys_trmt_1[trmt],
                            # y=sys_trmt_1[ctrl],
                            mode='markers',

                            marker=dict(symbol='star-open', size=6, color='black'),
                            showlegend=False),
                            row=1, col=1)

spc_df = subsys[(subsys['flux_sums']>30) & (subsys['spc']=='Sorghum') & (subsys['size']>5)]
fig = fig.add_trace(go.Scatter(x=spc_df['ds'],
                            y=spc_df['subsystems'],
                            # x=sys_trmt_1[trmt],
                            # y=sys_trmt_1[ctrl],
                            mode='markers',

                            marker=dict(symbol='star-open', size=6, color='black'),
                            showlegend=False),
                            row=1, col=2)



# Extract tick values and corresponding new labels
tickvals = list(y_labels.keys())
ticktext = list(y_labels.values())
# Update y-axis tick labels
fig.update_yaxes(tickvals=tickvals, ticktext=ticktext)

x_labels = {'02d_Control':"02d", '02d_FeLim':"", '03d ':"", 
    '04d_Control':"04d", '04d_FeLim':"", '04d':"", 
    '07d_Control':"07d", '07d_FeLim':"", '07d':"", 
    '14d_Control':"14d", '14d_FeLim':"", '14d':"", 
    '21d_Control':"21d", '21d_FeLim':""}
# Extract tick values and corresponding new labels
tickvals = list(x_labels.keys())
ticktext = list(x_labels.values())
fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=0)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
# fig.update_layout(height=450, width=900, font=dict(family='Times New Roman'))
fig.update_layout(
            font=dict(
                family="Arial",
                size=12
            ), showlegend=True
        )
# fig.update_traces(marker=dict(line=dict(width=subsys['size'] * 2)))
# plot_path = "SorghumPoplar_scatter_plots_allSubsys.png"
# pio.write_image(fig, plot_path, scale=5, width=wt, height=ht)
# fig.show()
