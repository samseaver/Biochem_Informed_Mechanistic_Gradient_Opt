import pandas as pa
import os
import numpy as np
import json

import plotly.express as px
import seaborn as sns
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.io as pio
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
pio.templates.default = "plotly_white" #"none"


from plotly.subplots import make_subplots
from Calculate_Carbon_Flux import Calculate_Carbon_Flux
from cobra.io import read_sbml_model


from dash import Dash, dcc, html, Input, Output
import dash_cytoscape as cyto

avogadro = 6.02214076e+23
species = ['Poplar', 'Sorghum']
genotype = 'Leaf'
time_stamps = ['02d', '04d', '07d', '14d', '21d']
sv = 'sv15/'
model_path = "/Users/selalaoui/Projects/QPSI_project/Enzyme_Abundance_all/data/metabolic_models/plastidial_models/ortho_jun20_models/sbicolor_3.1.1_plastid_Thylakoid_Reconstruction_ComplexFix_070224_noADP_duplicated_noP.xml"
co_model = read_sbml_model(model_path)

ccf_obj = Calculate_Carbon_Flux()
treatments = ["Control", "FeEX", "FeLim", "ZnEx", "ZnLim"]
msrs = ['Pred', 'FBA', 'Vbf', 'RES']
control_id = "Control"
projCols = ['tissue', 'treatment', 'time_stamp']
tissue = 'Leaf'

# treatments = ["control", "cold"]
# projCols = ['Treatment', 'Timestamp']
# spc = 'athaliana'
# time_stamp = 'ZT9'
# tissue = ''
# genotype = 'C24'

more = "_noADP_noP"
# prediction_file = f"/Users/selalaoui/Projects/AMN/omic_amn_mm/Result/{sv}{spc}_{genotype}_{time_stamp}_complexFix{more}_V_rxn.tsv"
# kapp_vbf_path = f"/Users/selalaoui/Projects/AMN/omic_amn_mm/Dataset_input/{spc}_complexFix_{genotype}_{time_stamp}_noADP_Vbf_kapp_maxCtrl_mixedRelab.csv"
# scores_path = f"/Users/selalaoui/Projects/QPSI_project/Enzyme_Abundance_all/integration_results/reaction_scores_binding_Jul2/plastidial_model/{spc}_objective_abundance_Control.tsv"
# sig_reactions = f"/Users/selalaoui/Projects/QPSI_project/Enzyme_Abundance_all/src/util/sig_reactions_{spc.lower()}_{genotype.lower()}.tsv"
s_matrix_path = f"/Users/selalaoui/Projects/AMN/omic_amn_mm/Result/s_matrix/s_matrix_plastid_noP_T_new.csv"
saveTo = f"/Users/selalaoui/Projects/AMN/omic_amn_mm/Result/{sv}"
chlorophyll_enzymes_file = "/Users/selalaoui/Projects/AMN/omic_amn_mm/Result/sv15/cloro_rxns.tsv"

media = ["cpd00001", "cpd00003", "cpd00004", "cpd00005",
	"cpd00006", "cpd00007", "cpd00009", "cpd00011", "cpd00012", 
	"cpd00013", "cpd00067", "cpd11632", "cpd00002", "cpd00008"]

pathway_dict = {"Photosynthesis":
	["Chlorophyll_Biosynthesiss",
		"Heme_and_Siroheme_biosynthesiss",
		"Phylloquinone_biosynthesiss"],
"Fatty acid metabolism": 
	["Fatty_acid_biosynthesis_(plastidial)",
	"Unsaturated_Fatty_Acid_biosynthesis",
	"Acetyl-CoA_biosynthesis",
	"Acetyl-CoA_carboxylase_complexes"],
"One-carbon metabolism":
	["Folate-mediated_one-carbon_metabolism",
		"Folate_biosynthesis"]}
classes = ['Photosynthesis', 'Fatty acid metabolism', 'One-carbon metabolism']
pathways = {"Chlorophyll_Biosynthesis": "Photosynthesis",
		"Heme_and_Siroheme_biosynthesis": "Photosynthesis",
		"Phylloquinone_biosynthesis": "Photosynthesis",
	"Fatty_acid_biosynthesis_(plastidial)": "Fatty acid metabolism",
	"Unsaturated_Fatty_Acid_biosynthesis": "Fatty acid metabolism",
	"Acetyl-CoA_biosynthesis": "Fatty acid metabolism",
	"Acetyl-CoA_carboxylase_complexes": "Fatty acid metabolism", 
	"Folate-mediated_one-carbon_metabolism": "One-carbon metabolism",
		"Folate_biosynthesis": "One-carbon metabolism"}
subsystms = ["Chlorophyll_Biosynthesis",
		"Heme_and_Siroheme_biosynthesis",
		"Phylloquinone_biosynthesis",
	"Fatty_acid_biosynthesis_(plastidial)",
	"Unsaturated_Fatty_Acid_biosynthesis",
	"Acetyl-CoA_biosynthesis",
	"Acetyl-CoA_carboxylase_complexes", 
	"Folate-mediated_one-carbon_metabolism",
		"Folate_biosynthesis"]

subsystms_sort = ["Chlorophyll Biosynthesis",
		"Heme and Siroheme biosynthesis",
		"Phylloquinone biosynthesis",
	"Fatty acid biosynthesis (plastidial)",
	"Unsaturated Fatty Acid biosynthesis",
	"Acetyl-CoA biosynthesis",
	"Acetyl-CoA carboxylase complexes", 
	"Folate-mediated one-carbon metabolism",
		"Folate biosynthesis"]
# cpd00001_d0 H2O
# cpd00002_d0 ATP 
# cpd00008_d0 ADP
# cpd00003_d0 NAD
# cpd00004_d0 NADH
# cpd00005_d0 NADPH
# cpd00006_d0 NADP
# cpd00007_d0 O2
# cpd00009_d0 Phosphate
# cpd00011_d0 CO2
# cpd00012_d0 PPi
# cpd00013_c0 NH3
# cpd00067_d0 H+
# cpd11632_d0 hn

def get_met_rxns(row):
	metabolite = co_model.metabolites.get_by_id(row["met_ID"])

	# Print the metabolite's name and ID
	# print(f'Metabolite ID: {metabolite.id}')
	# print(f'Metabolite Name: {metabolite.name}')

	# Get the reactions involving the metabolite
	involved_reactions = [rxn.id for rxn in metabolite.reactions]
	# involved_reactions = ",".join([rxn.id for rxn in metabolite.reactions])

	return involved_reactions

def get_rxn_ID(row):
	if any(y in row['rxn_ID'] for y in ['_f', '_r', '_i', '_o']):
		# print(row['rxn_ID'].rsplit("_", 1)[0])
		id_only = row['rxn_ID'].rsplit("_", 1)[0]
	else:
		id_only = row['rxn_ID']

	return id_only

def consolidate(grp_obj, value_cols=[]):
	# grp_obj[value_cols] = 2*grp_obj[value_cols] - grp_obj[value_cols].sum() #+grp_obj
	print(grp_obj)
	return grp_obj

def apply_literal_eval(row, col='subsystems'):
	if isinstance(row[col], str):
		import ast
		# if "Calvin-Benson-Bassham_cycles" in row[col]:
		# 	# print('---------------------> Here')
		# 	return ["Calvin-Benson-Bassham_cycles"]
		# else:
		# 	return ast.literal_eval(row[col])
		return ast.literal_eval(row[col])
	else:
		return []

def find_reactions(metabolites, trmt_pred_flux, trmt):
	# read the model 
	# print(trmt_pred_flux.head())
	# value = trmt_pred_flux.loc["EX_cpd00001_e0_i"]  # Using .iloc
	# print("**", value, "**")
	# print(abc)
	co_model = read_sbml_model(model_path)
	# find reactions 
	cyto_rxns = []
	for met in metabolites:
		print(met, "--"*7)
		metabolite = co_model.metabolites.get_by_id(met)

		# Print the metabolite's name and ID
		print(f'Metabolite ID: {metabolite.id}')
		print(f'Metabolite Name: {metabolite.name}')

		# Get the reactions involving the metabolite
		involved_reactions = [rxn.id for rxn in metabolite.reactions]
		cyto_rxns = cyto_rxns + involved_reactions
	return cyto_rxns
	# app = cyto_app(metabolites, cyto_rxns, trmt_pred_flux, trmt)

def find_reactions_df(metabolites, trmt_pred_flux, trmt):
	# read the model 
	# print(trmt_pred_flux.head())
	# value = trmt_pred_flux.loc["EX_cpd00001_e0_i"]  # Using .iloc
	# print("**", value, "**")
	# print(abc)

	# co_model = read_sbml_model(model_path)

	# find reactions 
	cyto_rxns = []
	for met in metabolites:
		print(met, "--"*7)
		metabolite = co_model.metabolites.get_by_id(met)

		# Print the metabolite's name and ID
		print(f'Metabolite ID: {metabolite.id}')
		print(f'Metabolite Name: {metabolite.name}')

		# Get the reactions involving the metabolite
		involved_reactions = [rxn.id for rxn in metabolite.reactions]
		cyto_rxns = cyto_rxns + involved_reactions
	return cyto_rxns
	# app = cyto_app(metabolites, cyto_rxns, trmt_pred_flux, trmt)


# @app.callback(Output('cytos', 'elements'),
#					  prevent_initial_call=False)
def update_elements(metabolites, cyto_rxns, netFlux_trmt_df, trmt_pred_flux):
	model = read_sbml_model(model_path)

	m_color = "blue"
	rxn_color = "green"
	media_color = "yellow"
	imlMet_color = "red"
	dash_element_list = list()
	i = 0

	missing = ""
	for e_id in cyto_rxns:
		# e_id += '_d0'
		if 'rxn' in e_id and model.reactions.has_id(e_id):
			rxn = model.reactions.get_by_id(e_id)
			# add node for the reaction
			flux = trmt_pred_flux.loc[e_id].values[0]
			# print(flux)
			print(e_id)
			other_f = 0
			if '_f' in e_id:
				other_f = trmt_pred_flux.loc[e_id.replace('_f', '_r')].values[0]  
			else:
				other_f = trmt_pred_flux.loc[e_id.replace('_r', '_f')].values[0]  
			if any(sfx in e_id for sfx in ['_f', '_r']) and (flux != 0) and (other_f != 0): 
				print(e_id, flux, other_f)
				# if '_r' in e_id:
				# 	continue
				print(abc)
			elif any(sfx in e_id for sfx in ['_f', '_r']) and (flux == 0) and (other_f == 0): 
				print(e_id, flux, other_f)
				if '_r' in e_id:
					continue
				# print(abc)

			elif any(sfx in e_id for sfx in ['_f', '_r']) and (flux == 0) and (other_f != 0):
				continue
			else:
				label =  f"{e_id}={flux:.2f}"
				color = 'gray' if flux == 0.0 else rxn_color
				dash_element_list.append({'data': {'id': e_id, 'label':label}, 'classes':color})

				clss = 'one_dir' #'two_dir' if rxn.reversibility else 'one_dir'
				for rct in rxn.reactants:
					# create an edge for each reactant
					r_id = rct.id
					net_flux = 3
					if r_id.split("_")[0] not in media:
						# continue 

					
						color = m_color
						# if media element, create the node
						if r_id in media:
							r_id = r_id+"_"+str(i)
							color = media_color
							i+=1
						# print(r_id, metabolites)
						if r_id in metabolites:

							net_flux = netFlux_trmt_df.loc[r_id].values[0]
							net_flux = np.floor(np.abs(net_flux))
							if net_flux < 5: 
								color = "red0"
							elif net_flux < 10:
								color = "red1"
							elif net_flux < 15: 
								color = "red2"
							else: 
								color = "red3"
						# color = media_color		
							# color = "red"
							print(r_id, " reac ", net_flux, " ", color)
						# print(color, " ")
						# dash_element_list.append({'data': {'id': r_id, 'label':rct.name, 'flux': 10}, 'classes': color})
						dash_element_list.append({'data': {'id': r_id, 'label':rct.name}, 'classes': color})
						# create the edge
						dash_element_list.append({'data': {'source': r_id, 'target': e_id}, 'classes':clss})

				for prod in rxn.products:
					# create an edge for each reactant
					r_id = prod.id
					net_flux = 3
					
					if r_id.split("_")[0] not in media:
						# continue 
						
						color = m_color
						# if media element, create the node
						if r_id in media:
							r_id = r_id+"_"+str(i)
							color = media_color
							i+=1
						if r_id in metabolites:
							net_flux = netFlux_trmt_df.loc[r_id].values[0]
							net_flux = np.floor(np.abs(net_flux))
							if net_flux < 5: 
								color = "red0"
							elif net_flux < 10:
								color = "red1"
							elif net_flux < 15: 
								color = "red2"
							else: 
								color = "red3"
							# net_flux = 10
							# color = "red"
							print(r_id, " prod ", net_flux, " ", color)
						# print(color, " ")							
						# print(r_id, " ", net_flux)
						# dash_element_list.append({'data': {'id': r_id, 'label':prod.name, 'flux': net_flux}, 'classes': color})
						dash_element_list.append({'data': {'id': r_id, 'label':prod.name}, 'classes': color})
						# create the edge
						dash_element_list.append({'data': {'source': e_id, 'target': r_id}, 'classes':clss})
		else:
			missing += e_id + ' '
	print("missing ", missing)
	return dash_element_list

def plot_chlorophill_lines(predFlux_df):
	# print(predFlux_df.head())
	# print(predFlux_df)
	# 'rxn_ID', 'treatment', 'predFlux', 'spc', 'time_stamp'
	# mask = predFlux_df.groupby(['rxn_ID', 'treatment', 'spc'])['predFlux'].transform('any') != 0

	# # Filter the DataFrame based on the mask
	# predFlux_df = predFlux_df[mask]


	subsys = pa.read_csv("/Users/selalaoui/Projects/AMN/omic_amn_mm/Result/Sorghum_Leaf_14d_complexFix_noADP_noP_V_rxn_fba_Vbf_RES_wide.tsv", sep='\t')
	subsys['subsystems'] = subsys.apply(lambda row: apply_literal_eval(row), axis=1)
	subsys = subsys.explode('subsystems')
	subsys['subsystems'] = subsys['subsystems'].str.replace('_in_plants_and_prokaryotes','')
	subsys['subsystems'] = subsys['subsystems'].str.replace('_in_plants','')

	predFlux_df = predFlux_df.merge(subsys[['rxn_ID', 'subsystems']], on=["rxn_ID"], how='left')


	# If you want to reset the index after filtering
	predFlux_df.reset_index(drop=True, inplace=True)
	print(predFlux_df)
	enzymes = pa.read_csv(chlorophyll_enzymes_file, sep='\t')
	print(enzymes)
	# print(abc)
	all_df = predFlux_df[predFlux_df['rxn_ID'].isin(enzymes["rxn_ID"].unique())]
	all_df = all_df.merge(enzymes, on=["rxn_ID"], how='inner')
	
	print(all_df.head())

	days = ['02d', '04d', '07d', '14d', '21d']
	# fig = px.parallel_coordinates(all_df, color="treatment",
	# 	dimensions=['time_stamp'],
	# 	# labels={"species_id": "Species",
	# 	# "sepal_width": "Sepal Width", "sepal_length": "Sepal Length",
	# 	# "petal_width": "Petal Width", "petal_length": "Petal Length", },
	# 	color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)
	# fig.show()
	# print(abc)

	# all_df.to_csv("groups_flux_sum.tsv", sep='\t', index=False)

	# wt = 500 , -0.7 / wt = 550, -0.55
	ht, wt, x_annot, fnt_size = 300, 450, -0.3, 11
	# ht, wt, x_annot, fnt_size = 600, 800, -0.55, 20
	# -------------------------- Chlorophill biosynthesis per reaction all treatments 
	mask = all_df.groupby(['rxn_ID', 'treatment', 'spc'])['predFlux'].transform('any') != 0
	# # Filter the DataFrame based on the mask

	all_df = all_df[mask]
	ec_nums = [ec for ec in enzymes['ec_number'].unique() 
					if ec in all_df['ec_number'].unique() 
					and ec in all_df['ec_number'].unique()]
	enzyme_list = [ec for ec in enzymes['enzyme'].unique() 
					if ec in all_df['enzyme'].unique()
					and ec in all_df['enzyme'].unique()]

	for trmt in ['FeLim']:#treatments:
		fig = px.line(
			all_df[all_df["treatment"].isin([trmt, 'Control'])],
			x="time_stamp",
			y="predFlux",
			color="rxn_ID",
			line_dash = 'treatment',
			title=f"Chlorophill biosynthesis reactions {trmt}",
			# symbol= 'subsystems',
			# color_continuous_scale='icefire',
			facet_row="ec_number",
			# facet_row="enzyme",
			facet_col="spc",
			labels={"predFlux": ""},
			height=ht, width=wt,
			facet_col_spacing=0.01,
			category_orders={"time_stamp": days, "enzyme":enzyme_list, "ec_number":ec_nums},
			facet_row_spacing=0.01
		)
		# for i, annotation in enumerate(fig.layout.annotations):
		# 	if annotation["text"].startswith("facet="):
		# 		# Rotate the annotation text
		# 		annotation["textangle"] = -90
		# 		# Adjust the position of the annotation
		# 		annotation["x"] = -0.1
		# 		annotation["y"] = annotation["y"] - 0.1

		# hide and lock down axes
		fig.update_xaxes(fixedrange=True, showgrid=False, zeroline=False, 
		showline=False, mirror=False, linewidth=.5, linecolor='lightgray', visible=False)
		fig.update_yaxes(fixedrange=True, showgrid=False, zeroline=False, 
		showline=False, mirror=False, linewidth=.5, linecolor='lightgray', visible=False) #visible=False, 

		# remove facet/subplot labels
		# fig.update_layout(annotations=[], overwrite=True)

		# strip down the rest of the plot
		fig.update_traces(line={'width': 1},)
		fig.update_layout(
			showlegend=True,
			# font_family="Times New Roman", #Courier New, Arial
			font=dict(
				family="Arial",#Courier New, Arial
				size=fnt_size,
			),
			plot_bgcolor="rgba(245, 243, 244, 1)",
			# annotations=[
			# 	 dict(
			# 		  # text="sex=Male",
			# 		  # x=row.x,
			# 		  # y=row.y,
			# 		  textangle=0,
			# 		  # xanchor="center",
			# 		  # yanchor="top",
			# 		  # showarrow=False,
			# 	 )
			# ],
			yaxis={"side": "right"},
			yaxis_title=None,
			margin=dict(l=100, r=250),

		)
		for annotation in fig.layout.annotations:
			annotation.text = annotation.text.split("=")[1]
			if 'EC' in annotation.text:
				annotation.x=x_annot, 
				annotation.xref="paper" 
				annotation.textangle=0
		# fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], x=x_annot, xref="paper", textangle=0) if 'ec' in a.text)
		# fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
			# Add the text annotation

		fnt_lng = 7
		# fig.add_annotation(
		# 	text="Control",
		# 	xref="paper", yref="paper",
		# 	x=0.89, y=1,
		# 	showarrow=False,
		# 	font=dict(size=fnt_lng),
		# 	align="right"
		# )
		# # Add the text annotation
		# fig.add_annotation(
		# 	text="FeLim",
		# 	xref="paper", yref="paper",
		# 	x=0.89, y=0.93,
		# 	showarrow=False,
		# 	font=dict(size=fnt_lng),
		# 	align="right"
		# )
		# # Add the line
		# fig.add_shape(
		# 	type="line",
		# 	xref="paper", yref="paper",
		# 	x0=0.9, y0=0.95,
		# 	x1=0.99, y1=0.95,
		# 	line=dict(color="black", width=1)
		# )
		# # Add the line
		# fig.add_shape(
		# 	type="line",
		# 	xref="paper", yref="paper",
		# 	x0=0.9, y0=0.90,
		# 	x1=0.99, y1=0.90,
		# 	line=dict(color="black", width=1, dash='2, 2, 2, 2')
		# )
		# =----------- in the blank area
		# fig.add_annotation(
		# 	text="Treatment",
		# 	xref="paper", yref="paper",
		# 	x=0, y=0.77,
		# 	showarrow=False,
		# 	font=dict(size=fnt_lng, color='black'),
		# 	align="right"
		# )
		fig.add_annotation(
			text="Control",
			xref="paper", yref="paper",
			x=0, y=0.7,
			showarrow=False,
			font=dict(size=fnt_lng, family='Arial'), #, color='black'
			align="right"
		)
		# Add the text annotation
		fig.add_annotation(
			text="FeLim",
			xref="paper", yref="paper",
			x=0, y=0.60,
			showarrow=False,
			font=dict(size=fnt_lng, family='Arial'),
			align="right"
		)
		# Add the line
		fig.add_shape(
			type="line",
			xref="paper", yref="paper",
			x0=0.25, y0=0.65,
			x1=0.4, y1=0.65,
			line=dict(color="black", width=1)
		)
		# Add the line
		fig.add_shape(
			type="line",
			xref="paper", yref="paper",
			x0=0.25, y0=0.58,
			x1=0.4, y1=0.58,
			line=dict(color="black", width=1, dash='2, 2, 2, 2')
		)
		fig.show()
		plot_path = f"{trmt}_chlorophyll_biosynthesis_reactions.png"
		#chlorophyll biosynthesis
		pio.write_image(fig, plot_path, scale=6, width=wt, height=ht)
		# print(abc)
	
	# df['abs_C_netFlux'] = np.abs(df['C_netFlux'])

	all_df_sums = predFlux_df[predFlux_df["subsystems"].isin(pathways.keys())]
	all_df_sums = all_df_sums[~all_df_sums['rxn_ID'].isin(['rxn04153_d0_r', 'rxn04153_d0_f', 'rxn04154_d0_f', 'rxn04154_d0_r'])]
	all_df_sums['class'] = all_df_sums['subsystems'].map(pathways)
	df_grouped = all_df_sums.groupby(['subsystems', "spc", "treatment", "time_stamp"]).agg({
		'predFlux': 'sum',
		'rxn_ID':','.join,
		# 'enzyme': ','.join,
		# 'ec_number': ','.join,
	}).reset_index()

	df = pa.merge(df_grouped, all_df_sums[["subsystems", "class"]], on='subsystems', how='left')
	# df.drop_duplicates(inplace=True)

	print(df[['subsystems', 'class', "spc", "treatment", "time_stamp", 'predFlux', 'rxn_ID']].head())
	days = ['02d', '04d', '07d', '14d', '21d']
	# df.to_csv("groups_flux_sum.tsv", sep='\t', index=False)

	# -------------------------- cumulative predFlux per pathway all treatments 
	ht, wt = 700, 1000
	for trmt in treatments:
		fig = px.line(
			df[df["treatment"] == trmt],
			# x=trmt,
			# y=control_name,
			x="time_stamp",
			y="predFlux",
			color="subsystems",
			title=f"Sum of predFlux per pathway - {trmt}",
			# line_dash = 'treatment',
			# symbol= 'subsystems',
			# color_continuous_scale='icefire',
			# range_color=[-1*mx, mx],
			facet_row="class",
			facet_col="spc",
			# facet_row=trmt_col,
			# hover_data=["rxn_ID"],
			# labels={"rxn_score_I_dist_"+trmt: "Flux Dist"},
			# category_orders=category_orders
			height=ht, width=wt,
			# , facet_row_spacing=0.08
			category_orders={"time_stamp": days, 'class': classes},
			facet_row_spacing=0.03
		)
		# fig.show()

	# -------------------------- cumulative predFlux Control and FeLim
	ht, wt = 500, 1500
	fig = px.line(
		df[df["treatment"].isin(['Control', 'FeLim'])],
		# x=trmt,
		# y=control_name,
		x="time_stamp",
		y="predFlux",
		color="subsystems",
		title=f"Sum of predFlux per pathway - Control and FeLim",
		line_dash = 'treatment',
		# line_width=1,
		# symbol= 'subsystems',
		# color_continuous_scale='icefire',
		# range_color=[-1*mx, mx],
		facet_row="spc",
		facet_col="class",
		# facet_row=trmt_col,
		# hover_data=["rxn_ID"],
		# labels={"rxn_score_I_dist_"+trmt: "Flux Dist"},
		# category_orders=category_orders
		height=ht, width=wt,
		# , facet_row_spacing=0.08
		category_orders={"time_stamp": days, 'class': classes},
		facet_row_spacing=0.03
	)
	# san serif
	fig.update_layout(font=dict(family='Arial'))
	# fig.show()
	# print(abc)
	return 1

def elements_multiple(cyto_rxns, spc, netFlux_df, predFlux_df, treatments=['FeLim']):
	graph_data_list = []
	percentile = 95
	netFlux_threshold = np.abs(netFlux_df['netFlux'])\
							.describe([percentile/100])[str(percentile)+'%']
	print("Max: ", np.abs(netFlux_df['netFlux']).max())
	print(" ", percentile,": ", netFlux_threshold)

	# find_reactions_df(netFlux_df, trmt_pred_flux, trmt)
	netFlux_df['rxn_ID'] = netFlux_df.apply(lambda row: get_met_rxns(row), axis=1)
	netFlux_df = netFlux_df.explode('rxn_ID')
	netFlux_df = netFlux_df[["met_ID", "netFlux", "C_netFlux", "spc", "treatment", "time_stamp", "rxn_ID"]]
	predFlux_df = predFlux_df[["rxn_ID", "treatment", "predFlux", "spc", "time_stamp"]]
	print(netFlux_df.shape)
	print(predFlux_df.shape)
	print(netFlux_df.head())
	print(predFlux_df.head())


	
	df = netFlux_df.merge(predFlux_df, on=["spc", "treatment", "time_stamp", "rxn_ID"], how='left')
	print(df.head())

	subsys = pa.read_csv("/Users/selalaoui/Projects/AMN/omic_amn_mm/Result/Sorghum_Leaf_14d_complexFix_noADP_noP_V_rxn_fba_Vbf_RES_wide.tsv", sep='\t')
	subsys['subsystems'] = subsys.apply(lambda row: apply_literal_eval(row), axis=1)
	subsys = subsys.explode('subsystems')
	subsys['subsystems'] = subsys['subsystems'].str.replace('_in_plants_and_prokaryotes','')
	subsys['subsystems'] = subsys['subsystems'].str.replace('_in_plants','')

	df = df.merge(subsys[['rxn_ID', 'subsystems']], on=["rxn_ID"], how='left')
	print(df.head())
	df[(np.abs(df['netFlux'])>=netFlux_threshold)].to_csv("metabolites_reactions_subsystem.tsv", sep='\t', index=False)
	print(df.head())

	
	df = df[(np.abs(df['netFlux'])>=netFlux_threshold)]
	# df[df['predFlux']==0]['netFlux'] = df[df['predFlux']==0]['netFlux']*0
	df.loc[df['predFlux'] == 0, 'netFlux'] = 0
	df.loc[df['predFlux'] == 0, 'C_netFlux'] = 0
	# df[df['predFlux']==0]['C_netFlux'] = df[df['predFlux']==0]['C_netFlux']*0

	# print(df.shape)
	# df = df.dropna()
	# print(df.shape)
	# df['subsystems'] = df.apply(lambda row: apply_literal_eval(row), axis=1)
	# df = df.explode('subsystems')
	# df['subsystems'] = df['subsystems'].str.replace('s_and_prokaryotes','')
	# df['subsystems'] = df['subsystems'].str.replace('s','')
	# df = df.explode('subsystems')
	# print(abc)
	print(df.head())
	print(df.shape)
	df = df[df["subsystems"].isin(pathways.keys())]
	df = df[df["subsystems"].isin(pathways.keys())]
	df = df[~df['rxn_ID'].isin(['rxn04153_d0_r', 'rxn04153_d0_f', 'rxn04154_d0_f', 'rxn04154_d0_r'])]
	df['class'] = df['subsystems'].map(pathways)
	print(df.shape)
	print(df.head())
	# print(df['subsystems'].unique())

	
	df.drop(columns=['rxn_ID', 'predFlux'], inplace=True)
	df.drop_duplicates(inplace=True)
	print(df.shape)

	df['abs_C_netFlux'] = np.abs(df['C_netFlux'])
	df_grouped = df.groupby(['subsystems', "spc", "treatment", "time_stamp"]).agg({
		'netFlux': 'mean',
		'C_netFlux': 'sum', 
		'abs_C_netFlux': 'sum',
		# "rxn_ID":','.join, 
		"met_ID":','.join 
	}).reset_index()

	df_grouped_totals = df.groupby(["spc", "treatment", "time_stamp"]).agg({
		'netFlux': 'mean',
		'C_netFlux': 'sum', 
		# 'predFlux': 'sum',
		'abs_C_netFlux': 'sum'
	}).reset_index()

	df_grouped_class = df.groupby(["class", "spc", "treatment", "time_stamp"]).agg({
		'netFlux': 'mean',
		'C_netFlux': 'sum', 
		# 'predFlux': 'sum',
		'abs_C_netFlux': 'sum'
	}).reset_index()

	print(df_grouped.head())

	df = pa.merge(df_grouped, df[["subsystems", "class"]], on='subsystems', how='left')
	df.drop_duplicates(inplace=True)
	print(df[['subsystems', "spc", "treatment", "time_stamp", 'C_netFlux', 'netFlux']].head())
	days = ['02d', '04d', '07d', '14d', '21d']
	df.to_csv("groups_flux_sum.tsv", sep='\t', index=False)
	df['subsystems'] = df['subsystems'].str.replace('_',' ')
	# df.sort_values(by="subsystems", key=lambda column: column.map(lambda e: subsystms.index(e)), inplace=True)

	# -------------------------- abs carbon netFlux Control only 
	ht, wt = 700, 1000
	fig = px.line(
		df[df["treatment"] == 'Control'],
		# x=trmt,
		# y=control_name,
		x="time_stamp",
		y="abs_C_netFlux",
		color="subsystems",
		title="Carbon Flux",
		# symbol= 'subsystems',
		# color_continuous_scale='icefire',
		# range_color=[-1*mx, mx],
		facet_row="class",
		facet_col="spc",
		# facet_row=trmt_col,
		# hover_data=["rxn_ID"],
		# labels={"rxn_score_I_dist_"+trmt: "Flux Dist"},
		# category_orders=category_orders
		height=ht, width=wt,
		# , facet_row_spacing=0.08
		category_orders={"time_stamp": days, 'class': classes},
		facet_row_spacing=0.03
	)
	fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

	# fig.show()

	# -------------------------- abs carbon netFlux Control and FeLim 
	fnt_size = 16
	fnt_lng = 13
	ht, wt = 400, 1000
	fig = px.line(
		df[df["treatment"].isin(['Control', 'FeLim'])],
		# x=trmt,
		# y=control_name,
		x="time_stamp",
		y="abs_C_netFlux",
		color="subsystems",
		symbol = "subsystems",
		title="Carbon Flux",
		line_dash = 'treatment',
		facet_row="spc",
		facet_col="class",
		# color_discrete_sequence=px.colors.qualitative.Pastel,
		# facet_row=trmt_col,
		# hover_data=["rxn_ID"],
		labels={"abs_C_netFlux": "Abs. C flux", "time_stamp":"Time Point"},
		# category_orders=category_orders
		height=ht, width=wt,
		# , facet_row_spacing=0.08
		category_orders={"time_stamp": days, 'class': classes, "subsystems": subsystms_sort},
		facet_row_spacing=0.03
	)
	fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

	fig.update_traces(line={'width': 1},)
	fig.update_layout(
		legend_title = "Metabolic Pathway", 
		showlegend=True,
		font=dict(
			family="Arial",#Courier New, Arial
			size=fnt_size,
		),
		legend=dict(
			font=dict(
				family="Arial",
				size=fnt_lng,
				color="black"
			)
		),
	)

	region_lst = []
	for trace in fig.data:
		print(trace["name"])
		pathway, trmt = trace["name"].split(",")[0], trace["name"].split(",")[1]
		# trmt = 

		if pathway not in region_lst and 'control' in trmt.lower():
			# trace["showlegend"] = True
			trace.update(name=pathway)
			region_lst.append(pathway)
			trace.update(showlegend=True)
		else:
			trace.update(showlegend=False)

	# Add the text annotation
	fig.add_annotation(
		text= "Treatment", #"<b>Treatment</b>",
		xref="paper", yref="paper",
		# x=0.89, y=1,
		x=1.15, y=-0.04,
		showarrow=False,
		font=dict(size=16, color="black", family='Arial'), #fnt_lng
		align="right"
	)

	fig.add_annotation(
		text="Control",
		xref="paper", yref="paper",
		# x=0.89, y=1,
		x=1.105, y=-0.13,
		showarrow=False,
		font=dict(size=13, color="black", family='Arial'),
		align="right"
	)
	# Add the line
	fig.add_shape(
		type="line",
		xref="paper", yref="paper",
		# x0=0.9, y0=0.95,
		# x1=0.97, y1=0.95,
		x0=1.12, y0=-0.09,
		x1=1.19, y1=-0.09,
		line=dict(color="black", width=1)
	)

	# Add the text annotation
	fig.add_annotation(
		text="FeLim",
		xref="paper", yref="paper",
		# x=0.89, y=0.93,
		x=1.095, y=-0.22,
		showarrow=False,
		font=dict(size=13, color="black", family='Arial'),
		align="right"
	)
	# Add the line
	fig.add_shape(
		type="line",
		xref="paper", yref="paper",
		# # x=0.89, y=0.93,
		# x0=0.9, y0=0.87,
		# x1=0.97, y1=0.87,
		x0=1.12, y0=-0.18,
		x1=1.19, y1=-0.18,
		line=dict(color="black", width=1, dash='3, 3, 3, 3')
	)

	fig.show()
	plot_path = f"{trmt}_Carbon_imbalance.png"
	#chlorophyll biosynthesis
	pio.write_image(fig, plot_path, scale=6, width=wt, height=ht)

	# -------------------------- netFlux
	# df['abs_netFlux'] = np.abs(df['netFlux'])
	ht, wt = 500, 1500
	fig = px.line(
		df[df["treatment"].isin(['Control', 'FeLim'])],
		# x=trmt,
		# y=control_name,
		x="time_stamp",
		y="netFlux", #"abs_netFlux",
		color="subsystems",
		title="Metabolite flux - Control and FeLim",
		line_dash = 'treatment',
		# color_continuous_scale='icefire',
		# range_color=[-1*mx, mx],
		facet_row="spc",
		facet_col="class",
		# facet_row=trmt_col,
		# hover_data=["rxn_ID"],
		# labels={"rxn_score_I_dist_"+trmt: "Flux Dist"},
		# category_orders=category_orders
		height=ht, width=wt,
		# , facet_row_spacing=0.08
		category_orders={"time_stamp": days, 'class': classes},
		facet_row_spacing=0.03
	)
	# fig.show()

	print(abc)

	netFlux_df.drop(columns=["rxn_ID"], inplace=True)
	for ts in time_stamps: 
		for trmt in treatments: 
			netFlux_trmt_df = netFlux_df[(netFlux_df['spc']==spc) 
										& (netFlux_df['treatment']==trmt)
										& (netFlux_df['time_stamp']==ts)]
			netFlux_trmt_df = netFlux_trmt_df[['met_ID', 'netFlux']]
			netFlux_trmt_df = netFlux_trmt_df.set_index('met_ID')

			trmt_pred_flux = predFlux_df[(predFlux_df['spc']==spc) 
										& (predFlux_df['treatment']==trmt)
										& (predFlux_df['time_stamp']==ts)]
			trmt_pred_flux = trmt_pred_flux[['rxn_ID', 'predFlux']]
			trmt_pred_flux = trmt_pred_flux.set_index('rxn_ID')


			netFlux_trmt_df = netFlux_trmt_df[(np.abs(netFlux_trmt_df['netFlux'])>=netFlux_threshold)]
			metabolites = list(netFlux_trmt_df.index.unique())

			# Include only a list of pre-determined reactions
			if cyto_rxns: 
				trmt_pred_flux = trmt_pred_flux[(trmt_pred_flux.index.isin(cyto_rxns))]
			# of keep all metabolites at or above the 95th percentile. 
			else: 
				cyto_rxns = find_reactions(metabolites, trmt_pred_flux, trmt)

			print(len(cyto_rxns))

			elements  = update_elements(metabolites, cyto_rxns, netFlux_trmt_df, trmt_pred_flux)
			graph_data_list.append({
					"id" : f'{ts} - {trmt}',
					"elements" : elements, 
				})
	return graph_data_list

def save_cytoscape_as_html(elements, stylesheet, trmt, filename='cytoscape_graph.html'):
	"""
	Save the Cytoscape graph as an HTML file.
	"""
	# Create the HTML structure for the Cytoscape graph
	html_content = f"""
	<!DOCTYPE html>
	<html>
	<head>
		<title>Cytoscape Graph</title>
		<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.19.0/cytoscape.min.js"></script>
		<style>
			#cy {{
				width: 100%;
				height: 900px;
			}}
		</style>
	</head>
	<body>
		<div id="cy"></div>
		<script>
			var cy = cytoscape({{
				container: document.getElementById('cy'),
				elements: {json.dumps(elements)},
				style: {json.dumps(stylesheet)},
				layout: {{
					name: 'cose'
				}}
			}});
		</script>
	</body>
	</html>
	"""
	
	# Write the HTML content to a file
	filename = f"html/{spc}_{genotype}_{time_stamp}_{trmt}_{filename}"
	with open(filename, 'w') as f:
		f.write(html_content)

	print(f"Graph saved to {filename}")


def cyto_app_oneGraph(metabolites, cyto_rxns, trmt_pred_flux, trmt):
	cyto_stylesheet=[
		# Group selectors
		{
			'selector': 'node',
			'style': {
				'content': 'data(label)',
				'font-size': '10px'
			}
		},
		# of interest color
		{
			'selector': '.red',
			'style': {
				'background-color': 'red',
				'line-color': 'red'
			}
		},
		# metanolites color
		{
			'selector': '.blue',
			'style': {
				'background-color': 'blue',
				'line-color': 'blue'
			}
		},
		# reactions color
		{
			'selector': '.green',
			'style': {
				'background-color': 'green',
				'line-color': 'green'
			}
		},
		{
			'selector': '.gray',
			'style': {
				'background-color': 'gray',
				'line-color': 'gray'
			}
		},
		# media color
		{
			'selector': '.yellow',
			'style': {
				'background-color': 'yellow',
				'line-color': 'yellow'
			}
		},
		{
			'selector': '.two_dir',
			'style': {
				'source-arrow-color': 'purple',
				'source-arrow-shape': 'triangle',
				'target-arrow-shape': 'triangle',
				'line-color': 'purple'
			}
		},
		{
			'selector': '.one_dir',
			'style': {
				'width': 1,
				'target-arrow-color': 'black',
				'target-arrow-shape': 'triangle',
				'line-color': 'black', 
				'curve-style': 'straight'
			}
		}
	]

	app = Dash(__name__)

	dash_element_list = update_elements(metabolites, cyto_rxns, trmt_pred_flux)

	save_cytoscape_as_html(dash_element_list, cyto_stylesheet, trmt)
	# print(dash_element_list)
	app.layout = html.Div([
		cyto.Cytoscape(
			id='cytoscape',
			elements=dash_element_list,
			style={'width': '100%', 'height': '900px'},
			layout={'name': 'cose'}, #'breadthfirst'},
			stylesheet=cyto_stylesheet
		)
	])


	return app


def cyto_app(cyto_rxns, spc, netFlux_df, predFlux_df, treatments=['FeLim']):
	cyto_stylesheet=[
		# Group selectors
		{
			'selector': 'node',
			'style': {
				'content': 'data(label)',
				'font-size': '10px'
			}
		},
		# of interest color
		{
			'selector': '.red',
			'style': {
				'background-color': 'mapData(flux, 0, 30, "#FFAAAA", "#8B0000")',  # Dark red to light red gradient
				'line-color': 'mapData(flux, 0, 30, "#FFAAAA", "#8B0000")',  # Border color gradient
				# 'background-color': 'mapData(flux, 0, 30, "white", "red")',  # Dark red to light red gradient
				# 'line-color': 'mapData(flux, 0, 30, "white", "red")',  # Border color gradient
				'shape': 'rectangle',  # Rectangle shape (for square)
				'width': 25,			# Set width
				'height': 25			# Set height
			}
		},
				# of interest color
		{
			'selector': '.red3',
			'style': {
				'background-color': 'darkred', # 'firebrick', '#660708ff'
				'line-color': 'darkred',		# Set the border color of the node
				'shape': 'rectangle',  # Rectangle shape (for square)
				'width': 25,			# Set width
				'height': 25			# Set height
			}
		},
		{
			'selector': '.red2',
			'style': {
				'background-color': 'firebrick', # 'firebrick'  '#a4161aff'
				'line-color': 'firebrick',		# Set the border color of the node
				'shape': 'rectangle',  # Rectangle shape (for square)
				'width': 25,			# Set width
				'height': 25			# Set height
			}
		},
		{
			'selector': '.red1',
			'style': {
				'background-color': 'red', # '#ba181bff'
				'line-color': 'red',		# Set the border color of the node
				'shape': 'rectangle',  # Rectangle shape (for square)
				'width': 25,			# Set width
				'height': 25			# Set height
			}
		},
		{
			'selector': '.red0',
			'style': {
				'background-color': 'lightcoral', # 'lightcoral'  '#e5383bff'
				'line-color': 'lightcoral',		# Set the border color of the node
				'shape': 'rectangle',  # Rectangle shape (for square)
				'width': 25,			# Set width
				'height': 25			# Set height
			}
		},
		# metanolites color
		{
			'selector': '.blue',
			'style': {
				'background-color': 'blue',
				'line-color': 'blue',		# Set the border color of the node
				'shape': 'rectangle',		# Set the shape to rectangle (for square)
				'width': 25,				 # Set the width of the node
				'height': 25
			}
		},
		# reactions color
		{
			'selector': '.green',
			'style': {
				'background-color': 'green',
				'line-color': 'green'
			}
		},
		{
			'selector': '.gray',
			'style': {
				'background-color': 'gray',
				'line-color': 'gray'
			}
		},
		# media color
		{
			'selector': '.yellow',
			'style': {
				'background-color': 'yellow',
				'line-color': 'yellow'
			}
		},
		{
			'selector': '.two_dir',
			'style': {
				'source-arrow-color': 'purple',
				'source-arrow-shape': 'triangle',
				'target-arrow-shape': 'triangle',
				'line-color': 'purple'
			}
		},
		{
			'selector': '.one_dir',
			'style': {
				'width': 1,
				'target-arrow-color': 'black',
				'target-arrow-shape': 'triangle',
				'line-color': 'black', 
				'curve-style': 'straight'
			}
		}
	]

	app = Dash(__name__)

	graph_data_list = elements_multiple(cyto_rxns, spc, netFlux_df, predFlux_df, treatments)

	app.layout = html.Div([
		html.H1(f"Chlorophyll pathways in {spc}"),
		
		# Create subgraphs dynamically using a loop
		html.Div([
			html.Div([
				html.H3("{}".format(graph_elements['id'])),
				cyto.Cytoscape(
					id=f'cyto-graph-{i + 1}',
					elements=graph_elements['elements'],
					layout={'name': 'cose'},
					stylesheet=cyto_stylesheet,
					style={'width': '500px', 'height': '500px'}
					# style={'width': '100%', 'height': '900px'},
				)
			], style={'display': 'inline-block', 'width': '48%'})
			for i, graph_elements in enumerate(graph_data_list)
		], style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-between'})
	])

	
	return app

	# Define the layout of the app
	


def graph_imbalance(app, treatments=['FeLim']): 
	file_name = f"html/{spc}_{genotype}_allTimePoints_{treatments[0]}_cytoscape_graph.html"
	with open(file_name, 'w') as f:
		f.write(app.index())
	print(app.index())
	app.run_server(debug=True)


def flux_scatter_cyto(netFlux_df, treatments):

	color_map = {'ZnEx': 'rgba(64, 61, 88, 1)', 'FeEX': 'rgba(0, 175, 181, 1)', 'ZnLim': 'rgba(170, 189, 140, 1)',
	'Control': 'rgba(206, 83, 116, 1)', 'FeLim': 'rgba(243, 155, 109, 1)'}


	netFlux_df = netFlux_df[np.abs(netFlux_df['netFlux']) > 5]
	# netFlux_df = netFlux_df.set_index('met_ID')
	# netFlux_df = netFlux_df.sort_values(by='netFlux')
	# netFlux_df.sort_index(inplace=True)

	
	for spc in species:
			## scatter plot 
		fig = make_subplots(rows=len(treatments), cols=len(time_stamps),
					# specs=specs, 
					# subplot_titles=titles, 
					# shared_xaxes=True,
					shared_xaxes=False, 
					shared_yaxes=True,
					# shared_yaxes=True,
					# vertical_spacing=0.02,
					horizontal_spacing=0.02, 
					subplot_titles=[f'{trmt} - {ts}' for trmt in treatments for ts in time_stamps])
		col = 0
		for ts in time_stamps:
			col += 1
			row = 0 
			for trmt in treatments:
				row += 1
				print(trmt, "-*-"*20)
				trmt_matrix = netFlux_df[(netFlux_df['spc'] == spc) & 
									(netFlux_df['time_stamp'] == ts) &
									(netFlux_df['treatment']== trmt)][['met_ID', 'netFlux', 'C_netFlux']]
				# trmt_matrix = netFlux_df[(netFlux_df['spc'] == spc) & 
				# 					(netFlux_df['time_stamp'] == ts) &
				# 					(netFlux_df['treatment'].isin(treatments))][['met_ID', 'netFlux', 'C_netFlux']]

				

				trmt_matrix = trmt_matrix.sort_values(by='netFlux')
				# metabolites = list(trmt_matrix.index.unique())
				# find_reactions(metabolites, model_path, trmt_pred[trmt], trmt_matrix, trmt)

				# print(trmt_matrix['netFlux'].describe())
				fig.add_trace(
							go.Scatter(x=trmt_matrix['met_ID'], 
									# y=trmt_matrix[trmt_matrix['treatment']== trmt]["netFlux"], 
									y=trmt_matrix["netFlux"], 
									name = f"{trmt} - Net Flux", 
									# mode = 'markers',
									marker = dict(
										color=color_map[trmt],  # Marker color
										size=7,		# Marker size
										symbol='circle' # Marker shape
									),
									line=dict(
										color=color_map[trmt],  # Line color
										width=2,		# Line width
										
									), 
									showlegend=True
								), row=row, col=col#, secondary_y=False#, showlegend=True  
						)

		fig.update_xaxes(title_text="Metabolite")
		fig.update_yaxes(title_text="Net Flux")
		# san serif
		fig.update_layout(height=800, width=1600, font=dict(family='Times New Roman'), showlegend=True)
		# plot_path = f"export_cBalance_{trmt}.png"
		# pio.write_image(fig, plot_path, scale=5, width=380, height=500)
		fig.show()


def flux_cdf(netFlux_df, treatments):
	color_map_sns = {'ZnEx': (64/255, 61/255, 88/255, 1), 
					'FeEX': (0/255, 175/255, 181/255, 1), 
					'ZnLim': (170/255, 189/255, 140/255, 1),
					'Control': (206/255, 83/255, 116/255, 1), 
					'FeLim': (243/255, 155/255, 109/255, 1)}

	## FOR CDF 
	# Set the seaborn style for better aesthetics
	sns.set(style="whitegrid")
	# Create the plot
	plt.figure(figsize=(10, 6))

	for trmt in treatments:
		print(trmt, "-*-"*20)

		# Plot KDE for each treatment DataFrame 
		# sns.kdeplot(trmt_matrix['netFlux'], label=f'Treatment {trmt}', 
		#			  fill=True, color=color_map_sns[trmt])
		sns.ecdfplot(trmt_matrix['netFlux'], label=f'Treatment {trmt}', 
					linewidth=1.5, color=color_map_sns[trmt]) #, linestyle='-.'
		# sns.ecdfplot(trmt_matrix['C_netFlux'], label=f'Carbon {trmt}', 
		#			 linewidth=1.5, linestyle='-.', color=color_map_sns[trmt])




	# Add titles and labels
	plt.title('PDF of Net Flux/Carbon flux by Treatment', fontsize=16)
	plt.xlabel('Net Flux/Carbon flux', fontsize=12)
	plt.ylabel('Density', fontsize=12)
	# Add a legend to differentiate treatments
	plt.legend(title='Treatment')
	# Show the plot
	plt.show()


def net_flux_cdfs(netFlux_df, clm, treatments=['Control', 'FeLim']):
	color_map_sns = {'ZnEx': (64/255, 61/255, 88/255, 1), 
					'FeEX': (0/255, 175/255, 181/255, 1), 
					'ZnLim': (170/255, 189/255, 140/255, 1),
					'Control': (206/255, 83/255, 116/255, 1), 
					'FeLim': (243/255, 155/255, 109/255, 1)}
	color_map_spc = {'FeEX': (64/255, 61/255, 88/255, 1), 
					'Poplar_Control': (0/255, 175/255, 181/255, 1), 
					'Sorghum_FeLim': (170/255, 189/255, 140/255, 1),
					'Sorghum_Control': (206/255, 83/255, 116/255, 1), 
					'Poplar_FeLim': (243/255, 155/255, 109/255, 1)}

	# Create a figure with subplots, one for each day
	fig, axes = plt.subplots(len(time_stamps), 1, figsize=(14, 6), sharey=True, sharex=True)

	# Plot CDF for each day (d1 and d2)
	for i, ts in enumerate(time_stamps):
		ax = axes[i]  # Get the subplot for the current day
		
		# Filter data for the current day
		day_data = netFlux_df[netFlux_df['time_stamp'] == ts]
		for trmt in treatments:
			for spc in species:

				# Plot CDF for var1 and var2 in the same subplot
				# Plot KDE for each treatment DataFrame 
				# sns.kdeplot(day_data['netFlux'], label=f'Treatment {trmt}', 
				#			  fill=True, color=color_map_sns[trmt])
				# sns.ecdfplot(day_data[(day_data['spc']==spc) & (day_data['treatment']==trmt)][clm]
				# 			, label=f'{spc} - {trmt}', 
				# 			linewidth=1.5, color=color_map_spc[f"{spc}_{trmt}"], 
				# 			ax=ax) #, linestyle='-.'
				sns.kdeplot(day_data[(day_data['spc']==spc) & (day_data['treatment']==trmt)][clm]
							, label=f'{spc} - {trmt}', 
							linewidth=1.5, color=color_map_spc[f"{spc}_{trmt}"], 
							ax=ax) #, linestyle='-.'
				# sns.ecdfplot(day_data['C_netFlux'], label=f'Carbon {trmt}', 
				#			 linewidth=1.5, linestyle='-.', color=color_map_sns[trmt])

		# Set title and labels
		ax.set_title(f"CDF for {ts}")
		ax.set_xlabel('Net Flux')
		ax.set_ylabel('CDF')
		
		# Add a legend
		ax.legend()

	# Adjust layout for better spacing
	plt.tight_layout()

	# Show the plot
	plt.show()

	return True


def net_flux_CDF_plotly(netFlux_df, clm, treatments=['Control', 'FeLim']): 
	color_map_spc = {'FeEX': 'rgba(64, 61, 88, 1)', 
					'Poplar_Control': 'rgba(0, 175, 181, 1)', 
					'Sorghum_FeLim': 'rgba(170, 189, 140, 1)',
					'Sorghum_Control': 'rgba(206, 83, 116, 1)', 
					'Poplar_FeLim': 'rgba(243, 155, 109, 1)'}

	dict_dash = {'Poplar': 'dashdot', 'Sorghum': 'solid'}

	# Create subplots for each 'day'
	fig = make_subplots(
		rows=1, cols=len(time_stamps),  # 1 row, 2 columns for 'd1' and 'd2'
		shared_xaxes=True,
		shared_yaxes=True,
		subplot_titles=[f'CDF for {ts}' for ts in time_stamps]
	)

	# Loop over each day and plot the CDF for var1 and var2
	for i, ts in enumerate(netFlux_df['time_stamp'].unique()):
		day_data = netFlux_df[netFlux_df['time_stamp'] == ts]
		legend = (i == 0)

		for trmt in treatments:
			for spc in species:
				# Compute the CDF for var1
				var_data = day_data[(day_data['spc']==spc) & (day_data['treatment']==trmt)][clm]
				# var1_sorted = sorted(var_data)
				# var1_cdf = pa.Series(var1_sorted).cumsum() / pa.Series(var1_sorted).sum()  # Normalize the cumulative sum

				var1_sorted = np.sort(var_data)
				# Calculate the cumulative proportion of the sorted data
				var1_cdf = np.cumsum(np.abs(var1_sorted)) / np.sum(np.abs(var1_sorted))
				
				
				# Plot CDF for var1
				fig.add_trace(
					go.Scatter(
						x=var1_sorted, 
						y=var1_cdf,  # CDF for var1
						mode='lines', 
						name=f'{spc} - {trmt}', 
						line=dict(color=color_map_spc[f"{spc}_{trmt}"],
							dash=dict_dash[spc]), 
						showlegend = legend
					),
					row=1, col=i + 1
				)
				
	# Update layout
	fig.update_layout(
		title_text="CDF of for all combinations",
		showlegend=True,
		height=400, width=1800
	)

	# Show plot
	fig.show()


def read_metabolite_netFluxes():
	netFlux_df = pa.DataFrame()
	for spc in species:
		for trmt in treatments:
			for ts in time_stamps:
				df = pa.read_csv(f"{saveTo}{spc}_{ts}_{trmt}_s_matrix.tsv", sep='\t')
				# print(df.tail(2))
				df.rename(columns={'Unnamed: 0': 'met_ID', 'sum': 'netFlux', "sum_c":'C_netFlux'}
						, inplace=True)  # Rename the new 'index' column to 'id'
				
				# Add columns for var1, var2, and var3
				df['spc'] = spc
				df['treatment'] = trmt
				df['time_stamp'] = ts
				
				# Compute the sum of the values in the DataFrame (ignoring the var columns)
				df = df.rename(columns={'sum': 'netFlux', "sum_c":'C_netFlux'})
				df = df[['met_ID', 'netFlux', 'C_netFlux', 'spc', 'treatment', 'time_stamp']]
				
				# Append the DataFrame to the list
				# print(df.head())
				# print(df.tail(2))
				netFlux_df = pa.concat([netFlux_df, df], ignore_index=True)

	# netFlux_df.reset_index(inplace=True)
	# print(netFlux_df.columns)
	# netFlux_df = netFlux_df[['met_ID', 'netFlux', 'C_netFlux', 'spc', 'treatment', 'time_stamp']]
	print(netFlux_df.tail())
	# print(abc)
	return netFlux_df

def read_reaction_predFlux(): 
	predFlux_df = pa.DataFrame()
	for spc in species:
		# for trmt in treatments:
		for ts in time_stamps:
			df = pa.read_csv(f"{saveTo}{spc}_Leaf_{ts}_complexFix_noADP_noP_V_rxn_fba_Vbf_RES.tsv", sep='\t')
			# df = pa.read_csv(f"{saveTo}{spc}_{ts}_{trmt}_s_matrix.tsv", sep='\t')
			
			df = df[["rxn_ID", "treatment", "Pred"]]
			# df['rxn_ID_only'] = df.apply(lambda row: get_rxn_ID(row), axis=1)
			df.rename(columns={'Pred': 'predFlux'}
					, inplace=True)  # Rename the new 'index' column to 'id'
			
			# Add columns for var1, var2, and var3
			df['spc'] = spc
			df['time_stamp'] = ts
			# value_cols = ["predFlux"]
			# # Consilidate the fluxes of reversible reactions:
			# # if V_r > V_f:
			# # 	 V_r = V_r - V_f
			# # 	 V_f = 0
			# # else:
			# # 	 V_f = V_f - V_r
			# # 	 V_r = 0
			# df = df.groupby(['rxn_ID_only', 'spc', 'time_stamp', 'treatment'], as_index=False).apply(lambda grp: consolidate(grp, value_cols))
			# pred_df['predFlux'] = df.apply(lambda row: row['predFlux'] * -1 if '_r' in row['rxn_ID'] else row['predFlux'], axis=1)

			# Append the DataFrame to the list
			# print(df.head())
			# print(abc)
			predFlux_df = pa.concat([predFlux_df, df], ignore_index=True)

	# predFlux_df.reset_index(inplace=True)
	print(predFlux_df.tail())
	return predFlux_df
	
	


if __name__ == '__main__':
	netFlux_df = read_metabolite_netFluxes()
	predFlux_df = read_reaction_predFlux()
	plot_chlorophill_lines(predFlux_df)

	chlorophyll_rxns = ["rxn19846_d0", "rxn01629_d0", "rxn00029_d0", "rxn00060_d0", "rxn02264_d0", 
	"rxn02288_d0", "rxn02303_d0", "rxn02304_d0", "rxn02733_d0", "rxn02959_d0", 
	"rxn04152_d0", "rxn04153_d0", "rxn04154_d0"]
	chlorophyll_rxns_rev = chlorophyll_rxns \
					+ [rxn+"_f" for rxn in chlorophyll_rxns] \
					+ [rxn+"_r" for rxn in chlorophyll_rxns]

	# flux_scatter_cyto(netFlux_df, ['Control', 'FeLim'])
	# clm = 'netFlux' # 'C_netFlux'
	# net_flux_cdfs(netFlux_df, clm)
	# net_flux_CDF_plotly(netFlux_df, clm)

	spc = 'Sorghum'
	app = cyto_app(chlorophyll_rxns_rev, spc, netFlux_df, predFlux_df)
	graph_imbalance(app)




# def cyto_app(cyto_rxns, spc, netFlux_df, predFlux_df, treatments=['FeLim']):
# 	cyto_stylesheet=[
# 		# Group selectors
# 		{
# 			'selector': 'node',
# 			'style': {
# 				'content': 'data(label)',
# 				'font-size': '10px'
# 			}
# 		},
# 		# # of interest color
# 		# {
# 		# 	'selector': '.red',
# 		# 	'style': {
# 		# 		'background-color': 'mapData(flux, 0, 30, "#FFAAAA", "#8B0000")',  # Dark red to light red gradient
# 		# 		'line-color': 'mapData(flux, 0, 30, "#FFAAAA", "#8B0000")',  # Border color gradient
# 		# 		# 'background-color': 'mapData(flux, 0, 30, "white", "red")',  # Dark red to light red gradient
# 		# 		# 'line-color': 'mapData(flux, 0, 30, "white", "red")',  # Border color gradient
# 		# 		'shape': 'rectangle',  # Rectangle shape (for square)
# 		# 		'width': 25,			# Set width
# 		# 		'height': 25			# Set height
# 		# 	}
# 		# },
# # 		$blood-red: rgba(102, 7, 8, 1);
# # $cornell-red: rgba(164, 22, 26, 1);
# # $cornell-red-2: rgba(186, 24, 27, 1);
# # $imperial-red: rgba(229, 56, 59, 1);
# # --blood-red: #660708ff;
# # --cornell-red: #a4161aff;
# # --cornell-red-2: #ba181bff;
# # --imperial-red: #e5383bff;
# 		# of interest color
# 		# {
# 		# 	'selector': '.red3',
# 		# 	'style': {
# 		# 		'background-color': 'darkred', # 'firebrick', '#660708ff'
# 		# 		'line-color': 'darkred',		# Set the border color of the node
# 		# 		'shape': 'rectangle',  # Rectangle shape (for square)
# 		# 		'width': 25,			# Set width
# 		# 		'height': 25			# Set height
# 		# 	}
# 		# },
# 		# {
# 		# 	'selector': '.red2',
# 		# 	'style': {
# 		# 		'background-color': 'firebrick', # 'firebrick'  '#a4161aff'
# 		# 		'line-color': 'firebrick',		# Set the border color of the node
# 		# 		'shape': 'rectangle',  # Rectangle shape (for square)
# 		# 		'width': 25,			# Set width
# 		# 		'height': 25			# Set height
# 		# 	}
# 		# },
# 		# {
# 		# 	'selector': '.red1',
# 		# 	'style': {
# 		# 		'background-color': 'red', # '#ba181bff'
# 		# 		'line-color': 'red',		# Set the border color of the node
# 		# 		'shape': 'rectangle',  # Rectangle shape (for square)
# 		# 		'width': 25,			# Set width
# 		# 		'height': 25			# Set height
# 		# 	}
# 		# },
# 		# {
# 		# 	'selector': '.red0',
# 		# 	'style': {
# 		# 		'background-color': 'lightcoral', # 'lightcoral'  '#e5383bff'
# 		# 		'line-color': 'lightcoral',		# Set the border color of the node
# 		# 		'shape': 'rectangle',  # Rectangle shape (for square)
# 		# 		'width': 25,			# Set width
# 		# 		'height': 25			# Set height
# 		# 	}
# 		# },
# 		{
# 			'selector': '.red3',
# 			'style': {
# 				'background-color': 'black', # 'firebrick', '#660708ff'
# 				'line-color': 'black',		# Set the border color of the node
# 				# 'shape': 'rectangle',  # Rectangle shape (for square)
# 				# 'width': 25,			# Set width
# 				# 'height': 25			# Set height
# 			}
# 		},
# 		{
# 			'selector': '.red2',
# 			'style': {
# 				'background-color': 'red', # 'firebrick'  '#a4161aff'
# 				'line-color': 'red',		# Set the border color of the node
# 				# 'shape': 'rectangle',  # Rectangle shape (for square)
# 				# 'width': 25,			# Set width
# 				# 'height': 25			# Set height
# 			}
# 		},
# 		{
# 			'selector': '.red1',
# 			'style': {
# 				'background-color': 'red', # '#ba181bff'
# 				'line-color': 'red',		# Set the border color of the node
# 				# 'shape': 'rectangle',  # Rectangle shape (for square)
# 				# 'width': 25,			# Set width
# 				# 'height': 25			# Set height
# 			}
# 		},
# 		{
# 			'selector': '.red0',
# 			'style': {
# 				'background-color': 'lightcoral', # 'lightcoral'  '#e5383bff'
# 				'line-color': 'lightcoral',		# Set the border color of the node
# 				# 'shape': 'rectangle',  # Rectangle shape (for square)
# 				# 'width': 25,			# Set width
# 				# 'height': 25			# Set height
# 			}
# 		},
# 		# metanolites color
# 		{
# 			'selector': '.blue',
# 			'style': {
# 				'background-color': 'black',
# 				'line-color': 'clack',		# Set the border color of the node
# 				'shape': 'rectangle',		# Set the shape to rectangle (for square)
# 				'width': 25,				 # Set the width of the node
# 				'height': 25
# 			}
# 		},
# 		# reactions color
# 		{
# 			'selector': '.green',
# 			'style': {
# 				'background-color': 'green',
# 				'line-color': 'green'
# 			}
# 		},
# 		{
# 			'selector': '.gray',
# 			'style': {
# 				'background-color': 'gray',
# 				'line-color': 'gray'
# 			}
# 		},
# 		# media color
# 		{
# 			'selector': '.yellow',
# 			'style': {
# 				'background-color': 'yellow',
# 				'line-color': 'yellow'
# 			}
# 		},
# 		{
# 			'selector': '.two_dir',
# 			'style': {
# 				'source-arrow-color': 'purple',
# 				'source-arrow-shape': 'triangle',
# 				'target-arrow-shape': 'triangle',
# 				'line-color': 'purple'
# 			}
# 		},
# 		{
# 			'selector': '.one_dir',
# 			'style': {
# 				'width': 1,
# 				'target-arrow-color': 'black',
# 				'target-arrow-shape': 'triangle',
# 				'line-color': 'black', 
# 				'curve-style': 'straight'
# 			}
# 		}
# 	]

# 	app = Dash(__name__)

# 	graph_data_list = elements_multiple(cyto_rxns, spc, netFlux_df, predFlux_df, treatments)

# 	app.layout = html.Div([
# 		html.H1(f"Chlorophyll pathways in {spc}"),
		
# 		# Create subgraphs dynamically using a loop
# 		html.Div([
# 			html.Div([
# 				html.H3("{}".format(graph_elements['id'])),
# 				cyto.Cytoscape(
# 					id=f'cyto-graph-{i + 1}',
# 					elements=graph_elements['elements'],
# 					layout={'name': 'cose'},
# 					stylesheet=cyto_stylesheet,
# 					style={'width': '500px', 'height': '500px'}
# 					# style={'width': '100%', 'height': '900px'},
# 				)
# 			], style={'display': 'inline-block', 'width': '48%'})
# 			for i, graph_elements in enumerate(graph_data_list)
# 		], style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-between'})
# 	])


# 	return app

# 	# Define the layout of the app

