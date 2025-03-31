#!/usr/bin/env python3

import glob
import subprocess
import numpy as np
import scipy
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import timeit

from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy import interpolate
import scipy.signal as signal
from scipy.fftpack import rfft, irfft, fftfreq

import os
import sys

#import headers.read_data as read_data
import headers.plots as plots
#import headers.bklhd_support_functions as bklhd
#import headers.data_manipulation as data_manipulation
import headers.timescale as timescale
import headers.timeseries_stats as timeseries_stats
#import headers.weights as weights
#import headers.newfit as newfit

import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import ConnectionPatch
from matplotlib import image
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
from scipy import stats

import seaborn as sns
import plotly.express as px

mpl.rcParams['pdf.fonttype'] = 42  # Use Type 3 (vector) fonts in PDF
mpl.rcParams['ps.fonttype'] = 42   # Use Type 3 (vector) fonts in PS
mpl.rcParams['text.usetex'] = False
mpl.rcParams['svg.fonttype'] = 'none'  # Save text as paths in SVG, preserving font appearance
mpl.rcParams['svg.image_inline'] = False  # Do not inline images, keep them as external references
mpl.rcParams['text.usetex'] = False  # Do not use TeX for text rendering
mpl.rcParams['font.family'] = 'Arial'

LETTER_FONT_SIZE = 9  # Fixed font size for labels
SCALE_BAR_FONT_SIZE = 7  # Fixed font size for scale bar text

samplingrate=1
window=2

bsamplingrate=2
bwindow=5

#xlo=538.8
xlo=545
xhi=582

# Load the data
dfo = pd.read_csv('data/figures/master_spreadsheet_geochemistry_2024_newcomp.csv')
dfb = pd.read_excel('data/figures/summary/Bowyer_2024_biostrat/ado6462_data_s2.xlsx', sheet_name='Biostratigraphic_data_long_f')
dfk = pd.read_csv('data/figures/summary/Kikumoto_2014.precamb.csv')
dfn = pd.read_csv('data/figures/summary/Canfield_2020.csv')
dfsr = pd.read_csv('data/figures/summary/Chen_2022_Shuram_newages.csv')
dffike = pd.read_csv('data/figures/summary/Fike_2006.csv')
dflee = pd.read_csv('data/figures/summary/Lee_2015.csv')
dfhuqf = pd.read_csv('data/figures/summary/Huqf.csv')
dfsed = pd.read_csv('data/figures/fig4/composite_huqf_shuramex_column.csv')

# Define the age constraints as a list of tuples (Composite_Height, AgeModel)
age_constraints_old = [
    (4.675, 578.2),
    (337.6, 574),
    (388, 572),
    (827, 567.7),
    (918.9, 562.7),
    (1193.9, 560.4),
    (1252.9, 558.4),
    (1300.6, 555.4),
    (1327.1, 553.4),
    (1396, 548),
    (1645.18, 541)
]

# Define the age constraints as a list of tuples (Composite_Height, AgeModel)
age_constraints_cantine24 = [
    (4.675, 578.2),
    (340.8,573), #added to expamd
    (341.1,572.6), #added to expamd
    (343.2,572.4), #added to expamd
    (430.6, 572),
    (897, 568),
#    (918.9, 569),
#    (827, 568),
    (964,558), #added
    (1065.5,551),
    (1252.9, 547.36),
    (1645.18, 541)
]

age_constraints_kikumoto14 = [
(35.6,547.29),
(55.56,551.06),
(62.5,568),
(104.63,573),
(158.26,614.00),
(276.73,632.48),
(281.87,635.26)
]

age_constraints_chen22 = {
    'Oman': {
        'height_column': 'Chen_2022_Age',
        'constraints': [
            (550.93, 568),
            (554.7156, 569),
            (559.85, 573),
            (565.28, 578),
        ]
    },
    'Russia': {
        'height_column': 'Chen_2022_Age',
        'constraints': [
            (530, 530),
            (563.87, 570),
            (576.266, 573),
            (581.54, 581),
        ]
    },
    'South China': {
        'height_column': 'Chen_2022_Age',
        'constraints': [
            (504.302, 504.302),
            (506.66, 506.66),
            (551.52, 551.52),
            (554.7835, 568),
            (569.95, 573),
            (590.102, 590.102),
            (636.77, 636.77),
        ]
    },
    'Australia': {
        'height_column': 'Chen_2022_Age',
        'constraints': [
            (550.9854, 568),
            (552.80, 569),
            (556.9, 570),
        ]
    },
    'China': {
        'height_column': 'Chen_2022_Age',
        'constraints': [
            (549, 550),
            (571, 567),
        ]
    },
    'Russia (Siberia)': {
        'height_column': 'Meter',
        'constraints': [
            (2291, 548),
            (1, 560),
        ]
    },
    'Russia (Siberia2)': {
        'height_column': 'Meter',
        'constraints': [
            (24, 565),
            (303, 560),
        ]
    },
}

# Sort age constraints by Composite_Height
age_constraints = sorted(age_constraints_cantine24)
age_constraints_k = sorted(age_constraints_kikumoto14)

# Extract Composite_Height and AgeModel into separate lists
heights, ages = zip(*age_constraints)
heights_k, ages_k = zip(*age_constraints_k)

# Perform linear interpolation to estimate ages for all entries
dfo['AgeModel'] = np.interp(dfo['Composite_Height'], heights, ages)
dfk['AgeModel'] = np.interp(dfk['Meter'], heights_k, ages_k)
dffike['AgeModel'] = np.interp(dffike['Composite_Height'], heights, ages)
dflee['AgeModel'] = np.interp(dflee['Composite_Height'], heights, ages)
dfhuqf['AgeModel'] = np.interp(dfhuqf['Composite_Height'], heights, ages)
dfsed['AgeModel'] = np.interp(dfsed['Composite_Height'], heights, ages)

def plot_barcode(ax, dfsed, heights, ages):

    dfsed['age_bot'] = np.interp(dfsed['Composite_Height'], heights, ages)
    dfsed['age_top'] = np.interp(dfsed['Composite_Height'] + dfsed['section_thickness'], heights, ages)

    # Define colors and tags for the barcode plot
    colorlist = ['#ca853c', '#441e11', '#842504']
    taglist = ['Peritidal', 'Storm', 'Silt']

    # Plot the barcode
    for i, (color, tag) in enumerate(zip(colorlist, taglist)):
        if tag not in dfsed.columns:
            print(f"Tag '{tag}' not found in DataFrame columns. Skipping.")
            continue

        # Filter rows where the tag column is not empty
        mask = dfsed[tag].notna() & (dfsed[tag] != ' ')
        for _, row in dfsed[mask].iterrows():
            # Ensure proper age ordering for plotting
            age_min, age_max = min(row.age_bot, row.age_top), max(row.age_bot, row.age_top)
            print(f"Tag: {tag}, Range: [{age_min}, {age_max}]")  # Debugging
            ax.fill_between([age_min, age_max],[i, i], [i + 1, i + 1],facecolor=color,edgecolor='none')

    # Set plot limits and remove y-axis ticks
    ax.set_ylim((0, len(taglist)))
    ax.set_yticks([])
    ax.set_yticklabels([])

# Apply interpolation for each country in Chen 2022
for country, data in age_constraints_chen22.items():
    # Filter the dataset by country
    country_data = dfsr[dfsr['Country'] == country]
    if country_data.empty:
        continue
    
    # Get the height column and constraints
    height_column = data['height_column']
    constraints = sorted(data['constraints'])  # Sort constraints by height
    
    # Extract heights and ages
    heights_c, ages_c = zip(*constraints)
    
    # Perform interpolation
    interpolated_ages_c = np.interp(country_data[height_column], heights_c, ages_c)
    
    # Assign the interpolated ages back to the main DataFrame
    dfsr.loc[country_data.index, 'AgeModel'] = interpolated_ages_c

dfsr = dfsr[dfsr["Section"].isin(['medium-certainty', 'high-certainty'])]
#dfsr = dfsr[~dfsr["Country"].isin(['Russia'])]
dfhuqf = dfhuqf[~dfhuqf["Section"].isin(['ST2'])]

# Add a categorical column for coloring
dfsr["Category"] = dfsr["Country"]
dfhuqf["Category"] = dfhuqf["Section"]
dffike["Category"] = "dffike"
dflee["Category"] = "dflee"

# Concatenate all dataframes into one for plotting
combined_df = pd.concat([dfsr, dfhuqf, dffike.assign(Category="dffike"), dflee.assign(Category="dflee")])

# Create an interactive scatter plot
fig = px.scatter(
    combined_df,
    x="AgeModel",
    y="d13C",
    color="Category",
    hover_data=["AgeModel", "d13C", "Category"],
    title="d13C vs Age Model (Interactive)"
)

fig.update_layout(
    xaxis=dict(range=[577, 540], autorange=False),  # Explicit range for the x-axis
    title_x=0.5  # Center the title
)

# Show the plot (or save it if running in a non-interactive environment)
fig.write_html("d13C_vs_AgeModel.html")

# Create an interactive scatter plot
fig = px.scatter(
    combined_df,
    x="AgeModel",
    y="87Sr_86Sr",
    color="Category",
    hover_data=["AgeModel", "87Sr_86Sr", "Category"],
    title="87Sr_86Sr vs Age Model (Interactive)"
)

fig.update_layout(
    xaxis=dict(range=[577, 540], autorange=False),  # Explicit range for the x-axis
    title_x=0.5  # Center the title
)

# Show the plot (or save it if running in a non-interactive environment)
fig.write_html("87Sr_86Sr_vs_AgeModel.html")

# Reshape data to long format
combined_long_mn = combined_df.melt(
    id_vars=["AgeModel", "Category"],
    value_vars=["Mn", "Mn_ppm", "Mn_Acetic_ppm"],
    var_name="Mn_Type",
    value_name="Mn_Value"
)

# Create an interactive scatter plot
fig = px.scatter(
    combined_long_mn,
    x="AgeModel",
    y="Mn_Value",
    color="Category",
    symbol="Mn_Type",  # Use different symbols for Mn types
    hover_data=["AgeModel", "Mn_Value", "Mn_Type", "Category"],
    title="Mn vs Age Model (Interactive)"
)

# Reverse the x-axis (older ages on the left)
fig.update_layout(
    xaxis=dict(range=[577, 540], autorange=False),
    title_x=0.5
)

# Save the interactive plot to an HTML file
fig.write_html("Mn_vs_AgeModel.html")

# Reshape data to long format
combined_long_sr = combined_df.melt(
    id_vars=["AgeModel", "Category"],
    value_vars=["Sr", "Sr_ppm", "Sr_Acetic_ppm"],
    var_name="Sr_Type",
    value_name="Sr_Value"
)

# Create an interactive scatter plot
fig = px.scatter(
    combined_long_sr,
    x="AgeModel",
    y="Sr_Value",
    color="Category",
    symbol="Sr_Type",  # Use different symbols for Mn types
    hover_data=["AgeModel", "Sr_Value", "Sr_Type", "Category"],
    title="Sr vs Age Model (Interactive)"
)

# Reverse the x-axis (older ages on the left)
fig.update_layout(
    xaxis=dict(range=[577, 540], autorange=False),
    title_x=0.5
)

# Save the interactive plot to an HTML file
fig.write_html("Sr_vs_AgeModel.html")

# Correct constants for mineral-specific calculations
CA_d18O_alph_params = np.array([17.5e3, 0., 29.1])
AR_d18O_alph_params = np.array([17.88e3, 0., 31.14])
DL_d18O_alph_params = np.array([0., 3.14e6, 3.14])
APO_d18O_params = np.array([117.4, 4.50, -1e10])

CAlist = ['CA']
DLlist = ['DL']
ARlist = ['AR']
APOlist = ['APO']

def vsmow_to_pdb(d18O_vsmow):
    return (d18O_vsmow - 30.92) / 1.03092

def ret_alpha_min_params(mineral):
    if mineral in CAlist:
        return CA_d18O_alph_params
    elif mineral in DLlist:
        return DL_d18O_alph_params
    elif mineral in ARlist:
        return AR_d18O_alph_params
    elif mineral in APOlist:
        return APO_d18O_params
    else:
        return np.nan, np.nan, np.nan

def ret_temp(d18Opdb, min, d18Oswvsmow=-1):
    b1, b2, c = ret_alpha_min_params(min)
    if np.isnan(c):
        return np.nan

    if c != APO_d18O_params[2]:  # If not apatite
        ao = (1e3 + np.array(d18Opdb, dtype=float)) / (1e3 + vsmow_to_pdb(d18Oswvsmow))
        logao = np.log(ao)
        cklogao = c + 1e3 * logao

        try:
            Tk = 0.5 * (b1 + np.sqrt(b1 * b1 + 4 * b2 * cklogao)) / cklogao
            Tc = Tk - 273.15
        except Exception as e:
            print(f"Error calculating temperature: {e}")
            Tc = np.nan
    else:  # For apatite, different calculation
        try:
            Tc = b1 - b2 * (d18Opdb - d18Oswvsmow)
        except Exception as e:
            print(f"Error calculating temperature for apatite: {e}")
            Tc = np.nan

    return Tc

def calculate_temperature(row):
    return ret_temp(row['d18O'], row['Min'], -1.5)

# Apply the temperature calculation function to the 'd18O' and 'Min' columns
dfo['T_calc_sw1_a21'] = dfo.apply(calculate_temperature, axis=1)

# Save the updated DataFrame to a new CSV file
dfo.to_csv('data/figures/master_spreadsheet_geochemistry_2024_with_age_model_and_temp.csv', index=False)

# Print a sample of the updated DataFrame to verify the results
print(dfo.head())

#check orgC with new age model for China (replot d13org data from new compilation; add in data from Birba for Miqrat?)

def pretty_figure(fig, axs):
    for i in range(len(axs)):
        ylims = axs[i].get_ylim()
        axs[i].set_ylim(ylims)
        axs[i].invert_xaxis()
        if i != 8:  # Skip axs[8] as we want x-axis visible for this subplot
            axs[i].get_xaxis().set_visible(False)
        axs[i].spines['bottom'].set_visible(True)
        axs[i].spines['top'].set_visible(False)
        axs[i].tick_params(direction='in')
    axs[0].spines['top'].set_visible(True)
    fig.subplots_adjust(hspace=0)  # wspace=0,

def add_panel_label(ax, label, position='top-left', weight='bold', label_y=0.95, label_x=0.05, zorder=10, **kwargs):
    kwargs.update({'weight': weight, 'fontsize': 10, 'color': 'white'})  # Add fontsize and white font color for better visibility
    bbox_props = dict(boxstyle="square,pad=0.1", edgecolor="black", facecolor="black")
    
    if position == 'top-left':
        ax.text(label_x, label_y, label, transform=ax.transAxes, bbox=bbox_props, zorder=zorder, **kwargs)
    elif position == 'top-right':
        ax.text(0.95, label_y, label, transform=ax.transAxes, bbox=bbox_props, ha='right', zorder=zorder, **kwargs)
    elif position == 'bottom-left':
        ax.text(label_x, 0.05, label, transform=ax.transAxes, bbox=bbox_props, zorder=zorder, **kwargs)
    elif position == 'bottom-right':
        ax.text(0.95, 0.05, label, transform=ax.transAxes, bbox=bbox_props, ha='right', zorder=zorder, **kwargs)

def plot_points(ax,dfdict,sdfs,tag,xlo,xhi,colorbymin=False,yticks=[],zorder=1):
	for sdf in sdfs:
		df=dfdict[sdf]
		x=np.array(df['AgeModel'])
		y=np.array(df[tag])
		mask=pd.notnull(x)
		mask&=pd.notnull(y)
		if tag == 'Mn' or tag == 'Sr':
			print(y[mask])
			y[mask]=0.001*y[mask]
		mask[mask]&=x[mask]>=xlo
		mask[mask]&=x[mask]<=xhi
		color='k'
		marker='.'
		label=''
		if colorbymin:
			color=plots.colordict[sdf]
			marker=plots.markerdict[sdf]
			label=plots.labeldict[sdf]
		ax.plot(x[mask],y[mask],ls='',color=color,marker=marker,label=label,zorder=zorder)
	ax.set_ylabel(plots.taglabeldict[tag])
	if tag == 'Mn' or tag == 'Sr':
		ax.set_ylabel(plots.taglabeldict[tag].replace('ppm','‰'))
	if len(yticks)>0:
		ax.set_yticks(yticks)

def minboxes(ax):
	ax.fill_between([579,576],[0,0],[1,1],color='#FCD489')
	ax.fill_between([576,572.77],[0,0],[1,1],color='#F3A06F')
	ax.fill_between([572.77,570.4],[0,0],[1,1],color='#E77C75')
	ax.fill_between([570.4,558],[0,0],[1,1],color='#FCD489')
	ax.fill_between([558,540],[0,0],[1,1],color='#F3A06F')
	ax.annotate('A',(np.mean([579,576]),0.45),ha='center',va='center',color='k')
	ax.annotate('DL',(np.mean([576,572.77]),0.45),ha='center',va='center',color='k')
	ax.annotate('C',(np.mean([572.77,570.4]),0.45),ha='center',va='center',color='k')
	ax.annotate('A',(np.mean([570.4,558]),0.45),ha='center',va='center',color='k')
	ax.annotate('DL',(np.mean([558,540]),0.45),ha='center',va='center',color='k')
	ax.set_ylim((0,1))
	ax.set_yticks([])

# Define the biological events panel
def biological_events_panel(ax, dfb):
    dfb = dfb.dropna(subset=['Morphogroups'])  # Remove rows with NaN in Morphogroups

    # Sort by Morphogroups for consistent plotting
    morphogroups = dfb['Morphogroups'].unique()
    y_positions = {group: idx for idx, group in enumerate(morphogroups)}

    # Define specific colors for 'Biota'
    color_map = {
        'Avalon': '#2c7fb8',  # Teal
        'White Sea': '#7fcdbb',  # Coral
        'Nama': '#edf8b1',  # Slate Gray
    }

    for _, row in dfb.iterrows():
        x_val = row['Age.occurrence']
        y_val = y_positions[row['Morphogroups']]
        color = color_map.get(row['Biota'], '#000000')  # Default to black if Biota not in color_map
        ax.scatter(x_val, y_val, color=color, alpha=0.7, edgecolor='k', s=10, zorder=10)
    
    # Customize axes
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_yticklabels([])
    ax.invert_xaxis()

fig, axs = plt.subplots(9, 1, figsize=(3.0, 7.8), gridspec_kw={'height_ratios':[0.3, 0.3, 0.3, 0.3, 0.1, 0.3, 0.3, 0.3, 0.1]}, sharex=True)

plot_barcode(axs[3], dfsed, heights, ages)
minboxes(axs[4])

#BEGIN Mn PANEL
df = combined_long_mn
ax=axs[5].twinx()
tag='Mn_Value'
qtiles=[25,50,75]
colors=sns.light_palette('#B34032',3,reverse=False)
x=np.array(df['AgeModel'])
y=np.array(df[tag])

xs,ys=timeseries_stats.moving_quantiles(x,y,bsamplingrate,bwindow,qtiles)
ax.fill_between(xs,ys[:,0],ys[:,2],fc=colors[1],zorder=100,lw=0)
ax.plot(xs,ys[:,1],color='#B34032',zorder=101)
ax.scatter(x, y, color='#B34032', zorder=200, s=2)

ax.set_ylabel('Mn (ppm)',fontsize=10)
ax.yaxis.label.set_color('#BB5B4F')
ax.tick_params(axis='y', colors='#BB5B4F')
ax.set_ylim((0,5000))
ax.get_xaxis().set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(direction='in')
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.yaxis.set_label_coords(1.15, 0.8)
ax.set_ylabel(ax.get_ylabel(),rotation=270,va="bottom") 
#END Mn PANEL

#BEGIN Sr PANEL
ax=axs[5]
df = combined_long_sr
tag='Sr_Value'
qtiles=[25,50,75]
colors=sns.light_palette('#F59E00',3,reverse=False)
x=np.array(df['AgeModel'])
y=np.array(df[tag])

xs,ys=timeseries_stats.moving_quantiles(x,y,bsamplingrate,bwindow,qtiles)
ax.fill_between(xs,ys[:,0],ys[:,2],fc=colors[1],zorder=100,lw=0)
ax.plot(xs,ys[:,1],color='#F59E00',zorder=101)
ax.scatter(x, y, color='#F59E00', zorder=200, s=2)

ax.set_ylabel('Sr (ppm)',fontsize=9,rotation=90,va="bottom")
ax.set_ylim((0,5000))
ax.get_xaxis().set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(direction='in')
ax.yaxis.set_label_position("left")
ax.yaxis.tick_left()
ax.yaxis.label.set_color('#F59E02')
ax.tick_params(axis='y', colors='#F59E02')
#END Sr PANEL

# BEGIN 87Sr_86Sr PANEL
ax = axs[1]
tag = '87Sr_86Sr'
qtiles = [25, 50, 75]
colors = sns.light_palette('#000000', 3, reverse=False)

df87Sr = combined_df

x = np.array(df87Sr['AgeModel'])
y = np.array(df87Sr[tag])

# Calculate moving quantiles for the filtered data
#xs, ys = timeseries_stats.moving_quantiles(x, y, bsamplingrate, bwindow, qtiles)

# Plot the quantile range for the filtered data
#ax.fill_between(xs, ys[:, 0], ys[:, 2], fc='#929292', zorder=10, lw=0)
# ax.plot(xs, ys[:, 1], color='k', zorder=101)
ax.scatter(x, y, color='k', zorder=11, s=2)

# Define the y-axis tick range and labels
#mask = pd.notnull(ys)
ymin, ymax = 0.7078, 0.7095
ax.set_yticks([0.708, 0.7085, 0.709])
ax.set_ylim((ymin, ymax))
ax.set_ylabel(plots.taglabeldict[tag].replace('Mineral ', '').replace(')', ' VPDB)'), fontsize=9)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax.yaxis.set_label_coords(-0.18, 0.5)
ax.tick_params(direction='in', zorder=102)
#END 87Sr_86Sr PANEL

#d13C ORG AND SILICICLASTIC PANEL
dfC = dfk #Three Gorges (Kikumoto)
dfS = dffike
dfD = dflee
dfN = dfn

# Main axis: d13C siliciclastic
ax = axs[2]  # Main axis
ax.patch.set_alpha(0)  # Transparent background for the main axis

# Plotting d13C siliciclastic
x = np.array(dfN['AgeModel'])
y = np.array(dfN['del13CIC'])
mask = pd.notnull(x) & pd.notnull(y)
mask[mask] &= (x[mask] >= xlo) & (x[mask] <= xhi)
colors = sns.light_palette('#A9A9A9', 3, reverse=False)
x = x[mask]
y = y[mask]
xs, ys = timeseries_stats.moving_quantiles(x, y, bsamplingrate, bwindow, qtiles)
#ax.fill_between(xs, ys[:, 0], ys[:, 2], fc=colors[1], zorder=1, lw=0)
#ax.plot(xs, ys[:, 1], color='#A9A9A9', zorder=2)
ax.scatter(x, y, color='#A9A9A9', zorder=3, s=2)

# Secondary axis: d13C org
ax_twinx = axs[2].twinx()  # Create a secondary y-axis
ax_twinx.patch.set_alpha(0)  # Transparent background for the twinx axis

# Plotting d13C org datasets
for df, color, zorder in zip([dfD, dfC, dfS], ['#213079', '#2990C3', '#37C8EC'], [100, 200, 300]):
    x = np.array(df['AgeModel'])
    y = np.array(df['d13Corg'])
    mask = pd.notnull(x) & pd.notnull(y)
    mask[mask] &= (x[mask] >= xlo) & (x[mask] <= xhi)
    
    # Generate a light palette for the fill color
    colors = sns.light_palette(color, n_colors=3, reverse=False)
    
    x = x[mask]
    y = y[mask]
    xs, ys = timeseries_stats.moving_quantiles(x, y, bsamplingrate, bwindow, qtiles)
    
    # Use the lighter shade for the fill, and the original color for line and scatter
    ax_twinx.fill_between(xs, ys[:, 0], ys[:, 2], fc=colors[1], zorder=zorder, lw=0)
    ax_twinx.plot(xs, ys[:, 1], color=color, zorder=zorder + 1)
    ax_twinx.scatter(x, y, color=color, zorder=zorder + 2, s=2)

# Set limits and labels for the twinx axis
ax_twinx.set_ylim(-40, -20)  # Correct Y-axis limits for d13C org
ax_twinx.set_ylabel('$δ^{13}C_{org}$ (‰)', fontsize=9)
ax_twinx.yaxis.set_label_position("left")
ax_twinx.yaxis.tick_left()
ax_twinx.set_yticks([-40, -35, -30, -25])

# Fix tick direction for twinx axis
ax_twinx.yaxis.set_tick_params(direction="in")
ax_twinx.xaxis.set_visible(False)
ax.xaxis.set_tick_params(direction="in") 

# Set limits and labels for the main axis
ax.set_ylim(-30, 0)
ax.set_ylabel('$δ^{13}C_{siliciclastics}$ (‰)', fontsize=9, va="bottom")
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.yaxis.label.set_color('#A9A9A9')
ax.tick_params(axis='y', colors='#A9A9A9')
ax.set_yticks([-30, -20, -10, 0])

# Disable left ticks and labels for the main axis
ax.spines["left"].set_visible(False)
ax.yaxis.set_ticks_position("right")
ax.yaxis.set_tick_params(left=False)
#End d13C ORG AND SILICICLASTIC PANEL

#BEGIN d13C PANEL
df = combined_df
ax=axs[0]
tag='d13C'
qtiles=[25,50,75]
colors=sns.light_palette('#000000',3,reverse=False)
x=np.array(df['AgeModel'])
y=np.array(df[tag])
mask=pd.notnull(x)&pd.notnull(y)
mask[mask]&=(x[mask]>=xlo)&(x[mask]<=xhi)
x=x[mask]
y=y[mask]

xs,ys=timeseries_stats.moving_quantiles(x,y,samplingrate,window,qtiles)
ax.fill_between(xs,ys[:,0],ys[:,2],fc=colors[1],zorder=100,lw=0)
ax.plot(xs,ys[:,1],color='k',zorder=101)
mask=pd.notnull(ys)
ymin=ys[mask].min()
ymax=ys[mask].max()
yll=ymin
yul=ymax
dy_ticks=4.
ax.set_yticks(np.arange(np.floor(yll/dy_ticks)*dy_ticks-dy_ticks,np.ceil(yul/dy_ticks)*dy_ticks+2*dy_ticks,dy_ticks))
ax.set_ylabel(plots.taglabeldict[tag],fontsize=9)
ax.set_ylim((ymin,ymax))
ax.yaxis.set_label_coords(-0.18, 0.5)
#END d13C PANEL

#BEGIN TEMP PANEL
tag='T_calc_sw1_a21'
tag2='T_MIT'
ax=axs[6]

# Extract the relevant data from the DataFrame
x = np.array(dfo['AgeModel'])
y = np.array(dfo['T_calc_sw1_a21'])
x2 = np.array(dfo['AgeModel'])
y2 = np.array(dfo['T_MIT'])

# Mask to filter out NaN values and 'Use' column
mask1 = pd.notnull(x) & pd.notnull(y)
mask2 = pd.notnull(x2) & pd.notnull(y2) & (dfo['Use'] == 1)

# Apply the masks to filter the data
x_filtered = x[mask1]
y_filtered = y[mask1]
x2_filtered = x2[mask2]
y2_filtered = y2[mask2]

# Calculate moving quantiles for the first dataset
xs, ys = timeseries_stats.moving_quantiles(x_filtered, y_filtered, samplingrate, window, qtiles)

# Define the interval for the dashed line
interval_start = 558
interval_end = 572

# Split the xs and ys arrays into three segments: before, within, and after the interval
before_interval = xs < interval_start
within_interval = (xs >= interval_start) & (xs <= interval_end)
after_interval = xs > interval_end

# Plot the quantile range for the first dataset
ax.fill_between(xs, ys[:, 0], ys[:, 2], fc='#87A274', zorder=100, lw=0)

# Plot the median line, solid outside the interval and dashed within the interval
ax.plot(xs[before_interval], ys[:, 1][before_interval], color='k', zorder=101, linestyle='solid')
ax.plot(xs[within_interval], ys[:, 1][within_interval], color='k', zorder=101, linestyle='dashed')
ax.plot(xs[after_interval], ys[:, 1][after_interval], color='k', zorder=101, linestyle='solid')

# Plot the scatter points with error bars for the second dataset
for _, row in dfo[mask2].iterrows():
    T_MIT = row['T_MIT']
    T_MIT_SE_low = row['T_MIT_SE_lower']
    T_MIT_SE_high = row['T_MIT_SE_upper']
    T_MIT_2SE_low = row['T_MIT_2SE_lower']
    T_MIT_2SE_high = row['T_MIT_2SE_upper']
    
    ax.errorbar(x=row['AgeModel'], y=T_MIT,
                yerr=[[T_MIT_SE_low], [T_MIT_SE_high]],
                color='black', marker='D', markersize=3,
                markeredgewidth=0.5, alpha=1.0, ecolor='black', elinewidth=1, mec='k',zorder=200)

    line = ax.plot([row['AgeModel'], row['AgeModel']], [T_MIT - T_MIT_2SE_low, T_MIT + T_MIT_2SE_high],
                   color='black', linewidth=0.3, alpha=1.0, linestyle='dashed',zorder=198)
    line[0].set_dashes([5, 5])  # Lengths of dashes and spaces

# Define the y-axis tick range and labels
dy_ticks = 20.0
ymin, ymax = 10, 70  # Manually set the y-axis range for temperature
ax.set_yticks([20,40,60])
ax.set_ylim((ymin, ymax))
ax.set_ylabel('Temperature (°C)', fontsize=9)

#BEGIN BIO PANEL
biological_events_panel(axs[7],dfb)
axs[8].get_xaxis().set_visible(True)
axs[8].set_xlabel('Age (Ma)', fontsize=9)
axs[8].tick_params(axis='x', which='both', top=False, bottom=True)
axs[8].set_xticks([580, 570, 560, 550])
axs[8].set_xticklabels([580, 570, 560, 550])
axs[8].tick_params(direction='in')

add_panel_label(axs[0], 'b', position='top-left', label_y=0.80, label_x=0.02, weight='bold',zorder=1000)
add_panel_label(axs[1], 'c', position='top-left', label_y=0.80, label_x=0.02, weight='bold')
add_panel_label(axs[2], 'd', position='top-left', label_y=0.80, label_x=0.02, weight='bold')
add_panel_label(axs[3], 'e', position='top-left', label_y=0.80, label_x=0.02, weight='bold')
add_panel_label(axs[4], 'f', position='top-left', label_y=0.35, label_x=0.02, weight='bold')
add_panel_label(axs[5], 'g', position='top-left', label_y=0.80, label_x=0.02, weight='bold')
add_panel_label(axs[6], 'h', position='top-left', label_y=0.80, label_x=0.02, weight='bold')
add_panel_label(axs[7], 'i', position='top-left', label_y=0.80, label_x=0.02, weight='bold')

for ax in axs[0:8]:
    plots.add_boxes_and_lines(ax)

axs[8].set_xlim((xlo, xhi))
timescale.plot_periods(axs[8], 1.0, 0, 0.12, bCropToPeriod=False, llist=2, bForceAbbrev=False)

pretty_figure(fig, axs)

plt.savefig('figures/shuram_stack_OmanT_dfo_122224.png',dpi=300, bbox_inches="tight")
plt.savefig('figures/shuram_stack_OmanT_dfo_122224.pdf', bbox_inches="tight")
plt.savefig('figures/shuram_stack_OmanT_dfo_122224.svg', format='svg', bbox_inches="tight")
plt.close()
