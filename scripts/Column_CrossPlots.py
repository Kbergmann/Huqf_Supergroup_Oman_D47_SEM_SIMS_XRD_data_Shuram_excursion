import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import colors
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from matplotlib.ticker import StrMethodFormatter
from matplotlib.colors import LogNorm, Normalize
from PIL import Image, ImageDraw, ImageFont, ImageStat
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# Set Arial as the default font
mpl.rcParams['font.family'] = 'Arial'
FONT_PATH = '/System/Library/Fonts/Supplemental/arial.ttf'
LETTER_FONT_SIZE = 10  # Fixed font size for labels
mpl.rcParams['pdf.fonttype'] = 42  # Use Type 3 (vector) fonts in PDF
mpl.rcParams['ps.fonttype'] = 42   # Use Type 3 (vector) fonts in PS
mpl.rcParams['text.usetex'] = False
mpl.rcParams['svg.fonttype'] = 'none'  # Save text as paths in SVG, preserving font appearance
mpl.rcParams['svg.image_inline'] = False  # Do not inline images, keep them as external references
mpl.rcParams['text.usetex'] = False  # Do not use TeX for text rendering

fs = 7

def add_panel_label(ax, label, position='top-left', font_size=LETTER_FONT_SIZE,label_y=0.97):
    label_x = 0.1 if position == 'top-left' else 0.95

    label_box = ax.text(
        label_x, label_y, label,
        transform=ax.transAxes,
        fontsize=font_size,
        fontweight='bold',
        va='top', ha='center' if position == 'top-left' else 'right',
        color='white',
        bbox=dict(facecolor='black', alpha=1, edgecolor='none', boxstyle='square,pad=0.1')
    )

    # Adjust label position slightly for top-right to ensure it's inside the axis
    if position == 'top-right':
        label_box.set_position((label_x - 0.05, label_y))

def create_legend1(axL, fontsize=30):
    """
    Creates the first part of the legend in the given axis.
    """
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

    # Legend for marker types
    legend_markers = [
        Line2D([0], [0], marker='s', markerfacecolor='white', markersize=6, label='Calcite', markeredgewidth=0.5, markeredgecolor='black', linestyle='None'),
        Line2D([0], [0], marker='D', markerfacecolor='white', markersize=6, label='Dolomite', markeredgewidth=0.5, markeredgecolor='black', linestyle='None'),
        Line2D([0], [0], marker='o', markerfacecolor='white', markersize=6, label='Deep water', markeredgewidth=1.0, markeredgecolor='black', linestyle='None')
    ]

    # Add a gap between sections by using an empty handle with an empty string label
    gap = [Line2D([0], [0], linestyle='None', label='')]

    # Legend for formation colors
    legend_formations = [
        mpatches.Patch(facecolor='#4EBFED', label='Khufai Fm.', edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='black', label='Top Khufai Fm.', edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#EF5979', label='Shuram Fm.', edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#6D6F72', label='Buah/Birba fms.', edgecolor='black', linewidth=0.5),
    ]

    # Another gap before the next section
    gap2 = [Line2D([0], [0], linestyle='None', label='')]

    # Legend for mineral colors
    legend_minerals = [
        mpatches.Patch(facecolor='#F3A06F', label='Dolomite', edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#E77C75', label='Calcite', edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#FCD489', label='Aragonite', edgecolor='black', linewidth=0.5)
    ]

    # Combine all legend elements with gaps
    handles = legend_formations + gap2 + legend_minerals + gap + legend_markers

# Create the legend in the subplot without bounding box
    leg = axL.legend(
        handles=handles, 
        loc='center left', 
        bbox_to_anchor=(0, 0.5), 
        fontsize=fontsize, 
        frameon=False,
        labelspacing=0.5,  # Increase space between legend entries
        handleheight=2,  # Adjust height of the legend patches
        handlelength=2  # Adjust length of the legend patches
    )
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0.5)
    axL.axis('off')  # Turn off the axis

def create_legend2(axF, fontsize=30):
    """
    Creates the second part of the legend in the given axis.
    """
    # Legend for lithofacies
    legend_lithofacies = [
        mpatches.Patch(facecolor='#A38669', label='Silt',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#FEDF67', label='Sandstone',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#DFE481', label='Peloidal grainstone',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#CDCD77', label='Oncoidal grainstone',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#43A44E', label='Ooidal grainstone',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#D5E500', label='Massive grainstone',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#678A54', label='Hummocky XSt grainstone',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#6CAD1F', label='Trough XSt grainstone',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#C9CFCA', label='Intraclast conglomerate',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#A2A499', label='Edgewise conglomerate',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#CD7768', label='Tepees',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#F24838', label='Pisolite',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#F98C75', label='Irregular laminite',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#F6725D', label='Irregularly laminated stromatolite',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#BC9F99', label='Isopachous stromatolite',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#EABFAD', label='Small laterally linked stromatolite',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#FE9D87', label='Medium columnar stromatolite',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#DF839E', label='Large domal stromatolite',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#C4CAE0', label='Conical stromatolite',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#E3BBC4', label='Crinkly laminite',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#97C0E9', label='Mudstone',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#B1E3F8', label='Silty mudstone',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#59C5E6', label='Fenestral Mudstone',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#73B5CB', label='Wackestone',edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor='#3298B1', label='Wackestone with sparse clasts',edgecolor='black', linewidth=0.5)
    ]

    # Create the legend in the subplot
    axF.legend(handles=legend_lithofacies, loc='lower center', fontsize=fontsize, ncol=5, 
               bbox_to_anchor=(0.5, 0.3), columnspacing=0.5, labelspacing=0.5, frameon=False, handleheight=2, handlelength=2)
    axF.axis('off')  # Turn off the axis

# Load Data
dfc = pd.read_csv('data/figures/fig4/composite_huqf_shuramex_column.csv')
df = pd.read_csv('data/figures/master_spreadsheet_geochemistry_2024_newcomp.csv')
#dfcom = pd.read_csv('data/figures/master_spreadsheet_geochemistry_2024_composite.csv')
dfcom = pd.read_csv('data/figures/master_spreadsheet_geochemistry_2024_newcomp.csv')

color_list = []
marker_list = []
edge_list = []
dunham_color = []
dunham_width = []

for member in df['Formation']:
    if member == 'Khufai':
        color_list.append('#4EBFED')
    elif member == 'Cement':
        color_list.append('#b75336')
    elif member == 'TopKhufai':
        color_list.append('black')
    elif member == 'Shuram':
        color_list.append('#EF5979')
    elif member == 'CementBurial':
        color_list.append('#827098')
    elif member == 'BuahBirba':
        color_list.append('#6D6F72')             
    else:
        color_list.append('black')

for member_type in df['Min']:
    if member_type == 'CA':
        marker_list.append('s')
    elif member_type == 'DL':
        marker_list.append('D')
    else:
        # Set a default marker if neither 'CA' nor 'DL'
        marker_list.append('o')
        
for member_ref in df['Data_Reference']:
    if member_ref == 'Fike_2006':
        edge_list.append('#808080')
    elif member_ref == 'Lee_2015':
        edge_list.append('#000000')
    else:
        edge_list.append('#FFFFFF')
        
df['color'] = color_list
df['marker'] = marker_list
df['edge'] = edge_list

commarker_list = []
comcolor_list = []

for member_type in dfcom['Min']:
    if member_type == 'CA':
        commarker_list.append('s')
    elif member_type == 'DL':
        commarker_list.append('D')
    else:
        # Set a default marker if neither 'CA' nor 'DL'
        commarker_list.append('o')

for member in dfcom['Formation']:
    if member == 'Khufai':
        comcolor_list.append('#4EBFED')
    elif member == 'Cement':
        comcolor_list.append('#b75336')
    elif member == 'TopKhufai':
        comcolor_list.append('black')
    elif member == 'Shuram':
        comcolor_list.append('#EF5979')
    elif member == 'CementBurial':
        comcolor_list.append('#827098')
    elif member == 'BuahBirba':
        comcolor_list.append('#6D6F72')             
    else:
        comcolor_list.append('black')

dfcom['commarker'] = commarker_list
dfcom['comcolor'] = comcolor_list

# Set the line width parameter
z = 0.1

# Define color codes for Dunham classification
for member in dfc['Dunham']:
    if member == 'bo':
        dunham_color.append('#8FB9C1')
    elif member == 'g':
        dunham_color.append('#FFB318')
    elif member == 'p':
        dunham_color.append('#FF8B00')
    elif member == 'w':
        dunham_color.append('#6C94AB')
    elif member == 'r':
        dunham_color.append('#2258A5')
    elif member == 'm':
        dunham_color.append('#4C5A79')
    elif member == 'cov':
        dunham_color.append('#A98E73') 
    elif member == 'ss':
        dunham_color.append('#BCAB88')   
    elif member == 'si':
        dunham_color.append('#856855') 
    elif member == 'cov':
        dunham_color.append('none')  # No color          
    else:
        dunham_color.append('none')
        
dfc['dunham_color'] = dunham_color

for member in dfc['Dunham']:
    if member == 'bo':
        dunham_width.append('0.2')
    elif member == 'g':
        dunham_width.append('0.25')
    elif member == 'p':
        dunham_width.append('0.15')
    elif member == 'w':
        dunham_width.append('0.1')
    elif member == 'r':
        dunham_width.append('0.3')
    elif member == 'm':
        dunham_width.append('0.04')
    elif member == 'cov':
        dunham_width.append('0.0') 
    elif member == 'ss':
        dunham_width.append('0.25')   
    elif member == 'si':
        dunham_width.append('0.05')           
    else:
        dunham_width.append('0')
        
dfc['dunham_width'] = dunham_width
ymax = 1438.13

# List of facies codes
facies_list = [
    'Si', 'Ss', 'Cov','Gpel', 'Gon', 'Goo', 'Ghcs','Gt', 'Gma',
    'Bcrl', 'Bsmstm', 'Bmstm','Bldmstm', 'Bisstm', 'Bconstm','Bilstm', 'Bil',
    'M', 'Ms', 'Mfal','Wic', 'W',
    'Tep', 'Pis', 'Icg','Iewc'
]

# Define explicit color mapping
color_mapping = {
    'Si': '#A38669', 'Ss': '#FEDF67', 'Cov': 'none',
    'Gpel': '#DFE481', 'Gon': '#CDCD77', 'Goo': '#43A44E', 'Ghcs': '#678A54',
    'Gt': '#6CAD1F', 'Gma': '#D5E500','Bcrl': '#E3BBC4','Bsmstm': '#EABFAD','Bmstm': '#FE9D87','Bldmstm': '#DF839E',
    'Bisstm': '#BC9F99','Bconstm': '#C4CAE0','Bilstm': '#F6725D','Bil': '#F98C75',
    'M': '#97C0E9', 'Ms': '#B1E3F8', 'Mfal': '#59C5E6',
    'Wic': '#3298B1', 'W': '#73B5CB',
    'Tep': '#CD7768', 'Pis': '#F24838',
    'Icg': '#C9CFCA', 'Iewc': '#A2A499'
}

def assign_colors(facies_list, color_mapping):
    color_map = {}
    for facies in facies_list:
        # Directly use the explicit color mapping, defaulting to black if not found
        color_map[facies] = color_mapping.get(facies, '#000000')
    return color_map

# Assign colors to facies codes
facies_color_map = assign_colors(facies_list, color_mapping)

# Define top and bottom height for each 'Section' you want to plot
sections_to_plot = {
    'MD6': {'top_height': 42.05, 'bottom_height': 0},
    'MDE': {'top_height': 340.79, 'bottom_height': 42.05},
    'MDE2': {'top_height': 511.76, 'bottom_height': 340.79},
    'MDE3': {'top_height': 578.36, 'bottom_height': 513.16},    
    'MDS': {'top_height': 800.36, 'bottom_height': 578.86},    
    'MD2': {'top_height': 1138.83, 'bottom_height': 806.23},
    'ST1': {'top_height': 1237.63, 'bottom_height': 1147.23},
    'SB1': {'top_height': 1438.13, 'bottom_height': 1238.23}
    }

# Select Published Data
df = df[~df['Group'].isin(['sec1','sec2'])]

color_list = []
marker_list = []
edge_list = []
edge_width = []

# Function to assign colors based on 'Formation' column for the formation labels on the strat column
def get_formation_color(formation):
    if formation == 'Khufai':
        return '#4EBFED'
    elif formation == 'TopKhufai':
        return 'black'
    elif formation == 'Buah':
        return '#6D6F72'
    elif formation == 'Birba':
        return '#CCCBCB'
    elif formation == 'Shuram':
        return '#EF5979'
    else:
        return 'none'

# Function to assign colors based on 'Min' and 'OrigMin' columns
def get_mineral_color(mineral):
    if mineral == 'DL':
        return '#F3A06F'
    elif mineral == 'CA':
        return '#E77C75'
    elif mineral == 'AR':
        return '#FCD489'
    else:
        return 'none'


for member in df['Group']:
    if member == 'pre':
        color_list.append('#4EBFED')
#    elif member == 'sec1':
#        color_list.append('#4c8338')
    elif member == 'onset':
        color_list.append('#1e191a')
    elif member == 'peak':
        color_list.append('#EF5979')
    elif member == 'rec':
        color_list.append('#6D6F72')  #6d6f72
#    elif member == 'sec2':
#        color_list.append('#f68d31')
    elif member == 'post':
        color_list.append('#6D6F72')  #d1d3d4           
    else:
        color_list.append('black')

for member_type in df['Min']:
    if member_type == 'CA':
        marker_list.append('s')
    elif member_type == 'DL':
        marker_list.append('D')
    else:
        # Set a default marker if neither 'CA' nor 'DL'
        marker_list.append('o')
        
edge_width= []
alpha_list = []
edge_list = []
for member_ref in df['Data_Reference']:
    if member_ref == 'Fike_2006':
        edge_list.append('#000000')
        edge_width.append(0.5)
        alpha_list.append(0.5)  # Semi-transparent
    else:
        edge_list.append('#3b5bb5')
        edge_width.append(1.0)
        alpha_list.append(1.0)  # Opaque

df['color'] = color_list
df['marker'] = marker_list
df['edge'] = edge_list
df['edge_width'] = edge_width
df['alpha'] = alpha_list

# Filter out rows with missing or NaN values in x3_mn and y3 columns
filtered_df = df.dropna(subset=['Mn_ppm', 'd13C'])
# Select Published Data
df_lee = df[df['Data_Reference'] == 'Fike_2006']  #Fike_2006  Lee_2015

# Main script adjustments
fig = plt.figure(figsize=(7.25, 8))
gs = fig.add_gridspec(4, 5, height_ratios=[1, 2.4, 0.05,0.5], width_ratios=[1, 1, 1, 1, 1], wspace=0.05, hspace=0.05)

# Create subplots for the main grid
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[1, 1])
ax3 = fig.add_subplot(gs[1, 2])
ax4 = fig.add_subplot(gs[1, 3])
ax5 = fig.add_subplot(gs[1, 4])

# Create subplots for the lower grid
ax1c = fig.add_subplot(gs[0, 1])
ax2c = fig.add_subplot(gs[0, 2])
ax3c = fig.add_subplot(gs[0, 3])
ax4c = fig.add_subplot(gs[0, 4])

# Create subplots for the color bar scales
cbar_ax1 = fig.add_subplot(gs[2, 1])
cbar_ax2 = fig.add_subplot(gs[2, 2])
cbar_ax3 = fig.add_subplot(gs[2, 3])
cbar_ax4 = fig.add_subplot(gs[2, 4])

# Additional subplots for the legends
axL = fig.add_subplot(gs[0, 0])
axF = fig.add_subplot(gs[3, :])

create_legend1(axL, fontsize=4)
create_legend2(axF, fontsize=4)

# Adjusting the vertical positions of the bottom row subplots
vertical_position = 0.02  # Change this value to adjust vertical position
height = 0.35  # Keep this consistent with the desired height

# Adjust the positions of color bar axes if needed
cbar_ax1.set_position([cbar_ax1.get_position().x0, cbar_ax1.get_position().y0 - 0.05, cbar_ax1.get_position().width, cbar_ax1.get_position().height])
cbar_ax2.set_position([cbar_ax2.get_position().x0, cbar_ax2.get_position().y0 - 0.05, cbar_ax2.get_position().width, cbar_ax2.get_position().height])
cbar_ax3.set_position([cbar_ax3.get_position().x0, cbar_ax3.get_position().y0 - 0.05, cbar_ax3.get_position().width, cbar_ax3.get_position().height])
cbar_ax4.set_position([cbar_ax4.get_position().x0, cbar_ax4.get_position().y0 - 0.05, cbar_ax4.get_position().width, cbar_ax4.get_position().height])

# Adjust the position of the axF
axF.set_position([axF.get_position().x0, axF.get_position().y0 - 0.1, axF.get_position().width, axF.get_position().height])

#axF.set_position([axF.get_position().x0, axF.get_position().y0 + 0.05, axF.get_position().width, axF.get_position().height])


# Set consistent aspect ratios and y-limits
for ax in [ax1c, ax2c, ax3c, ax4c]:
    ax.set_box_aspect(1)
    ax.set_ylim(-15, 10.5)

# Set font size for all subplots
for ax in [ax1,ax2, ax3, ax4, ax5, cbar_ax1,cbar_ax2,cbar_ax3,cbar_ax4,ax1c,ax2c,ax3c,ax4c]:
    ax.tick_params(labelsize=fs)
    ax.tick_params(axis='both', direction='in')
    ax.spines['top'].set_linewidth(0.4)
    ax.spines['right'].set_linewidth(0.4)
    ax.spines['bottom'].set_linewidth(0.4)
    ax.spines['left'].set_linewidth(0.4)

# Set font size for all subplots
for ax in [ax1]:
    ax.tick_params(labelsize=fs)
    ax.tick_params(axis='both', direction='in')
    ax.spines['top'].set_linewidth(0.0)
    ax.spines['right'].set_linewidth(0.0)
    ax.spines['bottom'].set_linewidth(0.4)
    ax.spines['left'].set_linewidth(0.4)

ax1.tick_params(axis='y', length=0)  # Hide y-axis ticks
ax1.set_ylim([0, ymax])  # Sets the y-limit from 0 to ymax
ax1.set_facecolor('none')

# Draw stratigraphic column boxes and annotations
ymax = 1438.13
last_height = 0

# Calculate midpoints for annotations
def calculate_midpoints(df, column):
    midpoints = {}
    for unit in df[column].unique():
        unit_rows = df[df[column] == unit]
        if len(unit_rows) > 0:
            first_occurrence = unit_rows['RefSec_CompositeHeight'].min()
            last_occurrence = unit_rows['RefSec_CompositeHeight'].max()
            midpoints[unit] = (first_occurrence + last_occurrence) / 2
    return midpoints

# Calculate midpoints for each column
formation_midpoints = calculate_midpoints(dfc, 'Formation')

# Define width for each column within ax1
formation_col_width = 0.1
min_col_width = 0.05
origmin_col_width = 0.05
facies_col_start = formation_col_width + min_col_width + origmin_col_width

last_height = 0
current_min = None
current_origmin = None
min_start_height = 0
origmin_start_height = 0

for i, row in dfc.iterrows():
    facies = row['Facies']
    facies_color = facies_color_map.get(facies, 'none')  # Assume 'none' if not found
    section_name = row['Section']
    strat_height = row['RefSec_CompositeHeight']
    dunham_width = float(row['dunham_width'])  # Ensure this is a float for calculations

    # Only add a rectangle if the height and width are significant
    if pd.notna(strat_height) and section_name in sections_to_plot and facies_color != 'none' and dunham_width > 0:
        top_height = sections_to_plot[section_name]['top_height']
        bottom_height = sections_to_plot[section_name]['bottom_height']
        
        if bottom_height <= strat_height <= top_height:
            mid_height = (last_height + strat_height) / 2  # Calculate the mean height between boundaries
            
            # Column 1: Formation
            formation_color = get_formation_color(row['Formation'])
            ax1.fill_betweenx([last_height, strat_height], 0, formation_col_width, color=formation_color)
            if (i == 0 or row['Formation'] != dfc.iloc[i-1]['Formation']) and row['Formation'] != 'TopKhufai':
                ax1.annotate(row['Formation'], (formation_col_width / 2, formation_midpoints[row['Formation']]), rotation=90, ha='center', va='center', fontsize=fs, color='black')

            # Column 2: Min
            min_color = get_mineral_color(row['Min'])
            ax1.fill_betweenx([last_height, strat_height], formation_col_width, formation_col_width + min_col_width, color=min_color)
            if current_min is None:
                current_min = row['Min']
                min_start_height = last_height
            if row['Min'] != current_min or i == len(dfc) - 1:
                mid_min_height = (min_start_height + last_height) / 2
#                ax1.annotate(current_min, (formation_col_width + min_col_width / 2, mid_min_height), rotation=90, ha='center', va='center', fontsize=fs, color='black')
                current_min = row['Min']
                min_start_height = last_height

            # Column 3: OrigMin
            origmin_color = get_mineral_color(row['OrigMin'])
            ax1.fill_betweenx([last_height, strat_height], formation_col_width + min_col_width, formation_col_width + min_col_width + origmin_col_width, color=origmin_color)
            if current_origmin is None:
                current_origmin = row['OrigMin']
                origmin_start_height = last_height
            if row['OrigMin'] != current_origmin or i == len(dfc) - 1:
                mid_origmin_height = (origmin_start_height + last_height) / 2
#                ax1.annotate(current_origmin, (formation_col_width + min_col_width + origmin_col_width / 2, mid_origmin_height), rotation=90, ha='center', va='center', fontsize=fs, color='black')
                current_origmin = row['OrigMin']
                origmin_start_height = last_height

            # Column 4: Facies with variable width boxes for Dunham code
            rectangle = plt.Rectangle((facies_col_start, last_height), facies_col_start + dunham_width, strat_height - last_height, facecolor=facies_color, linewidth=0.001, edgecolor='#000000')
            ax1.add_patch(rectangle)

            last_height = strat_height  # Update last_height to current strat

# Set the aspect ratio and limits for the stratigraphic column
ax1.set_ylim([0, ymax])
ax1.set_xlim([0, 1])
ax1.set_xticks([])
ax1.set_yticks([0, 200, 400, 600, 800, 1000, 1200, 1400])
ax1.set_facecolor('none')

# Step 1: Replace strings 'None' and '--' with NaN in the df
df = df.replace(['None', '--'], np.nan)

# Merge entries based on 'Section' and 'Height', combining non-NaN data from duplicate rows
df = df.groupby(['Section', 'Meter']).agg('first').reset_index()

# Step 1: Replace strings 'None' and '--' with NaN in the df
dfcom = dfcom.replace(['None', '--'], np.nan)

# Filter dfcom where 'Use' column equals 1
dfcom = dfcom[dfcom['Use'] == 1]

# Merge entries based on 'Section' and 'Height', combining non-NaN data from duplicate rows
dfcom = dfcom.groupby(['Section', 'Meter']).agg('first').reset_index()

axes = [ax4, ax5]  # List of axes
cmaps = ['cool', 'magma']  # List of colormaps

comaxes = [ax2, ax3]  # List of axes
comcmaps = ['viridis', 'hot']  # List of colormaps

# List of columns to be cleaned
nan_columns = ['TOC', 'Sr_ppm']
x_columns = ['d13Corg', 'Ca44_40']  # List of columns for x-axis
y_column = 'Composite_Height'  # Y-axis is the same for all

com_columns = ['Mn_ppm', 'Mg_ppm']
xcom_columns = ['d13C', 'd18O']  # List of columns for x-axis
ycom_column = 'Composite_Height'  # Y-axis is the same for all

columns = ['Mn_ppm', 'Mg_ppm', 'TOC', 'Sr_ppm']
maps = ['viridis', 'hot', 'cool', 'magma']  # List of colormaps
x_cols = ['d13C', 'd18O', 'd13Corg', 'Ca44_40']  # List of columns for x-axis
y_col = 'Composite_Height'  # Y-axis is the same for all
axl = [ax2, ax3, ax4, ax5]  # List of axes

# Create a dictionary to map each nan_column to its corresponding x_column
nan_to_x_mapping = {
    'Mn_ppm': 'd13C',
    'Mg_ppm': 'd18O',
    'TOC': 'd13Corg',
    'Sr_ppm': 'Ca44_40'
}

for _, row in dfcom.iterrows():
    for x_col, ax in zip(xcom_columns, comaxes):
        scatter_plot = ax.scatter(row[x_col], row[ycom_column], facecolors='grey', edgecolors='black', linewidth=0.2, marker=row['commarker'],  s=5)

for _, row in df.iterrows():
    for x_col, ax in zip(x_columns, axes):
        scatter_plot = ax.scatter(row[x_col], row[ycom_column], facecolors='grey', edgecolors='black', linewidth=0.2, marker=row['marker'],  s=5)

# Debug: print min and max values for Sr
print("Sr_ppm min:", df['Sr_ppm'].min())
print("Sr_ppm max:", df['Sr_ppm'].max())

# Replace NaN in 'Min' with a default category before mapping to markers
df['Min'] = df['Min'].fillna('Default')

# Set markers
df['marker'] = df['Min'].map({'CA': 's', 'DL': 'D', 'Default': 'v'}).fillna('v')

# Set edge width based on 'Data_Reference'
df['edge_width'] = df['Data_Reference'].map({'Fike_2006': 0.4}).fillna(0.7)

# Function to create logarithmic color scale
def create_log_color_scale(data):
    vmin = data.min()
    vmax = data.max()
    return LogNorm(vmin=vmin, vmax=vmax)

# Function to determine the appropriate normalization
def get_norm(data, is_toc):
    if is_toc:
        return create_log_color_scale(data)
    else:
        return Normalize(vmin=data.min(), vmax=data.max())

# Define normalization functions for each element
norm_mn = Normalize(vmin=df['Mn_ppm'].min(), vmax=df['Mn_ppm'].max())
norm_mg = Normalize(vmin=df['Mg_ppm'].min(), vmax=df['Mg_ppm'].max())
norm_toc = create_log_color_scale(df['TOC'])
norm_sr = Normalize(vmin=df['Sr_ppm'].min(), vmax=df['Sr_ppm'].max())

# Plotting scatter plots with colorbar for each nan_column
for nan_col, cmap, ax in zip(columns, maps, axl):
    valid_mask = ~df[nan_col].isna() & ~df[nan_to_x_mapping[nan_col]].isna() & ~df[y_col].isna()
    x_data = df.loc[valid_mask, nan_to_x_mapping[nan_col]]
    y_data = df.loc[valid_mask, y_col]
    c_data = df.loc[valid_mask, nan_col]
    marker_data = df.loc[valid_mask, 'marker']
    edge_width_data = df.loc[valid_mask, 'edge_width']
    alpha_data = df.loc[valid_mask, 'alpha']

    # Check if the nan_column is TOC and create the appropriate color scale
    if nan_col == 'TOC':
        norm = create_log_color_scale(c_data)
        for marker, group in df[valid_mask].groupby(marker_data):
            ax.scatter(group[nan_to_x_mapping[nan_col]], group[y_col], c=group[nan_col], cmap=cmap, norm=norm, 
                       marker=marker, s=20, edgecolors='black', linewidth=group['edge_width'], alpha=group['alpha'])
    else:
        # For other nan_columns, use the default normalization
        norm = Normalize(vmin=c_data.min(), vmax=c_data.max())

        # Check if the nan_column is 'Mn_ppm' or 'Mg_ppm'
        if nan_col in ['Mn_ppm', 'Mg_ppm']:  
            # Exclude 'Fike_2006' and 'Lee_2015' data
            valid_mask_excluded = valid_mask & (df['Data_Reference'] != 'Fike_2006') & (df['Data_Reference'] != 'Lee_2015')
            x_data = x_data[valid_mask_excluded]
            y_data = y_data[valid_mask_excluded]
            c_data = c_data[valid_mask_excluded]
            marker_data = df.loc[valid_mask_excluded, 'marker']
            edge_width_data = df.loc[valid_mask_excluded, 'edge_width']
        else:
            marker_data = df.loc[valid_mask, 'marker']
            edge_width_data = df.loc[valid_mask, 'edge_width']

        # Plot using grouped markers
        for marker, group in df[valid_mask].groupby(marker_data):
            ax.scatter(group[nan_to_x_mapping[nan_col]], group[y_col], c=group[nan_col], cmap=cmap, norm=norm, 
                       marker=marker, s=20, edgecolors='black', linewidth=0.4)

# For Mn_ppm scatter plot
cmap_mn = plt.cm.viridis
sm1 = cm.ScalarMappable(norm=norm_mn, cmap=cmap_mn)
sm1.set_array([])
cbar1 = fig.colorbar(sm1, cax=cbar_ax1, orientation='horizontal') 
cbar1.set_label('Mn (ppm)', fontsize=fs)
cbar1.ax.xaxis.set_ticks_position('bottom')
cbar1.ax.xaxis.set_label_position('bottom')

# For Mg_ppm scatter plot
cmap_mg = plt.cm.hot
sm2 = cm.ScalarMappable(norm=norm_mg, cmap=cmap_mg)
sm2.set_array([])
cbar2 = fig.colorbar(sm2, cax=cbar_ax2, orientation='horizontal')
cbar2.set_label('Mg (ppm)', fontsize=fs)
cbar2.ax.xaxis.set_ticks_position('bottom')
cbar2.ax.xaxis.set_label_position('bottom')

# For TOC scatter plot
cmap_toc = plt.cm.cool
sm3 = cm.ScalarMappable(norm=norm_toc, cmap=cmap_toc)
sm3.set_array([])
cbar3 = fig.colorbar(sm3, cax=cbar_ax3, orientation='horizontal')
cbar3.set_label('TOC (Wt %)', fontsize=fs)
cbar3.ax.xaxis.set_ticks_position('bottom')
cbar3.ax.xaxis.set_label_position('bottom')

# For Sr_ppm color bar
cmap_sr = plt.cm.magma
sm4 = cm.ScalarMappable(norm=norm_sr, cmap=cmap_sr)
sm4.set_array([])
cbar4 = fig.colorbar(sm4, cax=cbar_ax4, orientation='horizontal')
cbar4.set_label('Sr (ppm)', fontsize=fs)
cbar4.ax.xaxis.set_ticks_position('bottom')
cbar4.ax.xaxis.set_label_position('bottom')

# Set font size for tick labels and axis labels
point = 10

for _, row in dfcom.iterrows():
    ax2c.scatter(row['d18O'], row['d13C'], color=row['comcolor'], marker=row['commarker'], s=10, edgecolor='k', linewidth=0.4,  label=row['Min'])
    ax1c.scatter(row['Mn_ppm'], row['d13C'], color=row['comcolor'], marker=row['commarker'], s=10, edgecolor='k', linewidth=0.4, label=row['Min'])

for _, row in df.iterrows():
    if not pd.isna(row['TOC']):
        ax3c.scatter(row['d13Corg'], row['d13C'], color=row['color'], marker=row['marker'], s=10, edgecolor='k', linewidth=row['edge_width'], alpha=row['alpha'])
    else:
        ax3c.scatter(row['d13Corg'], row['d13C'], color=row['color'], marker=row['marker'], s=10, edgecolor='k', linewidth=row['edge_width'])

for _, row in df.iterrows():
    ax4c.scatter(row['Ca44_40'], row['d13C'], color=row['color'], marker=row['marker'], s=10, linewidth=0.4, edgecolor='k') 

ax1c.set_xlabel('Mn (ppm)', fontsize=fs, color='black')
ax1c.set_ylabel('$\delta^{13}$C (‰, VPDB)', fontsize=fs)
ax1c.xaxis.set_ticks_position('top')
ax1c.xaxis.set_label_position('top')

ax2.set_xlim(-15, 10)
ax2.set_xticks([-10, -5, 0, 5])
ax2.set_yticks([])
ax2.set_ylim(0, ymax)

ax3.set_xlim(-12, 7)
ax3.set_ylim(0, ymax)
ax3.set_xticks([-10, -5, 0, 5])
ax3.set_yticks([])

ax4.set_yticks([])
ax4.set_ylim(0, ymax)
ax4.set_xlim(-40,-20)
ax4.set_xticks([-40, -35, -30, -25])

ax5.set_xlim(-2, 0)
ax5.set_ylim(0, ymax)
ax5.set_xticks([-1.5, -1, -0.5])
ax5.set_yticks([])

ax2c.set_xlim(-12, 7)
ax2c.set_xticks([-10, -5, 0, 5])
ax2c.set_xlabel('$\delta^{18}$O (‰, VPDB)', fontsize=fs)
ax2c.xaxis.set_ticks_position('top')
ax2c.xaxis.set_label_position('top')

ax3c.set_xlim(-40,-20)
ax3c.set_xticks([-40, -35, -30, -25])
ax3c.set_xlabel('$\delta^{13}$C$_{org}$ (‰, VPDB)', fontsize=fs)
ax3c.xaxis.set_ticks_position('top')
ax3c.xaxis.set_label_position('top')

ax4c.set_xlim(-2, 0)
ax4c.set_xticks([-1.5, -1, -0.5])
ax4c.set_xlabel('$\delta^{44/40}$Ca (‰, SW)', fontsize=fs)
ax4c.xaxis.set_ticks_position('top')
ax4c.xaxis.set_label_position('top')

ax1c.set_yticks([-10, -5, 0, 5, 10])
ax1c.set_xlim(0,1750)
ax1c.set_ylabel('$\delta^{13}$C (‰, VPDB)', fontsize=fs)

ax2c.set_xticks([-10, -5, 0, 5, 10])
ax2c.set_yticks([])
#ax2c.set_xlabel('$\delta^{18}$O (‰, VPDB)', fontsize=fs)

ax3c.set_yticks([])
ax3c.set_xlim(-42,-24)
ax3c.set_xticks([-40, -35, -30, -25])
#ax3c.set_xlabel('$\delta^{13}$C$_{org}$ (‰, VPDB)', fontsize=fs)

ax4c.set_xlim(-2, 0)
ax4c.set_xticks([-1.5, -1, -0.5])
ax4c.set_yticks([])
#ax4c.set_xlabel('$\delta^{44/40}$Ca (‰, SW)', fontsize=fs)

# Add panel labels
labels_row3 = ['F', 'G', 'H', 'I']
for i, (ax, label) in enumerate(zip([ax2, ax3, ax4, ax5], labels_row3)):
    add_panel_label(ax, label, position='top-left',label_y=0.98)

labels_row4 = ['A', 'B', 'C', 'D']
for i, (ax, label) in enumerate(zip([ax1c, ax2c, ax3c, ax4c], labels_row4)):
    add_panel_label(ax, label, position='top-left', label_y=0.95)

# Apply the labels to the subplots with mixed positions
add_panel_label(ax1, 'E', position='top-right',label_y=0.98)

# Add top-right labels with white background boxes
ax2.annotate('$\delta^{13}$C (‰, VPDB)', (0.95, 0.97), xycoords='axes fraction', ha='right', va='center', fontsize=fs, color='black',
             bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'))
ax3.annotate('$\delta^{18}$O (‰, VPDB)', (0.95, 0.97), xycoords='axes fraction', ha='right', va='center', fontsize=fs, color='black',
             bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'))
ax4.annotate('$\delta^{13}$C$_{org}$ (‰, VPDB)', (0.95, 0.97), xycoords='axes fraction', ha='right', va='center', fontsize=fs, color='black',
             bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'))
ax5.annotate('$\delta^{44/40}$Ca (‰, SW)', (0.95, 0.97), xycoords='axes fraction', ha='right', va='center', fontsize=fs, color='black',
             bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'))


fig.tight_layout(rect=[0, 0.1, 1, 0.95])

# Show the plot
#plt.savefig('figures/Finals_July2024/column_strat_geochem_v3ax1_formation_top.png', dpi=600, bbox_inches='tight',transparent=True)
plt.savefig('figures/Finals_July2024/column_strat_geochem_v3ax1_formation_top_symbols.svg', format='svg', bbox_inches='tight',transparent=True)
plt.savefig('figures/Finals_July2024/column_strat_geochem_v3ax1_formation_top_symbols.pdf', bbox_inches='tight',transparent=True)
plt.close()

#plt.savefig('figures/Finals_April2024/diagenetic_crossplots_newcolors.png', dpi=600, bbox_inches="tight",transparent=True)
#plt.close()


