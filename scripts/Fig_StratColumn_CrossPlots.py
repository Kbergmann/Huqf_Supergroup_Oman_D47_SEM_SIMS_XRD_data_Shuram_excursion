"""
Composite Geochemical and Stratigraphic Plot Generator

This script generates stratigraphic column and geochemical scatter plots with legends, panel labels, and color mappings. 
It includes custom configurations for facies, formations, minerals, and Dunham classifications.

Author: Kristin Bergmann  
Date: November 2024  
Contact: kdberg@mit.edu and https://github.com/Kbergmann  

Dependencies:
- numpy
- pandas
- matplotlib

Usage:
1. Prepare input CSV files for stratigraphic, geochemical, and composite datasets.
2. Modify file paths and parameters to suit your data.
3. Run the script to generate `.svg` and `.pdf` outputs.

License:
This script is open-source and distributed under the MIT License.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable

# Configure Matplotlib defaults
mpl.rcParams.update({
    'font.family': 'Arial',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'text.usetex': False,
    'axes.labelsize': 8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'lines.solid_capstyle': 'round'
})

# Font size for plots
fs = 7

# ========== HELPER FUNCTIONS ==========
def add_panel_label(ax, label, position='top-left', label_y=0.95, font_size=10):
    """Adds panel labels to axes."""
    label_x = 0.1 if position == 'top-left' else 0.95
    ax.text(
        label_x, label_y, label,
        transform=ax.transAxes,
        fontsize=font_size,
        fontweight='bold',
        va='top',
        ha='center' if position == 'top-left' else 'right',
        bbox=dict(facecolor='black', alpha=1, edgecolor='none', boxstyle='square,pad=0.1'),
        color='white'
    )

def create_legend(ax, legend_elements, fontsize=4, ncol=5):
    """Creates a legend with given elements."""
    ax.legend(handles=legend_elements, loc='lower center', fontsize=fontsize, ncol=ncol, 
              bbox_to_anchor=(0.5, 0.3), frameon=False)
    ax.axis('off')

def assign_colors(df, column, color_mapping, default_color='black'):
    """Assigns colors to a DataFrame based on a mapping."""
    return df[column].map(color_mapping).fillna(default_color)

def create_colorbars(fig, cbar_axs, norms, cmaps, labels, fontsize=7):
    """Adds colorbars to axes."""
    for cbar_ax, norm, cmap, label in zip(cbar_axs, norms, cmaps, labels):
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(label, fontsize=fontsize)
        cbar.ax.xaxis.set_label_position('bottom')

# ========== LOAD DATA ==========
dfc = pd.read_csv('data/composite_huqf_shuramex_column.csv')
df = pd.read_csv('data/master_spreadsheet_geochemistry_2024_newcomp.csv')
dfcom = pd.read_csv('data/master_spreadsheet_geochemistry_2024_newcomp.csv')

# ========== DATA PROCESSING ==========
# Assign colors for formations
formation_colors = {
    'Khufai': '#4EBFED',
    'Cement': '#b75336',
    'TopKhufai': 'black',
    'Shuram': '#EF5979',
    'CementBurial': '#827098',
    'BuahBirba': '#6D6F72'
}
df['color'] = assign_colors(df, 'Formation', formation_colors)

# Assign marker styles for minerals
mineral_markers = {'CA': 'o', 'DL': 's'}
df['marker'] = df['Min'].map(mineral_markers).fillna('o')

# Process facies for stratigraphic column
facies_colors = {
    'Si': '#A38669', 'Ss': '#FEDF67', 'Cov': 'none',
    'Gpel': '#DFE481', 'Gon': '#CDCD77', 'Goo': '#43A44E'
}
dfc['facies_color'] = assign_colors(dfc, 'Facies', facies_colors)

# Filter data for plotting
df = df[df['Group'] != 'sec1']

# ========== PLOT SETUP ==========
fig = plt.figure(figsize=(7.25, 8))
gs = GridSpec(4, 5, height_ratios=[1, 2.4, 0.05, 0.5], width_ratios=[1, 1, 1, 1, 1], wspace=0.05, hspace=0.05)

# Define axes
ax1 = fig.add_subplot(gs[1, 0])  # Stratigraphic column
ax2, ax3, ax4, ax5 = [fig.add_subplot(gs[1, i]) for i in range(1, 5)]  # Scatter plots
cbar_axs = [fig.add_subplot(gs[2, i]) for i in range(1, 5)]  # Colorbars
axL = fig.add_subplot(gs[0, 0])  # Legend 1
axF = fig.add_subplot(gs[3, :])  # Legend 2

# Create legends
legend_elements = [Patch(facecolor=color, label=formation) for formation, color in formation_colors.items()]
create_legend(axL, legend_elements, fontsize=4)

# ========== PLOTTING ==========
# Stratigraphic column
for i, row in dfc.iterrows():
    ax1.fill_betweenx(
        [row['RefSec_CompositeHeight'], row['RefSec_CompositeHeight'] + 1],
        0, 1, color=row['facies_color'], linewidth=0.1
    )

# Geochemical scatter plots
for ax, x_col, label in zip([ax2, ax3, ax4, ax5], ['d13C', 'd18O', 'TOC', 'Sr_ppm'], 
                             ['$\delta^{13}$C (‰, VPDB)', '$\delta^{18}$O (‰, VPDB)', 'TOC (Wt %)', 'Sr (ppm)']):
    ax.scatter(df[x_col], df['Composite_Height'], color=df['color'], marker=df['marker'], s=10, alpha=0.8)
    ax.set_xlabel(label, fontsize=fs)
    ax.set_ylim(0, 1400)

# Add colorbars
norms = [Normalize(vmin=df[col].min(), vmax=df[col].max()) for col in ['Mn_ppm', 'Mg_ppm', 'TOC', 'Sr_ppm']]
cmaps = ['viridis', 'hot', 'cool', 'magma']
labels = ['Mn (ppm)', 'Mg (ppm)', 'TOC (Wt %)', 'Sr (ppm)']
create_colorbars(fig, cbar_axs, norms, cmaps, labels)

# Panel labels
panel_labels = ['A', 'B', 'C', 'D', 'E']
for ax, label in zip([ax1, ax2, ax3, ax4, ax5], panel_labels):
    add_panel_label(ax, label)

# ========== SAVE FIGURE ==========
plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.savefig('figures/composite_geochem_plot.svg', format='svg', bbox_inches='tight')
plt.savefig('figures/composite_geochem_plot.pdf', bbox_inches='tight')
plt.close()