import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Font and PDF settings
FONT_PATH = '/System/Library/Fonts/Supplemental/arial.ttf'
arial_font = fm.FontProperties(fname=FONT_PATH)
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42  # Embed fonts as vector
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['text.usetex'] = False

small_font = 8
plt.rc('axes', labelsize=small_font)
plt.rc('xtick', labelsize=small_font, direction='in')
plt.rc('ytick', labelsize=small_font, direction='in')

# Load data
df = pd.read_csv('data/figures/master_spreadsheet_geochemistry_2024_newcomp.csv')

# Normalize TOC data where 10% TOC is considered as 100% remaining (f = 1)
max_TOC = 12  # Maximum TOC assumed as 10%
df['f_remaining'] = df['TOC'] / max_TOC

# Initial conditions for modeling
delta13C_initial = -35  # permil, assumed initial δ13Corg
epsilon = 1.5  # permil, fractionation effect

# Generate f values from 1 (100% TOC) to nearly 0 for the model
f_values = np.linspace(1, 0.01, 100)  # Adjust the endpoint close to zero as needed

# Calculate the delta13C_residual across these f values
delta13C_residual = delta13C_initial - epsilon * np.log(f_values)

# Select Published Data
#df = df[df['Published'] == 'Yes']
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

# Data filtering
df = df[~df['Group'].isin(['sec1', 'sec2'])]

# Normalization
max_TOC = 12
df['f_remaining'] = df['TOC'] / max_TOC

# Colormap for d13C
norm = Normalize(vmin=df['d13C'].min(), vmax=df['d13C'].max())
cmap = cm.get_cmap('cool')

# Marker style assignment
df['Min'] = df['Min'].fillna('Default')
df['marker'] = df['Min'].map({'CA': 's', 'DL': 'D', 'Default': 'v'}).fillna('v')
df['edge_width'] = df['Data_Reference'].map({'Fike_2006': 0.4}).fillna(0.7)
df['alpha'] = df['Data_Reference'].map({'Fike_2006': 0.5}).fillna(1.0)

# Plot setup
fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))

# Subplot 1: εTOC vs TOC
for _, row in df.iterrows():
    color = cmap(norm(row['d13C']))
    axes[0].scatter(row['d13C'] - row['d13Corg'], row['TOC'],
                    color=color, marker=row['marker'], s=10,
                    edgecolor='k', linewidth=row['edge_width'], alpha=row['alpha'])

axes[0].set_xlabel(r'$\mathrm{\epsilon_{TOC}}$ (‰)', fontproperties=arial_font)
axes[0].set_ylabel(r'TOC (wt %)', fontproperties=arial_font)
axes[0].set_yscale('log')
axes[0].set_xlim(10, 40)
axes[0].set_xticks([15, 20, 25, 30, 35])
axes[0].set_xticklabels([15, 20, 25, 30, 35], fontproperties=arial_font, fontsize=small_font)
axes[0].tick_params(axis='y', labelsize=small_font)

# Subplot 2: δ13Corg vs TOC
for _, row in df.iterrows():
    axes[1].scatter(row['d13Corg'], row['TOC'],
                    color=row['color'], marker=row['marker'], s=10,
                    edgecolor='k', linewidth=row['edge_width'], alpha=row['alpha'])

axes[1].set_xlabel(r'$\mathrm{\delta^{13}C_{org}}$ (‰, VPDB)', fontproperties=arial_font)
axes[1].set_xlim(-40, -20)
axes[1].set_xticks([-40, -35, -30, -25])
axes[1].set_xticklabels(['–40', '–35', '–30', '–25'], fontproperties=arial_font, fontsize=small_font)
axes[1].set_yscale('log')
axes[1].set_ylabel(r'TOC (wt %)', fontproperties=arial_font, labelpad=2)
axes[1].tick_params(axis='y', labelsize=small_font)

# Subplot 3: δ13Corg vs TOC remaining (%)
max_TOC = 10
df['TOC_pct'] = (df['TOC'] / max_TOC) * 100
for _, row in df.iterrows():
    axes[2].scatter(row['d13Corg'], row['TOC_pct'],
                    color=row['color'], marker=row['marker'], s=10,
                    edgecolor='k', linewidth=row['edge_width'], alpha=row['alpha'])

# Rayleigh curve
initial_d13C = -35
epsilon = 1.5
f_values = np.linspace(1, 0.001, 100)
d13C_values = initial_d13C - epsilon * np.log(f_values)
TOC_pct_values = f_values * 100
axes[2].plot(d13C_values, TOC_pct_values, 'r--')

axes[2].set_xlabel(r'$\mathrm{\delta^{13}C_{org}}$ (‰, VPDB)', fontproperties=arial_font)
axes[2].set_ylabel(r'% TOC remaining (equiv. to $f$)', fontproperties=arial_font, labelpad=2)
axes[2].set_xlim(-40, -20)
axes[2].set_ylim(0, 40)
axes[2].set_xticks([-40, -35, -30, -25])
axes[2].set_xticklabels(['–40', '–35', '–30', '–25'], fontproperties=arial_font, fontsize=small_font)
axes[2].tick_params(axis='y', labelsize=small_font)

# Consistent tick and spine styling
for ax in axes:
    ax.tick_params(axis='both', direction='in', width=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)

fig.tight_layout()
plt.savefig('figures/Finals_July2024/d13C_ralyeigh_3plot_colorbar_symbols.png', format='png', bbox_inches='tight', transparent=True)
plt.savefig('figures/Finals_July2024/d13C_ralyeigh_3plot_colorbar_symbols.pdf', bbox_inches='tight', transparent=True)
plt.close()

# Sensitivity parameters
max_TOC_values = [25, 10, 5]
delta13C_initial_values = [-35, -33, -30]
epsilon_values = [0.5, 1.5, 3, 4.5]

# Setup figure
fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), sharey=True)
small_font = 10

f_values = np.linspace(1, 0.001, 100)

# Panel 0: Sensitivity to max TOC (adjusted magma colormap to avoid bright yellow)
cmap0 = cm.get_cmap('magma', len(max_TOC_values) + 2)
for i, max_TOC in enumerate(max_TOC_values):
    TOC_pct_values = f_values * max_TOC
    d13C_values = -33 - 1.5 * np.log(f_values)
    axes[0].plot(d13C_values, TOC_pct_values, color=cmap0(i), label=f'Max TOC={max_TOC}%')

# Panel 1: Sensitivity to Initial δ13C
cmap1 = cm.get_cmap('cool', len(delta13C_initial_values))
for i, d13C_init in enumerate(delta13C_initial_values):
    TOC_pct_values = f_values * 10
    d13C_values = d13C_init - 1.5 * np.log(f_values)
    axes[1].plot(d13C_values, TOC_pct_values, color=cmap1(i), label = f'$\mathrm{{\delta^{{13}}C_{{org, initial}}}}$={str(d13C_init).replace("-", "–")}‰')

# Panel 2: Sensitivity to epsilon (with data overlay)
cmap2 = cm.get_cmap('viridis', len(epsilon_values))
for i, eps in enumerate(epsilon_values):
    TOC_pct_values = f_values * 10
    d13C_values = -33 - eps * np.log(f_values)
    axes[2].plot(d13C_values, TOC_pct_values, color=cmap2(i), label=f'$\mathrm{{\epsilon}}$={eps}‰')

# Overlay data points
max_TOC_data = 10  # defined separately as per user's instruction
df['TOC_pct'] = (df['TOC'] / max_TOC_data) * 100
for ax in axes:
    for index, row in df.iterrows():
        ax.scatter(row['d13Corg'], row['TOC_pct'], color=row['color'], marker=row['marker'],
                   s=10, edgecolor='k', linewidth=row['edge_width'], alpha=row['alpha'], zorder=0)

axes[0].set_ylabel(r'% TOC remaining (equiv. to $f$)', fontsize=small_font, fontproperties=arial_font, labelpad=2)

# Formatting axes
for ax in axes:
    ax.set_xlim(-40, -20)
    ax.set_ylim(0, 10)
    ax.set_yticks([0,2.5,5,7.5,10])
    ax.set_yticklabels([0,2.5,5,7.5,10],  fontproperties=arial_font, fontsize=small_font)
    ax.set_xlabel(r'$\mathrm{\delta^{13}C_{org}}$ (‰, VPDB)', fontsize=small_font, fontproperties=arial_font)
    ax.set_xticks([-40, -35, -30, -25])
    ax.set_xticklabels(['–40', '–35', '–30', '–25'], fontproperties=arial_font, fontsize=small_font)
    ax.tick_params(axis='both', direction='in', width=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
    ax.legend(fontsize=8, prop=arial_font)

plt.tight_layout()

# Save figures
fig.savefig('figures/Finals_July2024/d13C_rayleigh_3plot_sensitivity.png', bbox_inches='tight', transparent=True)
fig.savefig('figures/Finals_July2024/d13C_rayleigh_3plot_sensitivity.pdf', bbox_inches='tight', transparent=True)
plt.close(fig)