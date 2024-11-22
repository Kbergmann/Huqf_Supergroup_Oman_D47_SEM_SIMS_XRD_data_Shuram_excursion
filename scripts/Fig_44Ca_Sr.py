"""
Script for Plotting δ44Ca vs. Sr/Ca

This script generates a plot of δ44Ca (calcium isotopes) versus Sr/Ca
ratios, allowing formation-specific data visualization with distinct colors and markers.

Author: Kristin Bergmann
Date: November 2024
Contact: kdberg@mit.edu and https://github.com/Kbergmann

Dependencies:
- numpy
- pandas
- matplotlib

Usage:
1. Prepare the input CSV file with δ44Ca and Sr/Ca data.
2. Modify file paths and column names as needed to match your dataset.
3. Run the script to generate `.svg` and `.pdf` plots for each specified formation.

License:
This script is open-source and distributed under the MIT License.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ===========================
# Configure Matplotlib Styles
# ===========================
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42  # Use Type 3 (vector) fonts in PDF
mpl.rcParams['ps.fonttype'] = 42   # Use Type 3 (vector) fonts in PS
mpl.rcParams['svg.fonttype'] = 'none'  # Save text as paths in SVG
mpl.rcParams['text.usetex'] = False
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['lines.solid_capstyle'] = 'round'

# ===========================
# Load and Filter Data
# ===========================
df = pd.read_csv('data/master_spreadsheet_geochemistry_2024_newcomp.csv')

# Select samples with non-null δ44Ca data
df = df[df['Ca44_40'].notnull()]

print(f"Data length after filtering: {len(df)}")

# ===========================
# Assign Colors and Markers
# ===========================
formation_colors = {
    'Khufai': '#4EBFED',
    'TopKhufai': 'black',
    'BuahBirba': '#757678',
    'Shuram': '#EF5979',
    'Cement': '#F78D32',
    'CementBurial': '#B1D56E'
}

color_list = []
marker_list = []

for member in df['Formation']:
    color_list.append(formation_colors.get(member, 'black'))

for member_type in df['Min']:
    if member_type == 'CA':
        marker_list.append('s')  # Square
    elif member_type == 'DL':
        marker_list.append('D')  # Diamond

df['color'] = color_list
df['marker'] = marker_list
df['alpha'] = np.ones(len(df))

# ===========================
# Plot Function
# ===========================
def plot_calcium_sr(ax, df):
    """
    Scatter plot of δ44Ca vs. Sr/Ca with markers and error bars.
    """
    for _, row in df.iterrows():
        ax.errorbar(
            x=row['Ca44_40'], y=row['Sr_Ca_mmol_mol'],
            xerr=None, yerr=None,
            color=row['color'], marker=row['marker'],
            markersize=6, markeredgewidth=0.5, alpha=row['alpha'],
            ecolor=row['color'], elinewidth=0.3, mec='k'
        )

    ax.set_xlabel(r'$\delta^{44/40}$Ca (‰, SW)', fontsize=8)
    ax.set_ylabel(r'Sr/Ca (mmol/mol)', fontsize=8)
    ax.set_xlim(-1.5, 0)
    ax.set_ylim(0, 1.2)

def make_plot(df, plot_suffix):
    """
    Generate δ44Ca vs. Sr/Ca plot for a given dataset.
    """
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    plot_calcium_sr(ax, df)
    fig.tight_layout()

    filename_svg = f'figures/{plot_suffix}_d44Ca_vs_Sr.svg'
    filename_pdf = f'figures/{plot_suffix}_d44Ca_vs_Sr.pdf'
    plt.savefig(filename_svg, format='svg', bbox_inches="tight", transparent=False, pad_inches=0)
    plt.savefig(filename_pdf, format='pdf', bbox_inches="tight", transparent=False, pad_inches=0)
    plt.close()

# ===========================
# Generate Plots by Formation
# ===========================
formations = df['Formation'].unique()
for formation in formations:
    filtered_df = df[df['Formation'] == formation]
    make_plot(filtered_df, plot_suffix=f'{formation}_d44Ca_Sr')

print("Plots created successfully!")