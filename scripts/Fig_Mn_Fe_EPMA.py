"""
Script for Plotting Fe vs. Mn with Generation-Specific Visualization

This script generates scatter plots of Fe (ppm) vs. Mn (ppm) from geochemical datasets, with distinct visualization based on generation, formation, and mineral type. It includes options to plot raw data and grouped data with means and standard errors (SE).

Author: Kristin Bergmann  
Date: November 2024  
Contact: kdberg@mit.edu and https://github.com/Kbergmann  

Dependencies:
- numpy
- pandas
- matplotlib

Usage:
1. Prepare the input CSV file with Fe and Mn data.
2. Modify file paths and column names as needed to match your dataset.
3. Run the script to generate `.svg` and `.pdf` plots for each specified formation.

License:
This script is open-source and distributed under the MIT License.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
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
mpl.rcParams['axes.xmargin'] = 0.1
mpl.rcParams['axes.ymargin'] = 0.1
mpl.rcParams['lines.solid_capstyle'] = 'round'

# ===========================
# Load and Preprocess Data
# ===========================
df = pd.read_csv('data/EPMA_final_MIT_Caltech.csv')

# Assign colors, markers, and line widths
formation_colors = {
    'Khufai': '#4EBFED',
    'TopKhufai': 'black',
    'BuahBirba': '#757678',
    'Shuram': '#EF5979',
    'Cement': '#F78D32',
    'CementBurial': '#B1D56E'
}

df['color'] = df['Formation'].map(formation_colors).fillna('black')
df['marker'] = df['Min'].map({'CA': 's', 'DL': 'D'}).fillna('o')
df['line'] = df['Generation'].map({1: 1.5, 2: 0.75, 3: 0.25}).fillna(0)
df['alpha'] = 1.0

# ===========================
# Plotting Function
# ===========================
def plot_fe_mn(ax, df, markersize=5):
    """
    Scatter plot of Fe vs. Mn with markers and error bars.
    """
    for _, row in df.iterrows():
        ax.errorbar(
            x=row['Mn_ppm'], y=row['Fe_ppm'],
            xerr=row.get('Mn_SE', None), yerr=row.get('Fe_SE', None),
            color=row['color'], marker=row['marker'],
            markersize=markersize, markeredgewidth=row['line'], alpha=row['alpha'],
            ecolor=row['color'], elinewidth=0.3, mec='k'
        )
    ax.set_xlabel(r'Mn (ppm)', fontsize=8)
    ax.set_ylabel(r'Fe (ppm)', fontsize=8)
    ax.set_xlim(-100, 7000)
    ax.set_ylim(-100, 7000)

def make_plot(df, plot_suffix, filename_suffix, markersize=5):
    """
    Generate Fe vs. Mn plot for a given dataset.
    """
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    plot_fe_mn(ax, df, markersize=markersize)
    fig.tight_layout()

    filename_svg = f'figures/{plot_suffix}_Fe_vs_Mn_{filename_suffix}.svg'
    filename_pdf = f'figures/{plot_suffix}_Fe_vs_Mn_{filename_suffix}.pdf'
    plt.savefig(filename_svg, format='svg', bbox_inches="tight", transparent=False)
    plt.savefig(filename_pdf, format='pdf', bbox_inches="tight", transparent=False)
    plt.close()

# ===========================
# Grouped Data Processing
# ===========================
# Filter groups with more than one data point
df_filtered = df.groupby(['Generation', 'Min', 'Formation']).filter(lambda x: len(x) > 1)

# Calculate mean and SEM for grouped data
df_grouped = df_filtered.groupby(['Generation', 'Min', 'Formation']).agg({
    'Fe_ppm': ['mean', sem],
    'Mn_ppm': ['mean', sem]
}).reset_index()

# Flatten MultiIndex columns
df_grouped.columns = ['Generation', 'Min', 'Formation', 'Fe_ppm', 'Fe_SE', 'Mn_ppm', 'Mn_SE']

# Merge metadata back into grouped data
metadata_cols = ['Generation', 'Min', 'Formation', 'color', 'marker', 'line', 'alpha']
df_grouped = df_grouped.merge(df[metadata_cols].drop_duplicates(), on=['Generation', 'Min', 'Formation'], how='left')

# ===========================
# Generate Plots
# ===========================
# Define plot suffixes and visibility rules
plots = {
    "all": lambda df: df,
    "Khufai": lambda df: df[df['Formation'] == 'Khufai'],
    "Shuram": lambda df: df[df['Formation'] == 'Shuram'],
    "Cement": lambda df: df[df['Formation'] == 'Cement'],
    "BuahBirba": lambda df: df[df['Formation'] == 'BuahBirba']
}

# Create plots for raw data
for plot_suffix, rule in plots.items():
    filtered_df = rule(df)
    make_plot(filtered_df, plot_suffix, 'all_data', markersize=5)

# Create plots for grouped data (means with SE)
for plot_suffix, rule in plots.items():
    filtered_df_grouped = rule(df_grouped)
    make_plot(filtered_df_grouped, plot_suffix, 'mean_se', markersize=10)

print("Plots created successfully!")