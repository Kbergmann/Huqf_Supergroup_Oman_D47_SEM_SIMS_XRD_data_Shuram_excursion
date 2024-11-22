"""
Script for Plotting Δ47 vs. Δ48

This script generates a plot of Δ47 (clumped isotope thermometry) versus Δ48
from geochemical datasets. It provides formation-specific visualization of clumped isotope data
with distinct formatting for markers and colors.

Author: Kristin Bergmann
Date: November 2024
Contact: kdberg@mit.edu and https://github.com/Kbergmann

Dependencies:
- numpy
- pandas
- matplotlib

Usage:
1. Prepare the input CSV file with Δ47 and Δ48 data.
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
# Input file path
df = pd.read_csv('data/Clumped_Oman.csv')

# Remove unwanted locations and formations
df = df[~df['Location'].isin(['Skip', 'Australia', 'Svalbard', 'Greenland', 'Standard'])]
df = df[~df['Formation'].isin(['Skip', 'Miocene', 'Macdonaldryggen', 'Unda', 'Qatar'])]

print(f"Data length after filtering: {len(df)}")

# ===========================
# Assign Colors and Markers
# ===========================
color_list = []
marker_list = []
black_formations = []

# Define colors for formations
formation_colors = {
    'Khufai': '#4EBFED',
    'TopKhufai': 'black',
    'BuahBirba': '#757678',
    'Shuram': '#EF5979',
    'Cement': '#F78D32',
    'CementBurial': '#B1D56E'
}

for member in df['Formation']:
    color_list.append(formation_colors.get(member, 'black'))
    if member not in formation_colors:
        black_formations.append(member)

# Define markers for mineralogy
for member_type in df['Mineralogy']:
    if member_type == 'Calcite':
        marker_list.append('s')  # Square
    elif member_type == 'Dolomite':
        marker_list.append('D')  # Diamond

# Add color, marker, and transparency columns to the dataframe
df['color'] = color_list
df['marker'] = marker_list
df['alpha'] = np.ones(len(df))

print(f"Unhandled formations (default black): {black_formations}")

# ===========================
# Plot Function
# ===========================
def D47_D48_plot(ax, df):
    """
    Create a scatter plot of Δ47 vs. Δ48 with error bars.
    
    Parameters:
        ax (matplotlib.axes.Axes): The axis to plot on.
        df (pd.DataFrame): The filtered dataframe containing the data.
    """
    for _, row in df.iterrows():
        # Plot the data point with solid error bars for Δ47
        ax.errorbar(x=row['D48'], y=row['D47'],
                    xerr=[[row['D48_SE']], [row['D48_SE']]],
                    yerr=[[row['D47_SE']], [row['D47_SE']]],
                    color=row['color'], marker=row['marker'],
                    markersize=6, markeredgewidth=0.5, alpha=row['alpha'],
                    ecolor=row['color'], elinewidth=1, mec='k', label=row['Formation'])

        # Add dashed error bars for 2SE
        ax.plot([row['D48'] - 2 * row['D48_SE'], row['D48'] + 2 * row['D48_SE']],
                [row['D47'], row['D47']],
                color=row['color'], linewidth=0.3, linestyle='dashed', alpha=row['alpha'])
        ax.plot([row['D48'], row['D48']],
                [row['D47'] - 2 * row['D47_SE'], row['D47'] + 2 * row['D47_SE']],
                color=row['color'], linewidth=0.3, linestyle='dashed', alpha=row['alpha'])

    # Add equilibrium line
    T_x = np.linspace(273, 1273, 100)
    D47_F21 = 1.038 * ((-5.897 / T_x) - (3.521 * (1e3 / (T_x**2))) + (2.391 * (1e7 / (T_x**3))) - (3.541 * (1e9 / (T_x**4)))) + 0.1856
    D48_F21 = 1.028 * ((6.002 / T_x) - (1.299 * (1e4 / (T_x**2))) + (8.996 * (1e6 / (T_x**3))) - (7.423 * (1e8 / (T_x**4)))) + 0.1245
    ax.plot(D48_F21, D47_F21, '-', color='#2A52BE', label='Equilibrium (Fiebig et al., 2021)', linewidth=1)

    # Set axis labels and limits
    ax.set_xlabel(r'$\Delta_{48}$ (‰)', fontsize=8)
    ax.set_ylabel(r'$\Delta_{47}$ (‰)', fontsize=8)
    ax.set_xlim(0.0, 0.6)
    ax.set_ylim(0.35, 0.65)

# ===========================
# Main Plotting Logic
# ===========================
def make_plot(df, plot_suffix):
    """
    Generate a plot for a specific subset of data and save it.
    
    Parameters:
        df (pd.DataFrame): The filtered dataframe containing the data.
        plot_suffix (str): Suffix to append to the output file names.
    """
    fig, ax = plt.subplots(figsize=(3.5, 3.5))  # Square plot
    D47_D48_plot(ax, df)
    plt.tight_layout()  # Adjust layout to avoid overlaps
    filename_svg = f'figures/{plot_suffix}_D47_vs_D48.svg'
    filename_pdf = f'figures/{plot_suffix}_D47_vs_D48.pdf'
    plt.savefig(filename_svg, format='svg', bbox_inches="tight", transparent=False, pad_inches=0)
    plt.savefig(filename_pdf, format='pdf', bbox_inches="tight", transparent=False, pad_inches=0)
    plt.close()

# Generate plots for each formation group
formations = df['Formation'].unique()
for formation in formations:
    filtered_df = df[df['Formation'] == formation]
    make_plot(filtered_df, plot_suffix=f'{formation}_D47D48')

print("Plots created successfully!")