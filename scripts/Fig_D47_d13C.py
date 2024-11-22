"""
Script for Plotting Δ47 vs. δ13C

This script generates a plot of Δ47 (clumped isotope thermometry) versus δ13C (carbon isotopes)
from geochemical datasets. It allows for formation-specific visualization of clumped isotope data
with distinct formatting for markers and colors.

Author: Kristin Bergmann
Date: November 2024
Contact: kdberg@mit.edu and https://github.com/Kbergmann

Dependencies:
- numpy
- pandas
- matplotlib
- seaborn

Usage:
1. Prepare the input CSV file with Δ47 and δ13C data.
2. Modify file paths and column names as needed to match your dataset.
3. Run the script to generate `.svg` and `.pdf` plots for each specified formation.

License:
This script is open-source and distributed under the MIT License.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse

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
error_color_list = []
black_formations = []

# Define colors for formations
for member in df['Formation']:
    if member == 'Khufai':
        color_list.append('#4EBFED')  # Blue
        error_color_list.append('#4EBFED')
    elif member == 'TopKhufai':
        color_list.append('black')  # Black
        error_color_list.append('black')
    elif member == 'BuahBirba':
        color_list.append('#757678')  # Gray
        error_color_list.append('#757678')
    elif member == 'Shuram':
        color_list.append('#EF5979')  # Red
        error_color_list.append('#EF5979')
    elif member == 'Cement':
        color_list.append('#F78D32')  # Orange
        error_color_list.append('#F78D32')
    elif member == 'CementBurial':
        color_list.append('#B1D56E')  # Green
        error_color_list.append('#B1D56E')
    else:
        color_list.append('black')
        error_color_list.append('black')
        black_formations.append(member)

# Define markers for mineralogy
for member_type in df['Mineralogy']:
    if member_type == 'Calcite':
        marker_list.append('s')  # Square
    elif member_type == 'Dolomite':
        marker_list.append('D')  # Diamond

# Add color, marker, and transparency columns to the dataframe
df['color'] = color_list
df['error_color'] = error_color_list
df['marker'] = marker_list
df['alpha'] = np.ones(len(df))

print(f"Unhandled formations (default black): {black_formations}")

# ===========================
# Plot Function
# ===========================
def carbonT_plot_continuous(ax, df):
    """
    Create a scatter plot of Δ47 vs. δ13C with error bars.
    
    Parameters:
        ax (matplotlib.axes.Axes): The axis to plot on.
        df (pd.DataFrame): The filtered dataframe containing the data.
    """
    for _, row in df.iterrows():
        # Plot the data point with solid error bars for T_MIT_SE
        ax.errorbar(x=row['d13C_VPDB'], y=row['T_MIT'],
                    yerr=[[row['T_MIT_SE_lower']], [row['T_MIT_SE_upper']]],
                    color=row['color'], marker=row['marker'],
                    markersize=6, markeredgewidth=0.5, alpha=row['alpha'],
                    ecolor=row['error_color'], elinewidth=1, mec='k', label=row['Formation'])

        # Add dashed error bars for T_MIT_2SE
        ax.plot([row['d13C_VPDB'], row['d13C_VPDB']],
                [row['T_MIT'] - row['T_MIT_2SE_lower'], row['T_MIT'] + row['T_MIT_2SE_upper']],
                color=row['error_color'], linewidth=0.3, alpha=row['alpha'], linestyle='dashed')

    # Set axis labels and limits
    ax.set_xlabel(r'$\delta^{13}$C (‰, VPDB)', fontsize=8)
    ax.set_ylabel(r'T Δ$_{47}$ (°C)', fontsize=8)
    ax.set_xlim(-20, 15)
    ax.set_ylim(0, 140)

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
    carbonT_plot_continuous(ax, df)
    plt.tight_layout()  # Adjust layout to avoid overlaps
    filename_svg = f'figures/{plot_suffix}_D47_vs_d13C.svg'
    filename_pdf = f'figures/{plot_suffix}_D47_vs_d13C.pdf'
    plt.savefig(filename_svg, format='svg', bbox_inches="tight", transparent=False, pad_inches=0)
    plt.savefig(filename_pdf, format='pdf', bbox_inches="tight", transparent=False, pad_inches=0)
    plt.close()

# Generate plots for each formation group
formations = df['Formation'].unique()
for formation in formations:
    filtered_df = df[df['Formation'] == formation]
    make_plot(filtered_df, plot_suffix=f'{formation}_D47C')

print("Plots created successfully!")