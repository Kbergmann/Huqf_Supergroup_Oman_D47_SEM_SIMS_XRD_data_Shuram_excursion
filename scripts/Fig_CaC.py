"""
Script for Plotting δ44/40Ca vs. δ13C

This script generates a plot of δ44/40Ca (calcium isotopes) versus δ13C (carbon isotopes)
from geochemical datasets. It is specifically designed for clumped isotope and geochemical
studies, allowing for clear visualization of formation-specific data.

Author: Kristin Bergmann
Date: November 2024
Contact: kdberg@mit.edu and https://github.com/Kbergmann

Dependencies:
- numpy
- pandas
- matplotlib

Usage:
1. Prepare the input CSV file with δ44/40Ca and δ13C data.
2. Modify file paths and column names in the script if necessary.
3. Run the script to generate `.svg` and `.pdf` plots for each formation in the dataset.

License:
This script is open-source and distributed under the MIT License.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
df = pd.read_csv('data/master_spreadsheet_geochemistry_2024_newcomp.csv')

# Filter for samples with δ44/40Ca data
df = df[df['Ca44_40'].notnull()]

# ===========================
# Assign Colors and Markers
# ===========================
color_list = []
marker_list = []
black_formations = []

# Define colors for formations
for member in df['Formation']:
    if member == 'Khufai':
        color_list.append('#4EBFED')  # Blue
    elif member == 'TopKhufai':
        color_list.append('black')  # Black
    elif member == 'BuahBirba':
        color_list.append('#757678')  # Gray
    elif member == 'Shuram':
        color_list.append('#EF5979')  # Red
    elif member == 'Cement':
        color_list.append('#F78D32')  # Orange
    elif member == 'CementBurial':
        color_list.append('#B1D56E')  # Green
    else:
        color_list.append('black')
        black_formations.append(member)
        print(f"Unhandled formation: {member}")

# Define markers for mineralogy
for member_type in df['Min']:
    if member_type == 'CA':
        marker_list.append('s')  # Square
    elif member_type == 'DL':
        marker_list.append('D')  # Diamond

# Add color, marker, and transparency columns to the dataframe
df['color'] = color_list
df['marker'] = marker_list
df['alpha'] = np.ones(len(df))

print(f"Data length after filtering: {len(df)}")

# ===========================
# Plot Function
# ===========================
def calcium_plot_continuous(ax, df):
    """
    Create a scatter plot of δ44/40Ca vs. δ13C with error bars.
    
    Parameters:
        ax (matplotlib.axes.Axes): The axis to plot on.
        df (pd.DataFrame): The filtered dataframe containing the data.
    """
    for _, row in df.iterrows():
        ax.errorbar(x=row['d13C_VPDB'], y=row['Ca44_40'],
                    color=row['color'], marker=row['marker'],
                    markersize=6, markeredgewidth=0.5, alpha=row['alpha'],
                    ecolor=row['color'], elinewidth=0.3, mec='k')

    # Set axis labels and limits
    ax.set_xlabel(r'$\delta^{13}$C (‰, VPDB)', fontsize=8)
    ax.set_ylabel(r'$\delta^{44/40}$Ca (‰, SW)', fontsize=8)
    ax.set_xlim(-20, 15)
    ax.set_ylim(-1.5, 0)

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
    calcium_plot_continuous(ax, df)
    plt.tight_layout()  # Adjust layout to avoid overlaps
    filename_svg = f'figures/{plot_suffix}_d44Ca_vs_d13C.svg'
    filename_pdf = f'figures/{plot_suffix}_d44Ca_vs_d13C.pdf'
    plt.savefig(filename_svg, format='svg', bbox_inches="tight", transparent=False, pad_inches=0)
    plt.savefig(filename_pdf, format='pdf', bbox_inches="tight", transparent=False, pad_inches=0)
    plt.close()

# Generate plots for each formation group
formations = df['Formation'].unique()
for formation in formations:
    filtered_df = df[df['Formation'] == formation]
    make_plot(filtered_df, plot_suffix=f'{formation}_CaC')

print("Plots created successfully!")