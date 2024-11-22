"""
Script for Plotting δ18O vs. Temperature with δ18Owater Contours

This script generates a plot of δ18O (VPDB, mineral) versus temperature (Δ47-derived),
overlayed with δ18Owater contours. Formation-specific data visualization includes
distinct colors, markers, and error bars.

Author: Kristin Bergmann
Date: November 2024
Contact: kdberg@mit.edu and https://github.com/Kbergmann

Dependencies:
- numpy
- pandas
- matplotlib
- seaborn (optional, for KDE visualization)

Usage:
1. Prepare the input CSV file with δ18O and temperature data.
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
df = pd.read_csv('data/Clumped_Oman.csv')

# Remove unwanted locations and formations
df = df[~df['Location'].isin(['Skip', 'Australia', 'Svalbard', 'Greenland', 'Standard'])]
df = df[~df['Formation'].isin(['Skip', 'Miocene', 'Macdonaldryggen', 'Unda', 'Qatar'])]

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

for member_type in df['Mineralogy']:
    if member_type == 'Calcite':
        marker_list.append('s')  # Square
    elif member_type == 'Dolomite':
        marker_list.append('D')  # Diamond

df['color'] = color_list
df['marker'] = marker_list
df['alpha'] = np.ones(len(df))

# ===========================
# Define Helper Functions
# ===========================
def make_water(D47_T, d18O_VPDB_mineral, mineral='Calcite'):
    """
    Calculate δ18Owater from Δ47 temperature and δ18Omineral.
    """
    thousandlna_A21 = 17.5 * (1e3 * (1 / (D47_T + 273.15))) - 29.1
    a_A21 = np.exp((thousandlna_A21 / 1000))
    eps_A21 = (a_A21 - 1) * 1e3
    d18O_VSMOW = (d18O_VPDB_mineral * 1.03092) + 30.92
    d18Ow = d18O_VSMOW - eps_A21
    return d18Ow

def make_contour_water():
    """
    Generate δ18Owater contours for plotting.
    """
    T_range = np.linspace(0, 140, 120)
    d18O_range = np.linspace(-20, 15, 120)
    d18Ow_contour = np.zeros((len(T_range), len(d18O_range)))

    for i, T in enumerate(T_range):
        for j, d18O in enumerate(d18O_range):
            d18Ow_contour[i, j] = make_water(T, d18O, 'Calcite')

    return T_range, d18O_range, d18Ow_contour

def fmt_mineral(x):
    """
    Format δ18Owater labels for contour plots.
    """
    s = f"{x:.0f}"
    if s == '5':
        s = r'$\delta^{18}$O = ' + s
    return rf"{s} ‰"

def plot_d18O_temperature(ax, df):
    """
    Scatter plot of δ18O vs. Temperature with error bars.
    """
    for _, row in df.iterrows():
        ax.errorbar(
            x=row['d18O_VPDB_mineral'], y=row['T_MIT'],
            xerr=None,
            yerr=[[row['T_MIT_SE_lower']], [row['T_MIT_SE_upper']]],
            color=row['color'], marker=row['marker'],
            markersize=6, markeredgewidth=0.5, alpha=row['alpha'],
            ecolor=row['color'], elinewidth=1, mec='k'
        )

    ax.set_xlabel(r'mineral $\delta^{18}$O (‰, VPDB)', fontsize=8)
    ax.set_ylabel(r'T $\Delta_{47}$ ($\degree$C)', fontsize=8)
    ax.set_xlim(-20, 15)
    ax.set_ylim(0, 140)

def make_plot(df, plot_suffix):
    """
    Generate δ18O vs. Temperature plot with δ18Owater contours.
    """
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # Generate and plot δ18Owater contours
    T_range, d18O_range, d18Ow_contour = make_contour_water()
    dp = ax.contour(d18O_range, T_range, d18Ow_contour, colors='grey', alpha=0.8, linewidths=0.4, levels=[-20, -10, 0, 10, 20])
    ax.clabel(dp, dp.levels, inline=True, inline_spacing=0.1, fmt=fmt_mineral, fontsize=8)

    # Add scatter plot of data
    plot_d18O_temperature(ax, df)

    # Adjust layout and save
    fig.tight_layout()
    filename_svg = f'figures/{plot_suffix}_d18O_vs_Temperature.svg'
    filename_pdf = f'figures/{plot_suffix}_d18O_vs_Temperature.pdf'
    plt.savefig(filename_svg, format='svg', bbox_inches="tight", transparent=False, pad_inches=0)
    plt.savefig(filename_pdf, format='pdf', bbox_inches="tight", transparent=False, pad_inches=0)
    plt.close()

# ===========================
# Generate Plots by Formation
# ===========================
formations = df['Formation'].unique()
for formation in formations:
    filtered_df = df[df['Formation'] == formation]
    make_plot(filtered_df, plot_suffix=f'{formation}_d18O_T')

print("Plots created successfully!")