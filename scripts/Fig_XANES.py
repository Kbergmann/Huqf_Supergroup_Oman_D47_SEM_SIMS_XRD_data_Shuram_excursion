"""
XANES Data Visualization Script

Processes and visualizes Mn and Fe XANES absorbance spectra from a .mat file.
Outputs are saved as figures in PNG format.

Author: Kristin Bergmann  
Date: November 2024  
Contact: kdberg@mit.edu

License:
Distributed under the MIT License.
"""

import matplotlib.pyplot as plt
import scipy.io
import pandas as pd

# ========== LOAD DATA ==========
# Define file path for the .mat file
mat_file_path = '/data/figures/fig9/xanes.mat'
mat = scipy.io.loadmat(mat_file_path)

# Extract data to a pandas DataFrame (replace 'data_key' with the correct key in your .mat file)
# If your data is stored differently, adjust this step accordingly.
data_key = 'data_key'  # Replace with the correct key
if data_key in mat:
    df = pd.DataFrame(mat[data_key])
    df.to_csv('xanes_all.csv', index=False)  # Save as CSV for easy reference
else:
    raise KeyError(f"Key '{data_key}' not found in the .mat file.")

# ========== PLOT CONFIGURATION ==========
# Define colors and styles
colors = ['#f27121', '#2e3191', '#bc519d', '#00ac4d', '#00adee', '#231f20', '#818385', '#989a9d']
linestyles = ['-', '-', '-', '-', '-', '-', '-', '--']

# Labels for Mn and Fe spectra
mn_labels = [
    'MD 668m', 'MD 510m', 'KD 519m', 'MD 400m', 'MD 360m',
    'Manganoan calcite', 'Rhodochrosite', 'MnO$_2$'
]
fe_labels = [
    'MD 668m', 'MD 510m', 'KD 519m', 'MD 400m', 'MD 360m',
    'Siderite', 'Biotite', 'Magnetite', 'Hematite'
]

# Function to configure axes
def configure_axis(ax, xlim, ylim, xlabel, ylabel, xticks, legend_labels):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(xticks)
    ax.legend(legend_labels, fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)

# ========== PLOT FIGURE 1: MANGANESE ==========
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

# Replace these with correct keys from the .mat file
mn_spectra_keys = [
    'MDS_109_6_Mn', 'MDE2_176_3_Mn', 'WS7_7_Mn',
    'MDE2_85_Mn', 'MDS_S2_Mn', 'Manganoan', 'Rhodochrosite', 'MnO2'
]

for spectrum, color, linestyle in zip(mn_spectra_keys, colors, linestyles):
    data = mat[spectrum]
    ax1.plot(data[:, 0], data[:, 1], color=color, linestyle=linestyle)

configure_axis(
    ax=ax1,
    xlim=[6525, 6600],
    ylim=[0, 2.5],
    xlabel='Energy (eV)',
    ylabel='Absorbance',
    xticks=[6530, 6550, 6570, 6590],
    legend_labels=mn_labels
)

# ========== PLOT FIGURE 2: IRON ==========
# Replace these with correct keys from the .mat file
fe_spectra_keys = [
    'MDS_109_6_Fe', 'MDE2_176_3_Fe', 'WS7_7_Fe',
    'MDE2_85_Fe', 'MDS_S2_Fe', 'Biotite', 'Siderite', 'Magnetite', 'Hematite'
]

for spectrum, color, linestyle in zip(fe_spectra_keys, colors + ['#231f20'], linestyles):
    data = mat[spectrum]
    ax2.plot(data[:, 0], data[:, 1], color=color, linestyle=linestyle)

configure_axis(
    ax=ax2,
    xlim=[7100, 7150],
    ylim=[0, 2],
    xlabel='Energy (eV)',
    ylabel='Absorbance',
    xticks=[7100, 7110, 7120, 7130, 7140, 7150],
    legend_labels=fe_labels
)

# ========== SAVE AND CLOSE ==========
fig.suptitle('XANES Absorbance Spectra for Mn and Fe', fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('figures/XANES_Absorbance_Spectra.png', dpi=300, bbox_inches="tight")
plt.savefig('figures/XANES_Absorbance_Spectra.svg', format='svg', bbox_inches="tight")
plt.close()