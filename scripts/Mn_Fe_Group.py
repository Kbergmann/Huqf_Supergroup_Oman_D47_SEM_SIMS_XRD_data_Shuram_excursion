import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from scipy.stats import sem
import matplotlib as mpl

df = pd.read_csv('data/figures/EPMA_final_MIT_Caltech.csv')

mpl.rcParams['pdf.fonttype'] = 42  # Use Type 3 (vector) fonts in PDF
mpl.rcParams['ps.fonttype'] = 42   # Use Type 3 (vector) fonts in PS
mpl.rcParams['text.usetex'] = False
mpl.rcParams['svg.fonttype'] = 'none'  # Save text as paths in SVG, preserving font appearance
mpl.rcParams['svg.image_inline'] = False  # Do not inline images, keep them as external references
mpl.rcParams['text.usetex'] = False  # Do not use TeX for text rendering

small_font = 8
plt.rc('axes', labelsize=small_font)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['axes.xmargin'] = 0.1
plt.rcParams['axes.ymargin'] = 0.1
plt.rcParams["mathtext.default"] = 'regular'
plt.rcParams['lines.solid_capstyle'] = 'round'

color_list = []
marker_list = []
line_list = []

black_formations = []

for member in df['Formation']:
    if member == 'Khufai':
        color_list.append('#4EBFED')
    elif member == 'TopKhufai':
        color_list.append('black')
    elif member == 'BuahBirba':
        color_list.append('#757678')
    elif member == 'Shuram':
        color_list.append('#EF5979')
    elif member == 'Cement':
        color_list.append('#F78D32')
    elif member == 'CementBurial':
        color_list.append('#B1D56E')
    else:
        color_list.append('black')
        black_formations.append(member)

for generation in df['Generation']:
    if generation == 1:
        line_list.append(1.5)
    elif generation == 2:
        line_list.append(0.75)
    elif generation == 3:
        line_list.append(0.25)
    elif pd.isna(generation):
        line_list.append(0)
    else:
        line_list.append(0)

for member_type in df['Min']:
    if member_type == 'CA':
        marker_list.append('s')
    elif member_type == 'DL':
        marker_list.append('D')
    else:
        marker_list.append('o')

df['color'] = color_list
df['marker'] = marker_list
df['line'] = line_list
df['alpha'] = np.ones(len(df))

def to_grayscale(color):
    r, g, b = plt.colors.to_rgba(color)[:3]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return (gray, gray, gray, 1.0)

def to_alpha(alpha):
    return 0.15

def metal_plot_continuous(ax, df_, colorbar, norm, color_col, alpha, markersize=5):
    for _, row in df_.iterrows():
        Fe_ppm = row.Fe_ppm
        Mn_ppm = row.Mn_ppm
        Fe_SE = row.Fe_SE if 'Fe_SE' in row and not np.isnan(row.Fe_SE) else None
        Mn_SE = row.Mn_SE if 'Mn_SE' in row and not np.isnan(row.Mn_SE) else None

        ax.errorbar(x=Mn_ppm, y=Fe_ppm,
                    yerr=Fe_SE,
                    xerr=Mn_SE,
                    color=row.color, marker=row.marker,
                    markersize=markersize,
                    markeredgewidth=row.line, alpha=row.alpha,
                    ecolor=row.color, elinewidth=0.3, mec='k')

    ax.set_xlabel(r'Mn (ppm)', fontsize=8)
    ax.set_ylabel(r'Fe (ppm)', fontsize=8)
    ax.set_xlim(-100, 7000)
    ax.set_ylim(-100, 7000)

def make_plot(df, plot_suffix, filename_suffix, markersize):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    metal_plot_continuous(ax, df, 0, 0, df['color'], df['alpha'], markersize=markersize)
    fig.tight_layout()
    filename = f'figures/Finals_July2024/SuppTM_{plot_suffix}_square_11_nsc_ncrop_plot_Fe_Mn_{filename_suffix}.svg'
    filename2 = f'figures/Finals_July2024/SuppTM_{plot_suffix}_square_11_nsc_ncrop_plot_Fe_Mn_{filename_suffix}.pdf'
    plt.savefig(filename, format='svg', bbox_inches="tight", transparent=False)
    plt.savefig(filename2, format='pdf', bbox_inches="tight", transparent=False)
    plt.close()


# Filter out groups with only one data point before calculating standard error
df_filtered = df.groupby(['Generation', 'Min', 'Formation']).filter(lambda x: len(x) > 1)

# Group the filtered data and calculate mean and standard error
df_grouped = df_filtered.groupby(['Generation', 'Min', 'Formation']).agg({
    'Fe_ppm': ['mean', sem],
    'Mn_ppm': ['mean', sem]
}).reset_index()

# Flatten the MultiIndex columns
df_grouped.columns = ['Generation', 'Min', 'Formation', 'Fe_ppm', 'Fe_SE', 'Mn_ppm', 'Mn_SE']

# Merge the grouped data with the original data for color, marker, line width, and alpha values
df_grouped = df_grouped.merge(df[['Generation', 'Min', 'Formation', 'color', 'marker', 'line', 'alpha']].drop_duplicates(), on=['Generation', 'Min', 'Formation'], how='left')

# Define a dictionary that maps color rules to plots
plots = {
    "all": lambda df: df,
    "KhufaiBuahBirba": lambda df: df.assign(alpha=df.apply(lambda row: to_alpha(row['alpha']) if row['Formation'] in ['TopKhufai', 'Shuram', 'Cement','CementBurial'] else row['alpha'], axis=1)),
    "Khufai": lambda df: df.assign(alpha=df.apply(lambda row: to_alpha(row['alpha']) if row['Formation'] in ['TopKhufai', 'Shuram', 'Cement','CementBurial','BuahBirba'] else row['alpha'], axis=1)),
    "TopKhufai": lambda df: df.assign(alpha=df.apply(lambda row: to_alpha(row['alpha']) if row['Formation'] in ['Shuram', 'BuahBirba', 'Khufai', 'Cement','CementBurial'] else row['alpha'], axis=1)),
    "Shuram": lambda df: df.assign(alpha=df.apply(lambda row: to_alpha(row['alpha']) if row['Formation'] in ['TopKhufai', 'BuahBirba', 'Khufai', 'Cement','CementBurial'] else row['alpha'], axis=1)),
    "BuahBirba": lambda df: df.assign(alpha=df.apply(lambda row: to_alpha(row['alpha']) if row['Formation'] in ['TopKhufai', 'Shuram', 'Khufai', 'Cement','CementBurial'] else row['alpha'], axis=1)),
    "Cement": lambda df: df.assign(alpha=df.apply(lambda row: to_alpha(row['alpha']) if row['Formation'] in ['TopKhufai', 'Shuram', 'BuahBirba', 'Khufai','CementBurial'] else row['alpha'], axis=1)),
    "BurialCement": lambda df: df.assign(alpha=df.apply(lambda row: to_alpha(row['alpha']) if row['Formation'] in ['TopKhufai', 'Shuram', 'BuahBirba', 'Khufai', 'Cement'] else row['alpha'], axis=1))
}

# Create plots for the original data
for plot_suffix, color_rule in plots.items():
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    df_plot = color_rule(df)
    sorted_df_plot = df_plot.sort_values(by='alpha', ascending=(df_plot['alpha'] != 1).any())
    make_plot(sorted_df_plot, plot_suffix, 'all_data', markersize=5)

# Create plots for the grouped data (mean with SE)
for plot_suffix, color_rule in plots.items():
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    df_plot = color_rule(df_grouped)
    sorted_df_plot = df_plot.sort_values(by='alpha', ascending=(df_plot['alpha'] != 1).any())
    make_plot(sorted_df_plot, plot_suffix, 'mean_se', markersize=10)