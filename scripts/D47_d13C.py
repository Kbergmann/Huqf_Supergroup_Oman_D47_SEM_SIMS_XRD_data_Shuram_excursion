import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse  # Import the Ellipse class
import numpy as np
import matplotlib as mpl


df = pd.read_csv('data/Clumped_Oman.csv')

# Set Arial as the default font
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42  # Use Type 3 (vector) fonts in PDF
mpl.rcParams['ps.fonttype'] = 42   # Use Type 3 (vector) fonts in PS
mpl.rcParams['text.usetex'] = False

df = df[~df['Location'].isin(['Skip','Australia','Svalbard','Greenland','Standard'])]
df = df[~df['Formation'].isin(['Skip','Miocene','Macdonaldryggen','Unda','Qatar'])]

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
plt.rcParams['axes.xmargin'] = 1
plt.rcParams["mathtext.default"] = 'regular'
plt.rcParams['lines.solid_capstyle'] = 'round'

color_list = []
marker_list = []
error_color_list = []

black_formations = []

for member in df['Formation']:
    if member == 'Khufai':
        color_list.append('#4EBFED')  # White
        error_color_list.append('#4EBFED')  # Error bars black
    elif member == 'TopKhufai':
        color_list.append('black')  # Black
        error_color_list.append('black')
    elif member == 'BuahBirba':
        color_list.append('#757678')  # Grey
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

print("Formations assigned the color black:", black_formations)

for member_type in df['Mineralogy']:
    if member_type == 'Calcite':
        marker_list.append('s')
    elif member_type == 'Dolomite':
        marker_list.append('D')

df['color'] = color_list
df['error_color'] = error_color_list
df['marker'] = marker_list
df['alpha'] = np.ones(len(df))

# Define a function to convert colors to grayscale
def to_grayscale(color):
    # Convert RGB to grayscale using the formula: 0.299 * R + 0.587 * G + 0.114 * B
    r, g, b = matplotlib.colors.to_rgba(color)[:3]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return (gray, gray, gray, 1.0)

# Define a function to convert colors to grayscale
def to_alpha(alpha):
    # Convert RGB to grayscale using the formula: 0.299 * R + 0.587 * G + 0.114 * B
    alpha = 0.3
    return alpha

def carbonT_plot_continuous(ax, df_, colorbar, norm, color_col, alpha):
    for _, row in df_.iterrows():
        T_MIT = row.T_MIT
        d13C_nominal = row.d13C_VPDB
        
        # Error bars for T_MIT_SE
        T_MIT_SE_low = row.T_MIT_SE_lower
        T_MIT_SE_high = row.T_MIT_SE_upper
        # Error bars for T_MIT_2SE
        T_MIT_2SE_low = row.T_MIT_2SE_lower
        T_MIT_2SE_high = row.T_MIT_2SE_upper

        # Plot the data point with solid error bars for T_MIT_SE
        ax.errorbar(x=d13C_nominal, y=T_MIT,
                    yerr=[[T_MIT_SE_low], [T_MIT_SE_high]],
                    color=row.color, marker=row.marker,
                    markersize=6,
                    markeredgewidth=0.5, alpha=row.alpha,
                    ecolor=row.error_color, elinewidth=1, mec='k', label=row.Formation, zorder=row.alpha)

        # Add dashed error bars for T_MIT_2SE with custom dashes
        line = ax.plot([d13C_nominal, d13C_nominal], [T_MIT - T_MIT_2SE_low, T_MIT + T_MIT_2SE_high],
                       color=row.error_color, linewidth=0.3, alpha=row.alpha, linestyle='dashed', zorder=row.alpha)
        line[0].set_dashes([5, 5])  # Lengths of dashes and spaces

    ax.set_xlabel(r'$\delta^{13}$C (‰, VPDB)', fontsize=8)
    ax.set_ylabel(r'T $\Delta_{47}$ ($\degree$C)', fontsize=8)

    ax.set_xlim(-20, 15)
    ax.set_ylim(0, 140)

def make_plot(df, plot_suffix):
    # Set up a single plot
    fig, ax = plt.subplots(figsize=(3.5, 3.5))  # Create a single plot

    # Set x-axis ticks and labels
    x_ticks = [-20, -15, -10, -5, 0, 5, 10, 15]
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(x_ticks))
#    ax.grid(True, color='grey', linewidth=0.3)
    ax.yaxis.set_tick_params(which='both', labelleft=True)

    # Call the specific plot function for the single plot
    carbonT_plot_continuous(ax, df, 0, 0, df['color'], df['alpha'])

    # Adjust spacings if necessary
    fig.tight_layout()  # Adjust layout to make sure everything fits without overlap

    # Save and close figure
    filename = f'figures/Finals_July2024/{plot_suffix}_square_31_nsc_ncrop_D47_d13C_Oman.svg'
    filename2 = f'figures/Finals_July2024/{plot_suffix}_square_31_nsc_ncrop_D47_d13C_Oman.pdf'
    plt.savefig(filename, format='svg', bbox_inches="tight", transparent=False, pad_inches=0)
    plt.savefig(filename2, format='pdf', bbox_inches="tight", transparent=False, pad_inches=0)
    plt.close()

plots = {
    "Khufai": lambda df: df.assign(alpha = df.apply(lambda row: to_alpha(row['alpha']) if row['Formation'] in ['TopKhufai', 'Shuram', 'Cement','CementBurial','BuahBirba'] else row['alpha'], axis=1)),
     "TopKhufai": lambda df: df.assign(alpha = df.apply(lambda row: to_alpha(row['alpha']) if row['Formation'] in ['Shuram', 'BuahBirba', 'Khufai', 'Cement','CementBurial'] else row['alpha'], axis=1)),
     "Shuram": lambda df: df.assign(alpha = df.apply(lambda row: to_alpha(row['alpha']) if row['Formation'] in ['TopKhufai', 'BuahBirba', 'Khufai', 'Cement','CementBurial'] else row['alpha'], axis=1)),
     "BuahBirba": lambda df: df.assign(alpha = df.apply(lambda row: to_alpha(row['alpha']) if row['Formation'] in ['TopKhufai', 'Shuram', 'Khufai', 'Cement','CementBurial'] else row['alpha'], axis=1)),
     "Cement": lambda df: df.assign(alpha = df.apply(lambda row: to_alpha(row['alpha']) if row['Formation'] in ['TopKhufai', 'Shuram', 'BuahBirba', 'Khufai','CementBurial'] else row['alpha'], axis=1)),
     "BurialCement": lambda df: df.assign(alpha = df.apply(lambda row: to_alpha(row['alpha']) if row['Formation'] in ['TopKhufai', 'Shuram', 'BuahBirba', 'Khufai', 'Cement'] else row['alpha'], axis=1))
}

# Loop over the dictionary and create a plot for each color rule
for plot_suffix, color_rule in plots.items():
    # Create a new figure and axes for the plot
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    
    # Apply the color rule to the dataframe
    df_plot = color_rule(df)

    # Sort the df_plot based on alpha values, with alpha = 1 being last
    sorted_df_plot = df_plot.sort_values(by='alpha', ascending=(df_plot['alpha'] != 1).any())
    
    # Make the plot
    make_plot(sorted_df_plot, plot_suffix)