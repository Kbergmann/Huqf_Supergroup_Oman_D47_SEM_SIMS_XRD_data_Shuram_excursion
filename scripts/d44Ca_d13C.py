import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse  # Import the Ellipse class
import numpy as np
import matplotlib as mpl


df = pd.read_csv('data/Clumped_Neoproterozoic.csv')

# Set Arial as the default font
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42  # Use Type 3 (vector) fonts in PDF
mpl.rcParams['ps.fonttype'] = 42   # Use Type 3 (vector) fonts in PS
mpl.rcParams['text.usetex'] = False

# read data
df = pd.read_csv('data/figures/master_spreadsheet_geochemistry_2024.csv')
#Select Ca Samples
df = df[df['Ca44_40'].notnull()]

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
black_formations = []

expected_formations = ['Khufai', 'TopKhufai', 'BuahBirba', 'Shuram', 'Cement', 'CementBurial']

for member in df['Formation']:
    if member == 'Khufai':
        color_list.append('#4EBFED')  # Purple #800080
    elif member == 'TopKhufai':
        color_list.append('black')  # Dodger Blue  #4B0082
    elif member == 'BuahBirba':
        color_list.append('#757678')  # Pale Pink  #FF00FF Magenta #FFC0CB
    elif member == 'Shuram':
        color_list.append('#EF5979')  # Orchid (Lighter Purple)
    elif member == 'Cement':
        color_list.append('#F78D32')  # Orange
    elif member == 'CementBurial':
        color_list.append('#B1D56E')  # Green
    else:
        color_list.append('black')
        black_formations.append(member)
        print(f"Unhandled formation: {member}")

# Display the formations assigned to black
print("Formations assigned the color black:", black_formations)

for member_type in df['Min']:
    if member_type == 'CA':
        marker_list.append('s')
    elif member_type == 'DL':
        marker_list.append('D')

df['color'] = color_list
df['marker'] = marker_list
df['alpha'] = np.ones(len(df))

print(df.head())

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

def calcium_plot_continuous(ax, df_, colorbar, norm, color_col, alpha):
    for _, row in df_.iterrows():
        Ca44_40 = row['Ca44_40']
        d13C_nominal = row['d13C']

        ax.errorbar(x=d13C_nominal, y=Ca44_40,
             yerr=None,
             xerr=None,
             color=row.color, marker=row.marker,
             markersize=6,
             markeredgewidth=0.5, alpha=row.alpha,
             ecolor=row.color, elinewidth=0.3, mec='k', label=member_type)

    ax.set_xlabel(r'$\delta^{13}$C (‰, VPDB)', fontsize=8)
    ax.set_ylabel(r'$\delta^{44/40}$Ca (‰, SW)', fontsize=8)

    ax.set_xlim(-20, 15)
    ax.set_ylim(-1.5, 0)


def fmt_mineral(x):
    s = f"{x:.0f}"
    if s == '5':
        s = r'$\delta^{18}$O = ' + s
    return rf"{s} \‰" if plt.rcParams["text.usetex"] else f"{s} ‰"

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
    calcium_plot_continuous(ax, df, 0, 0, df['color'], df['alpha'])

    # Adjust spacings if necessary
    fig.tight_layout()  # Adjust layout to make sure everything fits without overlap

    filename = f'figures/Finals_July2024/{plot_suffix}_square_33_nsc_ncrop_plot_d44Ca_d13C_oman_noKDEs.svg'
    filename2 = f'figures/Finals_July2024/{plot_suffix}_square_33_nsc_ncrop_plot_d44Ca_d13C_oman_noKDEs.pdf'
    plt.savefig(filename, format='svg', bbox_inches="tight", transparent=False, pad_inches=0)  # Omitting DPI since it's not necessary for PDF
    plt.savefig(filename2, format='pdf', bbox_inches="tight", transparent=False, pad_inches=0) 
    plt.close()


# Define a dictionary that maps color rules to plots
print(df['Formation'].unique())

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