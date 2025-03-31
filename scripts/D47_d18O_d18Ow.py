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
#    elif member == 'Birba':
#        color_list.append('#E0E0E1')  # Grey
#        error_color_list.append('#E0E0E1')
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

# function to plot the CI ellipses
def plot_ci_oellipses(ax, df_):
    for _, group in df_.groupby('Formation'):
        x_mean = group['d18O_VPDB_mineral'].mean()
        y_mean = group['T_MIT'].mean()
        cov_matrix = group[['d18O_VPDB_mineral', 'T_MIT']].cov().values

        # Perform PCA
        eigvals, eigvecs = np.linalg.eig(cov_matrix)
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))  # Calculate orientation angle

        # Calculate semi-axes lengths
        semi_major_axis = 2 * np.sqrt(eigvals[0])
        semi_minor_axis = 2 * np.sqrt(eigvals[1])

        # Create Ellipse for the edge with alpha=0.5 (or any desired alpha value)
        cov_ellipse_edge = Ellipse((x_mean, y_mean), width=semi_major_axis, height=semi_minor_axis, angle=angle,
                                   edgecolor=group['color'].iloc[0], lw=0.5, fill=False)

        # Add both ellipses to the plot
        ax.add_patch(cov_ellipse_edge)
        
        # could add a kde plot instead of ellipse:  
        #sns.kdeplot(x=group['T_MIT'], y=group['d18Ow_VSMOW'], fill=True, color=group['color'].iloc[0],
        #            levels=5, ax=ax, bw_adjust=1.5, alpha=0.03, zorder=1)


def plot_ci_cellipses(ax, df_):
    for _, group in df_.groupby('Formation'):
        x_mean = group['d13C_VPDB'].mean()
        y_mean = group['T_MIT'].mean()
        cov_matrix = group[['d13C_VPDB', 'T_MIT']].cov().values

        # Perform PCA
        eigvals, eigvecs = np.linalg.eig(cov_matrix)
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))  # Calculate orientation angle

        # Calculate semi-axes lengths
        semi_major_axis = 2 * np.sqrt(eigvals[0])
        semi_minor_axis = 2 * np.sqrt(eigvals[1])

        # Create Ellipse for the edge with alpha=0.5 (or any desired alpha value)
        cov_ellipse_edge = Ellipse((x_mean, y_mean), width=semi_major_axis, height=semi_minor_axis, angle=angle,
                                   edgecolor=group['color'].iloc[0], lw=0.5, fill=False)

        # Add both ellipses to the plot
        ax.add_patch(cov_ellipse_edge)
        
def triple_plot_continuous(ax, df_, colorbar, norm, color_col, alpha):
    for _, row in df_.iterrows():
        T_MIT = row.T_MIT
        d18O_nominal = row.d18O_VPDB_mineral

        # Error bars for T_MIT_SE
        T_MIT_SE_low = row.T_MIT_SE_lower
        T_MIT_SE_high = row.T_MIT_SE_upper
        # Error bars for T_MIT_2SE
        T_MIT_2SE_low = row.T_MIT_2SE_lower
        T_MIT_2SE_high = row.T_MIT_2SE_upper

        # Plot the data point with solid error bars for T_MIT_SE
        ax.errorbar(x=d18O_nominal, y=T_MIT,
                    yerr=[[T_MIT_SE_low], [T_MIT_SE_high]],
                    color=row.color, marker=row.marker,
                    markersize=6,
                    markeredgewidth=0.5, alpha=row.alpha,
                    ecolor=row.error_color, elinewidth=1, mec='k', label=row.Formation, zorder=row.alpha*10)

        # Add dashed error bars for T_MIT_2SE with custom dashes
        line = ax.plot([d18O_nominal, d18O_nominal], [T_MIT - T_MIT_2SE_low, T_MIT + T_MIT_2SE_high],
                       color=row.error_color, linewidth=0.3, alpha=row.alpha, linestyle='dashed', zorder=row.alpha*10)
        line[0].set_dashes([5, 5])  # Lengths of dashes and spaces

    ax.set_xlabel(r'mineral $\delta^{18}$O (‰, VPDB)', fontsize=8)
    ax.set_ylabel(r'T $\Delta_{47}$ ($\degree$C)', fontsize=8)

    ax.set_xlim(-20, 15)
    ax.set_ylim(0, 140)

def carbonT_plot_continuous(ax, df_, colorbar, norm, color_col, alpha):
    for _, row in df_.iterrows():
        T_MIT = row.T_MIT
        d13C_nominal = row.d13C_VPDB
        T_MIT_low = row.T_MIT_95CL_lower
        T_MIT_high = row.T_MIT_95CL_upper
#        plot_ci_cellipses(ax7, df_)

        ax7.errorbar(x=d13C_nominal, y=T_MIT,
             yerr=np.array([[T_MIT_low, T_MIT_high]]).T,
             xerr=None,
             color=row.color, marker=row.marker,
             markersize=5,
             markeredgewidth=0.5, alpha=row.alpha,
             ecolor=row.color, elinewidth=0.3, mec='k', label=member_type)

    ax7.set_xlabel(r'$\delta^{13}$C (‰, VPDB)', fontsize=8)
    ax7.set_ylabel(r'T $\Delta_{47}$ ($\degree$C)', fontsize=8)

    ax7.set_xlim(-20, 15)
    ax7.set_ylim(0, 140)
    
def d18O_kdes(ax1, ax6, df_, colorbar, norm, color_col, alpha=0.6): 
    ax6.get_xaxis().set_visible(False)
    ax6.get_yaxis().set_visible(False)
    ax6.spines["right"].set_visible(False)
    ax6.spines["top"].set_visible(False)
    ax6.spines["bottom"].set_visible(False)

    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["left"].set_visible(False)

    for c in pd.unique(color_list):
        mask = df_['color'] == c
        dfi = df_[mask]
        
        # Convert data to Series
        x_series = dfi['d18O_VPDB_mineral'].values
        y_series = dfi['T_MIT'].values
        sns.kdeplot(x=x_series, fill=True, color=c, ax=ax1, bw_adjust=1.5, alpha=0.6, lw=0.5)
        sns.kdeplot(y=y_series, fill=True, color=c, ax=ax6, bw_adjust=1.5, alpha=0.6, lw=0.5)

    ax1.set_xlim(-20, 15)
    ax6.set_ylim(0, 140)

def make_calcite(D47_T, d18Ow, mineral='Calcite'):
    thousandlna_A21 = 17.5 * (1e3 * (1 / (D47_T + 273.15))) - 29.1
    a_A21 = np.exp((thousandlna_A21 / 1000))
    eps_A21 = (a_A21 - 1) * 1e3
    calciteSMOW = d18Ow + eps_A21
    calciteVPDB = (calciteSMOW - 30.92) / 1.03092
    return calciteVPDB

def make_dolomite(D47_T, d18Ow, mineral = 'Dolomite'):
	thousandlna_H14 = (3.14*(1e6*((D47_T+ 273.15)**-2)))-3.14
	a_H14 = np.exp((thousandlna_H14/1000))
	eps_H14 = (a_H14-1) * 1e3
	doloSMOW = d18Ow + eps_H14
	doloVPDB = (doloSMOW - 30.92) / 1.03092
	return doloVPDB

def make_water(D47_T, d18O_VPDB_mineral, mineral='Calcite'):
    thousandlna_A21 = 17.5 * (1e3 * (1 / (D47_T + 273.15))) - 29.1
    a_A21 = np.exp((thousandlna_A21 / 1000))
    eps_A21 = (a_A21 - 1) * 1e3
    d18O_VSMOW = (d18O_VPDB_mineral * 1.03092) + 30.92
    d18Ow =  d18O_VSMOW - eps_A21 
    return d18Ow


def make_contour_water(df_, ax):
    # Define the temperature (T) and delta O-18 (d18O) ranges
    T_range = np.linspace(0, 140, num=120)  # for example, 120 points between 10 and 130
    d18O_range = np.linspace(-20, 15, num=120)  # matching number of points in d18O_range

    # Initialize the contour grid
    d18Ow_contour = np.zeros((len(T_range), len(d18O_range)))

    # Populate the contour grid
    for i, T in enumerate(T_range):
        for j, d18O in enumerate(d18O_range):
            d18Ow_contour[i, j] = make_water(T, d18O, 'Calcite')

    return T_range, d18O_range, d18Ow_contour

    
def make_contour_dolomite(df_, ax):
    # Get the x-axis and y-axis limits from ax1 and ax6
    xlim = ax1.get_xlim()
    ylim = ax6.get_ylim()

    # Calculate T_range and d18Ow_range
    T_range = np.linspace(xlim[0], xlim[1], 100)
    d18Ow_range = np.linspace(ylim[0], ylim[1], 100)

    T_contour = []
    d18O_contour = []
    d18Ow_contour = []

    # Calculate d18O_contour using make_rock function
    d18O_d_contour = []
          
    for i in range(len(T_range)):
	    for j in range(len(d18Ow_range)):
		    d18O_d_contour.append(make_dolomite(T_range[j], d18Ow_range[i], 'Dolomite'))           

    d18O_d_contour = np.array(d18O_d_contour)
    d18O_d_contour = d18O_d_contour.reshape((len(T_range), len(d18Ow_range)))

    return T_range, d18Ow_range, d18O_d_contour

def fmt_mineral(x):
    s = f"{x:.0f}"
    if s == '5':
        s = r'$\delta^{18}$O = ' + s
    return rf"{s} \‰" if plt.rcParams["text.usetex"] else f"{s} ‰"

def make_plot(df, plot_suffix):
    # Set up a single plot directly without using subplot indexing
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # Set x-axis ticks and labels
    x_ticks = [-20, -15, -10, -5, 0, 5, 10, 15]
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(["-20", "-15", "-10", "-5", "0", "5", "10", "15"]))

    # Call the plotting function for generating contour data and plots
    T_range, d18O_range, d18Ow_contour = make_contour_water(df, ax)
    dp = ax.contour(d18O_range, T_range, d18Ow_contour, colors='grey', alpha=0.8, linewidths=0.4, levels=[-20, -10, 0, 10, 20])
    ax.clabel(dp, dp.levels, inline=True, inline_spacing=0.1, fmt=fmt_mineral, fontsize=8)

    # Plotting continuous data
    triple_plot_continuous(ax, df, 0, 0, df['color'], df['alpha'])

    # Adjust spacings if necessary
    fig.tight_layout()  # Adjust layout to ensure proper spacing and no overlap

    # Manually set y-axis labels to be visible
    ax.yaxis.set_tick_params(which='both', labelleft=True)

    # Save and close figure
    filename = f'figures/Finals_July2024//{plot_suffix}_square_32_nsc_ncrop_plot_D47_d18O_oman_noKDEs.svg'
    filename2 = f'figures/Finals_July2024//{plot_suffix}_square_32_nsc_ncrop_plot_D47_d18O_oman_noKDEs.pdf'
    plt.savefig(filename, format='svg', bbox_inches="tight", transparent=False, pad_inches=0)  # DPI is not needed for vector formats like PDF
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