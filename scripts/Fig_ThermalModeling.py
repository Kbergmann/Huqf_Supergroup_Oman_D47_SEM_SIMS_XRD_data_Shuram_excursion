"""
Script for modeling the thermal history of carbonate rocks during burial

This script generates a plot of δ44/40Ca (calcium isotopes) versus δ13C (carbon isotopes)
from geochemical datasets. It is specifically designed for clumped isotope and geochemical
studies, allowing for clear visualization of formation-specific data.

Key outputs include burial temperature histories, Δ47 temperatures, and 
visualizations of diagenetic alteration effects. It uses code developed by Jordan Hemingway and Greg Henkes in Hemingway & Henkes, 2021 which should also
be cited if reused.

Author: Kristin Bergmann
Date: November 2024
Contact: kdberg@mit.edu and https://github.com/Kbergmann

Dependencies:
- numpy
- pandas
- matplotlib

License:
This script is open-source and distributed under the MIT License.
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import isotopylog as ipl  # Ensure this custom library is accessible
import matplotlib as mpl
import string  # Used for subplot labeling

# Configure Matplotlib for consistent and publication-quality plots
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42  # Type 3 fonts for vector graphics compatibility
mpl.rcParams['ps.fonttype'] = 42

# Constants
CALIBRATION = 'Aea21'
REF_FRAME = 'I-CDES'
ISO_PARAMS = 'Brand'
MODEL = 'HH21'
NT = 100  # Number of timepoints for thermal history calculations
T0_C = 30  # Surface temperature in Celsius
GEOTHERMAL_GRADIENT_PER_M = 25 / 1000  # °C/m
BURIAL_RATE = 1e-6 / 1000  # Ma/m
TOTAL_SIMULATION_TIME = 500  # Ma

# Helper Functions
def calculate_heating_rate_and_time_at_Tf(file_path):
    """
    Calculates maximum burial temperature, heating rate, and time spent at peak temperature.
    Parameters:
        file_path (str): Path to the CSV file containing thermal history data.
    Returns:
        Tf_C (float): Maximum burial temperature in Celsius.
        beta_CMa (float): Average heating rate (°C/Ma).
        tTf_Ma (float): Duration at peak burial temperature (Ma).
        D470 (float): Initial clumped isotope value for calculations.
    """
    df = pd.read_csv(file_path)
    df.sort_values(by='time', inplace=True)

    D470 = 0.600  # Initial clumped isotope value (example default)
    df['temp_diff'] = df['WellC'].diff()
    df['time_diff'] = df['time'].diff()
    df['heating_rate'] = df['temp_diff'] / df['time_diff']

    Tf_C = df['WellC'].max()
    threshold = 0.5  # ± range around Tf to define "close to Tf"
    close_to_Tf = df[(df['WellC'] >= Tf_C - threshold) & (df['WellC'] <= Tf_C + threshold)]
    tTf_Ma = close_to_Tf['time_diff'].sum()

    beta_CMa = df[df['temp_diff'] > 0]['heating_rate'].mean()
    return Tf_C, beta_CMa, tTf_Ma, D470

def model_thermal_history(input_ages, input_temps, D470):
    """
    Models Δ47 evolution using thermal history input.
    Parameters:
        input_ages (array): Timepoints (Ma).
        input_temps (array): Temperatures (°C) at corresponding timepoints.
        D470 (float): Initial Δ47 value.
    Returns:
        Modeled outputs including temperatures and Δ47 values at each time step.
    """
    t = np.linspace(input_ages[0], input_ages[-1], NT)
    T = np.interp(t, input_ages, input_temps)

    t = t.max() - t[::-1]
    T = T[::-1] + 273.15  # Convert to Kelvin

    ed = ipl.EDistribution.from_literature(mineral='calcite', reference=MODEL)
    d0 = [D470, 0, 0]
    d0_std = [0.010, 0, 0]

    D, Dstd = ipl.geologic_history(t, T, ed, d0, d0_std=d0_std, calibration=CALIBRATION, iso_params=ISO_PARAMS, ref_frame=REF_FRAME)
    TfDs = [ipl.T_from_Deq(d, clumps='CO47', calibration=CALIBRATION, ref_frame=REF_FRAME) - 273.15 for d in D]
    return t, D, Dstd, TfDs

def plot_Huqf_model_outputs_and_errors(file_path, Tf_C, beta_CMa, tTf_Ma, D470):
    """
    Visualizes burial history and Δ47 evolution for Huqf Group wells.
    Parameters:
        file_path (str): Path to the CSV file.
        Tf_C, beta_CMa, tTf_Ma, D470: Model parameters.
    """
    df = pd.read_csv(file_path)
    minerals = ['calcite', 'dolomite']

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    subplot_labels = list(string.ascii_uppercase)[:6]

    for row, mineral in enumerate(minerals):
        for col, temp_col in enumerate(['WellC', 'WellA', 'WellB']):
            t_model, _, _, TfDs = model_thermal_history(df['time'], df[temp_col], D470)
            temp_diff = TfDs[0] - TfDs[-1]

            ax = axs[row, col]
            ax.plot(df['time'], df[temp_col], linestyle='--', label=f'{temp_col} History')
            ax.plot(t_model, TfDs, linestyle='-', label=f'Reordered Δ47 ({mineral})')

            ax.invert_xaxis()
            ax.text(0.05, 0.95, subplot_labels[row * 3 + col], transform=ax.transAxes, fontsize=9,
                    bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
            ax.legend()

    plt.tight_layout()
    plt.savefig("Huqf_model_outputs.pdf", bbox_inches="tight")
    plt.close()

# Main script execution
if __name__ == "__main__":
    file_paths = ['data/Oman_WellA_B_C.csv']
    for file_path in file_paths:
        Tf_C, beta_CMa, tTf_Ma, D470 = calculate_heating_rate_and_time_at_Tf(file_path)
        plot_Huqf_model_outputs_and_errors(file_path, Tf_C, beta_CMa, tTf_Ma, D470)