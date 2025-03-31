# Import standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import isotopylog as ipl
import time
import string  # Importing string module to use ascii_uppercase for subplot labels
import matplotlib as mpl

# Optional: Set a consistent style for plots
#plt.style.use('seaborn-whitegrid')

# Start timing for performance measurement
start_time = time.time()
mpl.rcParams['font.family'] = 'Arial'

# Define constants and input parameters
CALIBRATION = 'Aea21'
REF_FRAME = 'I-CDES'
ISO_PARAMS = 'Brand'
global MINERAL
MODEL = 'HH21'
NT = 100  # Timepoints for D47 calculation
T0_C = 30  # Surface temperature in Celsius
GEOTHERMAL_GRADIENT_PER_M = 25 / 1000.	# Convert degC/km to degC/m
BURIAL_RATE = 1e-6 / 1000.	# Convert Ma/km to Ma/m
TOTAL_SIMULATION_TIME = 500	 #Ma

mpl.rcParams['pdf.fonttype'] = 42  # Use Type 3 (vector) fonts in PDF
mpl.rcParams['ps.fonttype'] = 42   # Use Type 3 (vector) fonts in PS
mpl.rcParams['text.usetex'] = False

# Define functions for model calculations and data processing
# Note: Functions are annotated for clarity and maintainability

def calculate_heating_rate_and_time_at_Tf(file_path):
	# Load the CSV data
	df = pd.read_csv(file_path)
	# Ensure the DataFrame is sorted by time
	df.sort_values(by='time', inplace=True)

	# Preset initial D47 value, adjust as necessary
	D470 = 0.600

	# Calculate differences between consecutive temperatures and times
	df['temp_diff'] = df['WellC'].diff()
	df['time_diff'] = df['time'].diff()

	# Calculate heating rates between consecutive points (C/Ma)
	df['heating_rate'] = df['temp_diff'] / df['time_diff']

	# Identify the maximum burial temperature
	Tf_C = df['WellC'].max()

	# Calculate time spent at Tf (assuming a specific threshold for "close to Tf")
	threshold = 0.5	 # Example threshold, adjust based on your criteria
	close_to_Tf = df[(df['WellC'] >= Tf_C - threshold) & (df['WellC'] <= Tf_C + threshold)]
	tTf_Ma = close_to_Tf['time_diff'].sum()

	# Average heating rate during heating period (excluding cooling or constant periods)
	beta_CMa = df[df['temp_diff'] > 0]['heating_rate'].mean()

	return Tf_C, beta_CMa, tTf_Ma, D470

def iterate_model(Tf_C, beta_CMa, tTf_Ma, D470):
	"""
	Function to iteratively run HH21 model using global constants for calibration, iso_params, ref, ref_frame, and nt.
	"""
	# Use global constants directly within the function
	calibration = CALIBRATION
	iso_params = ISO_PARAMS
	ref = MODEL	 # Assuming MODEL is equivalent to 'ref' in this context
	ref_frame = REF_FRAME
	mineral = MINERAL
	nt = NT

	'''
	Function to iteratively run HH21 model.

	Parameters
	----------
	Tf_C : float
		Maximum burial temperature, in degrees celsius.
	
	beta_CMa : float
		Heating rate, in degrees celsius per million years.
	
	tTf_Ma : float
		Time spent at Tf, in Ma.
	
	D470 : float
		Initial D47 value in permil.

	calibration : string
		String of the T-D47 calibration to use. Defaults to ``Aea21'' for
		Anderson et al. (2021).

	iso_params : float
		The parameters to use for calculating clumped isotopes. Defaults to
		``Brand'' for Brand (2010).

	mineral : string
		String of the mineral of interest, either ``calcite'' or ``dolomite''.
		Defaults to ``calcite''.

	ref : string
		String of the reference to use for importing literature Edistribution
		values; can be ``PH12'', ``Hea14'', ``SE15'', or ``HH21''. 
		Defaults to ``HH21''.

	ref_frame : string
		String of the reference frame to use. Defaults to ``I-CDES'' for
		Bernasconi et al. (2021).

	nt : int
		Number of time steps to consider. Defaults to ``500''.


	Returns
	-------
	D : np.array
		D47 values at each time point, in permil. Reported using the
		same reference frame as designated on the inputs.

	D : np.array
		Standard deviation of D47 values at each time point, in periml. 
		Reported using the same reference frame as designated on the inputs.

	TfD : float
		Equilibrium temperature calculated from Dfinal, in degrees celsius.
		Calculated using the same reference frame and calibration as designated
		in the inputs.

	t : np.array
		Time vector, in seconds

	T : np.array
		Temperature vector, in Kelvin

	Notes
	-----
	* Time should be inputted in seconds (ed in inverse seconds)
	* Assumes d13C and d18O are both zero permil! Also assumes D47 std. dev. of 
	0.010 permil.
	* Assumes carbonate is initially in equilibrium (i.e., D0 = Deq(T0))
	'''

	#scalar for converting Ma to s
	Ma_to_s = 1e6*365*24*3600.

	#make EDistribution object from literature values
	# NOTE: in inverse seconds!
	ed = ipl.EDistribution.from_literature(
		mineral = mineral, 
		reference = ref
		)
	
	#define the initial composition and the time-temperature evolutions
	# NOTE: Assumes d13C and d18O are both zero permil! Also assumes D47 std.
	# dev. of 0.010 permil.
	d0 = [D470, 0, 0]
	d0_std = [0.010, 0, 0]
	
	#define initial and final temperature, assuming initially in equilibrium
	T0 = ipl.T_from_Deq(D470,
						clumps = 'CO47', 
						calibration = calibration, 
						ref_frame = ref_frame
						) # in K
	
	Tf = Tf_C + 273.15 #maximum burial temperature, converted to K
	b = beta_CMa/Ma_to_s # heating rate, converted to seconds
	
	#intial heating time, final cooling time based on beta and timestep
	t0 = 0 #seconds
	tf = (Tf-T0)/b #seconds

	#calculate time spent at Tf
	tTf = tTf_Ma*Ma_to_s #tTf converted from Ma to sec

	#calculate timestep such that total vector has length nt
	dt = (2*tf + tTf)/nt
	ntr = int(tf/dt)
	ntTf = nt - 2*ntr
	
	#calculate T-t paths based on variables above.
	# Paths are assumed to be symmetrical, i.e. trapezoid shaped
	T = np.concatenate([np.linspace(T0, Tf, ntr), 
						np.linspace(Tf,Tf, ntTf), 
						np.linspace(Tf, T0, ntr)
						])

	t = np.concatenate([np.linspace(t0, tf, ntr), 
						np.linspace(tf, tf+tTf, ntTf), 
						np.linspace(tf+tTf, tf+tTf+tf, ntr)
						])
# 	plt.plot(t,T)
# 	plt.savefig('debug2.png')
# 	plt.close()
	'''
   # Assuming 1 Ma timestep for the entire model duration
	nt = TOTAL_SIMULATION_TIME	# Number of timesteps in Ma
	print('nt:',nt)
	# Time vector setup, assuming t starts from 0 and ends at total_simulation_time
	t = np.linspace(0, TOTAL_SIMULATION_TIME, nt)
	print('nt:',nt,TOTAL_SIMULATION_TIME)
	
	# Temperature vector setup based on your model's specific logic
	# This example assumes a simple linear model for demonstration purposes
	T = np.linspace(T0, Tf, nt)
	'''
	# Calculate geologic history to get D47 and its std deviation
	D, Dstd = ipl.geologic_history(t, T, ed, d0, d0_std=d0_std, calibration=CALIBRATION, iso_params=ISO_PARAMS, ref_frame=REF_FRAME)
	
	# Convert D47 values and errors into temperatures and temperature bounds
	TfDs = [ipl.T_from_Deq(d, clumps='CO47', calibration=CALIBRATION, ref_frame=REF_FRAME) - 273.15 for d in D]
	upper_TfDs = [ipl.T_from_Deq(d + ds, clumps='CO47', calibration=CALIBRATION, ref_frame=REF_FRAME) - 273.15 for d, ds in zip(D, Dstd)]
	lower_TfDs = [ipl.T_from_Deq(d - ds, clumps='CO47', calibration=CALIBRATION, ref_frame=REF_FRAME) - 273.15 for d, ds in zip(D, Dstd)]
	
	print(f"Running model with Tf_C={Tf_C}, beta_CMa={beta_CMa}, tTf_Ma={tTf_Ma}, D470={D470}")
	
	return t, D, Dstd, TfDs, upper_TfDs, lower_TfDs

def model_thermal_history(inputAge,inputTemp, D470):
	"""
	Copied from iterate_model above
	Function to iteratively run HH21 model using global constants for calibration, iso_params, ref, ref_frame, and nt.
	"""
	# Use global constants directly within the function
	calibration = CALIBRATION
	iso_params = ISO_PARAMS
	ref = MODEL	 # Assuming MODEL is equivalent to 'ref' in this context
	ref_frame = REF_FRAME
	mineral = MINERAL
	nt = NT

	'''
	Function to iteratively run HH21 model.

	Parameters
	----------
	intputAge : numpy array float
		Age of points in thermal record in Mya
	
	inputTemp : numpy array float
		Temperature of thermal record in degC.

	D470 : float
		Initial D47 value in permil.

	Returns
	-------
	D : np.array
		D47 values at each time point, in permil. Reported using the
		same reference frame as designated on the inputs.

	D : np.array
		Standard deviation of D47 values at each time point, in periml. 
		Reported using the same reference frame as designated on the inputs.

	TfD : float
		Equilibrium temperature calculated from Dfinal, in degrees celsius.
		Calculated using the same reference frame and calibration as designated
		in the inputs.

	t : np.array
		Time vector, in My

	T : np.array
		Temperature vector, in degC

	Notes [COPIED FROM SOURCE]
	-----
	* Time should be inputted in seconds (ed in inverse seconds)
	* Assumes d13C and d18O are both zero permil! Also assumes D47 std. dev. of 
	0.010 permil.
	* Assumes carbonate is initially in equilibrium (i.e., D0 = Deq(T0))
	'''

	#scalar for converting Ma to s
	Ma_to_s = 1e6*365*24*3600.

	#make EDistribution object from literature values
	# NOTE: in inverse seconds!
	ed = ipl.EDistribution.from_literature(
		mineral = mineral, 
		reference = ref
		)
	
	#define the initial composition and the time-temperature evolutions
	# NOTE: Assumes d13C and d18O are both zero permil! Also assumes D47 std.
	# dev. of 0.010 permil.
	d0 = [D470, 0, 0]
	d0_std = [0.010, 0, 0]
	
	#Interpolate a new age array to have size nt
	t=np.linspace(inputAge[0],inputAge[-1],nt)
	#Interpolate a new temperature array to have size nt based on input observations and new age array
	T = np.interp(t,inputAge,inputTemp)

	#Make the new age array a time array
	t=t.max()-t[::-1]
	#Convert time to seconds
	t*=Ma_to_s
	#Reverse the temperature array
	T=T[::-1]
	#Convert temperature to Kelvin
	T+=273.15
	

# 	plt.plot((t.max()-t)/Ma_to_s,T-273.15)
# 	plt.scatter(inputAge,inputTemp)
# 	plt.savefig('debug2.png')
# 	plt.close()
	'''
   # Assuming 1 Ma timestep for the entire model duration
	nt = TOTAL_SIMULATION_TIME	# Number of timesteps in Ma
	print('nt:',nt)
	# Time vector setup, assuming t starts from 0 and ends at total_simulation_time
	t = np.linspace(0, TOTAL_SIMULATION_TIME, nt)
	print('nt:',nt,TOTAL_SIMULATION_TIME)
	
	# Temperature vector setup based on your model's specific logic
	# This example assumes a simple linear model for demonstration purposes
	T = np.linspace(T0, Tf, nt)
	'''
	# Calculate geologic history to get D47 and its std deviation
	D, Dstd = ipl.geologic_history(t, T, ed, d0, d0_std=d0_std, calibration=CALIBRATION, iso_params=ISO_PARAMS, ref_frame=REF_FRAME)
	
	# Convert D47 values and errors into temperatures and temperature bounds
	TfDs = [ipl.T_from_Deq(d, clumps='CO47', calibration=CALIBRATION, ref_frame=REF_FRAME) - 273.15 for d in D]
	upper_TfDs = [ipl.T_from_Deq(d + ds, clumps='CO47', calibration=CALIBRATION, ref_frame=REF_FRAME) - 273.15 for d, ds in zip(D, Dstd)]
	lower_TfDs = [ipl.T_from_Deq(d - ds, clumps='CO47', calibration=CALIBRATION, ref_frame=REF_FRAME) - 273.15 for d, ds in zip(D, Dstd)]
	#Reverse output arrays
	t=np.array(t)[::-1]
	D=np.array(D)[::-1]
	Dstd=np.array(Dstd)[::-1]
	TfDs=np.array(TfDs)[::-1]
	upper_TfDs=np.array(upper_TfDs)[::-1]
	lower_TfDs=np.array(lower_TfDs)[::-1]
	#Convert model time to age in Mya
	t=t.max()-t
	t/=Ma_to_s

	print(f"Running model with Tf_C={Tf_C}, beta_CMa={beta_CMa}, tTf_Ma={tTf_Ma}, D470={D470}")
	return t, D, Dstd, TfDs, upper_TfDs, lower_TfDs

def calculate_depth(t,max_depth,min_depth,burial_time):
	"""
	Calculates depth at a given time considering max depth and burial time.
	"""
	depth=max_depth
	if t < burial_time:
		depth=(t/burial_time)*max_depth
	return depth


def calculate_D47_temperature(t,max_depth,min_depth,geothermal_gradient,burial_time,D470,calibration,iso_params,model,ref_frame,nt,mineral,T0_C):
	"""
	Calculates the D47 temperature of a mineral given burial parameters.
	"""
	b=geothermal_gradient*(max_depth-min_depth)/burial_time
	tTf=t-burial_time
	Tf=T0_C+geothermal_gradient*(max_depth-min_depth)
	if tTf < 0:
		Tf=T0_C+b*t
		tTf=0
	#calculate mineral results
	_, _, TfD, _, _ = iterate_model(Tf, 
							   b, 
							   tTf, 
							   D470, 
							   calibration = calibration,
							   iso_params = iso_params,
							   mineral = mineral,
							   ref = model,
							   ref_frame = ref_frame,
							   nt = nt
							   )
	return TfD

def find_max_depth(Ttarget,t,da,db,min_depth,geothermal_gradient,burial_rate,D470,calibration,iso_params,model,ref_frame,nt,mineral,T0_C,tol=1e-4,bConfine=True):
	"""
	Iteratively finds a fit to a target temperature by varying depth
	"""
	TfDa=D47Ts_at_time_maxdepth(t,da,min_depth,geothermal_gradient,burial_rate*da,D470,calibration,iso_params,model,ref_frame,nt,mineral,T0_C)
	TfDb=D47Ts_at_time_maxdepth(t,db,min_depth,geothermal_gradient,burial_rate*db,D470,calibration,iso_params,model,ref_frame,nt,mineral,T0_C)
	if bConfine:
		if TfDa > Ttarget or TfDb < Ttarget:
			return None,None
	else:
		while TfDa > Ttarget:
			da-=1
			TfDa=D47Ts_at_time_maxdepth(t,da,min_depth,geothermal_gradient,burial_rate*da,D470,calibration,iso_params,model,ref_frame,nt,mineral,T0_C)
		while TfDb < Ttarget:
			db+=1
			TfDb=D47Ts_at_time_maxdepth(t,db,min_depth,geothermal_gradient,burial_rate*db,D470,calibration,iso_params,model,ref_frame,nt,mineral,T0_C)
	dn=0.5*(da+db)
	TfDn=D47Ts_at_time_maxdepth(t,dn,min_depth,geothermal_gradient,burial_rate*dn,D470,calibration,iso_params,model,ref_frame,nt,mineral,T0_C)
	while np.abs(TfDn-Ttarget)>tol:
		if TfDn > Ttarget:
			db=dn
		else:
			da=dn
		dn=0.5*(da+db)
		TfDn=D47Ts_at_time_maxdepth(t,dn,min_depth,geothermal_gradient,burial_rate*dn,D470,calibration,iso_params,model,ref_frame,nt,mineral,T0_C)
	if TfDn > Ttarget:
		db=dn
	else:
		da=dn
	dn=0.5*(da+db)
	return dn,TfDn

def find_max_time(Ttarget,d,ta,tb,min_depth,geothermal_gradient,burial_rate,D470,calibration,iso_params,model,ref_frame,nt,mineral,T0_C,tol=1e-4,bConfine=True):
	"""
	Iteratively finds a fit to a target temperature by varying depth
	"""
	TfDa=D47Ts_at_time_maxdepth(ta,d,min_depth,geothermal_gradient,burial_rate*d,D470,calibration,iso_params,model,ref_frame,nt,mineral,T0_C)
	TfDb=D47Ts_at_time_maxdepth(tb,d,min_depth,geothermal_gradient,burial_rate*d,D470,calibration,iso_params,model,ref_frame,nt,mineral,T0_C)
	if bConfine:
		if TfDa > contour or TfDb < contour:
			return None,None
	else:
		while TfDa > Ttarget:
			ta-=1
			TfDa=D47Ts_at_time_maxdepth(ta,d,min_depth,geothermal_gradient,burial_rate*d,D470,calibration,iso_params,model,ref_frame,nt,mineral,T0_C)
		while TfDb < Ttarget:
			tb+=1
			TfDb=D47Ts_at_time_maxdepth(tb,d,min_depth,geothermal_gradient,burial_rate*d,D470,calibration,iso_params,model,ref_frame,nt,mineral,T0_C)
	tn=0.5*(ta+tb)
	TfDn=D47Ts_at_time_maxdepth(tn,d,min_depth,geothermal_gradient,burial_rate*d,D470,calibration,iso_params,model,ref_frame,nt,mineral,T0_C)
	while np.abs(TfDn-Ttarget)>tol:
		if TfDn > Ttarget:
			tb=tn
		else:
			ta=tn
		tn=0.5*(ta+tb)
		TfDn=D47Ts_at_time_maxdepth(tn,d,min_depth,geothermal_gradient,burial_rate*d,D470,calibration,iso_params,model,ref_frame,nt,mineral,T0_C)
	if TfDn > Ttarget:
		tb=tn
	else:
		ta=tn
	tn=0.5*(ta+tb)
	return tn,TfDn

def process_data(file_path, mineral, calibration, ref_frame, nt, t0_c, geothermal_gradient_per_m, burial_rate):
	"""
	Processes D47 data for a given mineral type and calculates reordered temperatures.
	
	Parameters:
		- file_path (str): Path to the CSV file containing the data.
		- mineral (str): Mineral type to process ('calcite' or 'dolomite').
		- calibration (str): Calibration setting for the model.
		- ref_frame (str): Reference frame setting for the model.
		- nt (int): Number of time points for D47 calculation.
		- t0_c (float): Surface temperature in Celsius.
		- geothermal_gradient_per_m (float): Geothermal gradient per meter.
		- burial_rate (float): Burial rate in Ma/m.
	
	Returns:
		- reordered_temp_df (DataFrame): DataFrame containing original and reordered temperatures.
	"""
	df = pd.read_csv(file_path)
	df_filtered = df[df['M'].str.upper() == mineral_filter.upper()]

	d470 = ipl.Deq_from_T(t0_c + 273, calibration=calibration, ref_frame=ref_frame)

	# Initialize lists to store calculated data
	reordered_temps = []
	geothermal_gradients = []

	for i, row in df_filtered.iterrows():
		max_depth = row['Depth']
		bhT = row['CurrentBoreholeT']
		geothermal_gradient = bhT / max_depth
		reorderedT = calculate_reordered_temperature(row['Age'], max_depth, geothermal_gradient, burial_rate * max_depth, d470, calibration, ref_frame, nt, mineral_filter, t0_c)
		
		reordered_temps.append(reorderedT)
		geothermal_gradients.append(geothermal_gradient)

	# Construct reordered_temp_df with all necessary columns for plotting
	reordered_temp_df = df_filtered.copy()	# Start with the filtered DataFrame
	reordered_temp_df['Reordered T'] = reordered_temps
	reordered_temp_df['Geotherm'] = geothermal_gradients

	return reordered_temp_df


def plot_Huqf_model_outputs_and_errors(file_path, Tf_C, beta_CMa, tTf_Ma, D470):
    minerals = ['calcite', 'dolomite']  # Define the minerals to iterate through

    # Read and sort data once if it's common for all subplots
    df = pd.read_csv(file_path)
    df.sort_values(by='time', inplace=True)

    # Create a figure with two rows of plots (one for each mineral)
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # Creates a 2x3 grid of subplots for two minerals

    # Generate subplot labels (A, B, C, ...)
    subplot_labels = list(string.ascii_uppercase)[:6]  # Assuming 6 subplots

    for row, mineral in enumerate(minerals):
        global MINERAL
        MINERAL = mineral  # Set the global mineral variable before calling the model function

        for col in range(3):
            label_index = row * 3 + col
            temp_history = ['WellC', 'WellA', 'WellB'][col]
            temp_hist_label = ['Well C', 'Well A', 'Well B'][col]
            color = ['blue', 'green', 'purple'][col]

            t_model, D, Dstd, TfDs, upper_TfDs, lower_TfDs = model_thermal_history(np.array(df['time']), np.array(df[temp_history]), D470)
            temp_diff = TfDs[0] - TfDs[-1]  # Calculate temperature difference

            ax = axs[row, col]  # Access subplot directly
            ax.plot(df['time'], df[temp_history], label=f'{temp_hist_label} Temperature History', linestyle='--', color=color)
            ax.plot(t_model, TfDs, label=f'Reordered $\Delta_{{47}}$ Temperature ({mineral}) \nTemperature change estimate: {temp_diff:.1f}°C', linestyle='-', color=color)
            ax.fill_between(t_model, lower_TfDs, upper_TfDs, color=color, alpha=0.2, label=f'$\Delta_{{47}}$ Error Envelope')

            ax.invert_xaxis()
            ax.set_ylabel('' if col > 0 else 'Temperature (°C)')
            ax.set_xlabel('' if row == 0 else 'Time (Ma)')

            ax.legend()
#            ax.grid(True)
            ax.set_ylim(0, 225)
            ax.set_xlim(left=max(df['time']), right=0)


            # Conditional placement of the legend
            if subplot_labels[label_index] in ['A', 'B', 'D', 'E']:
                # Position legend on the far left below the subplot label
                ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.88))
            if subplot_labels[label_index] in ['C', 'F']:
                # Position legend on the far left below the subplot label
                ax.legend(loc='lower right', bbox_to_anchor=(1, 0))

            # Annotations
            ax.text(0.05, 0.95, subplot_labels[label_index], transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', color = 'white', bbox=dict(facecolor='black', alpha=1, edgecolor='none'))
 #           ax.text(df['time'].iloc[1] + 0.01 * (df['time'].iloc[-1] - df['time'].iloc[0]), 100,
 #                   f'Temperature change estimate for {mineral}:\n{temp_diff:.1f}°C',
 #                   ha='right', fontsize=10)

    plt.tight_layout()
    output = f"figures/{file_path.split('/')[-1].replace('.csv', '_huqf_temp_with_model_7_24.pdf')}"
#    output = file_path.split('/')[-1].replace('.csv', '_huqf_temp_with_model_and_errors.png')
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output}")

if __name__ == "__main__":
    file_paths = ['data/Oman_WellA_B_C.csv']
    for file_path in file_paths:
        Tf_C, beta_CMa, tTf_Ma, D470 = calculate_heating_rate_and_time_at_Tf(file_path)
        plot_Huqf_model_outputs_and_errors(file_path, Tf_C, beta_CMa, tTf_Ma, D470)