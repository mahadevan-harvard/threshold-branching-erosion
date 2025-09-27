import os

import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import minimize_scalar

from skimage import measure
from skimage import morphology

from erosion import simulate as fb

##############################################################################################
# FUNCTIONS 
##############################################################################################

#Code for robustness
def remove_isolated_islands(binary_image, min_size=1):
	# Ensure binary_image is a numpy array of type boolean
	binary_image = np.array(binary_image, dtype=bool)

	# Label connected components in the binary image
	labeled_image, _ = measure.label(binary_image, connectivity=2, return_num=True)

	# Create a mask to preserve components that touch the boundary
	boundary_mask = np.zeros_like(binary_image, dtype=bool)
	boundary_mask[0, :] = True
	boundary_mask[-1, :] = True
	boundary_mask[:, 0] = True
	boundary_mask[:, -1] = True

	# Find boundary-connected components
	boundary_connected_components = np.unique(labeled_image[boundary_mask])
	
	# Create a mask to preserve boundary-connected components
	preserve_mask = np.isin(labeled_image, boundary_connected_components)

	# Remove small objects (islands) not connected to the boundary
	cleaned_image = binary_image & preserve_mask

	# Optionally, remove small objects below a certain size threshold
	if min_size > 1:
		cleaned_image = morphology.remove_small_objects(cleaned_image, min_size=min_size)

	return cleaned_image

def count_holes(phi_array, a, min_size=10):

	# Threshold and invert
	phi = np.copy(phi_array)
	phi[phi < a] = 0
	phi[phi > a] = 1
	binary_optimal = remove_disconnected_channels(1 - phi)

	# Remove small features
	cleaned = morphology.remove_small_objects(np.logical_not(binary_optimal), min_size=min_size)

	# Label connected components
	labeled_components = measure.label(cleaned, connectivity=1)

	# Get all non-background labels
	unique_labels = np.unique(labeled_components)
	unique_labels = unique_labels[unique_labels != 0]

	# Define boundary mask (top, bottom, left)
	boundary_mask = np.zeros_like(labeled_components, dtype=bool)
	boundary_mask[0, :] = boundary_mask[-1, :] = True
	boundary_mask[:, 0] = True  # only left edge

	# Identify boundary-touching components
	boundary_labels = np.unique(labeled_components[boundary_mask])
	boundary_labels = boundary_labels[boundary_labels != 0]

	# Count non-boundary components
	non_boundary_components = len(unique_labels) - len(boundary_labels)
	return non_boundary_components

def maxhole(phi_array):
	"""
	Calculate Robustness
	"""
	result = minimize_scalar(lambda a: -count_holes(phi_array,a), bounds=(0, 0.5), method='bounded')
	x_max = result.x
	#x_max = 0.2
	maxnum = count_holes(phi_array,x_max)
	
	return maxnum

##############
# Alt count
#################

def remove_disconnected_channels(binary_image):
	"""
	Removes all foreground regions (True pixels) that are not connected to the top edge.
	:param binary_image: Binary image where True represents channels.
	:return: Processed binary image with only top-connected regions.
	"""
	# Label connected components
	labeled, num_labels = measure.label(binary_image, connectivity=2, return_num=True)
	
	# Find labels that touch the top edge
	top_edge_labels = set(labeled[:, 0]) - {0} # Labels present in the top row

	# Create a mask keeping only regions connected to the top edge
	filtered_binary = np.isin(labeled, list(top_edge_labels))

	return filtered_binary.astype(np.uint8)

# Code for efficiency 

def efficiency_ce(phi_array, p, dpdx, dpdy, bx, deltaP):
	phi = np.clip(phi_array,a_min=0.1,a_max=1)
	# I want to calculate the conductivity.
	# q*L/ \Delta P
	q = np.sum((-fb.kappa(phi)* dpdy)[:,-1]) # Sum the flux at the linesink
	K = q/deltaP
	return K

def extract_parameters(parameter_file, keys, default_values=None):
	"""
	Extract specific parameters from a parameter file with optional default values.

	Args:
		parameter_file (str): Path to the parameter file.
		keys (list): List of parameter names to extract.
		default_values (dict): Dictionary of default values for missing parameters.

	Returns:
		dict: A dictionary with the extracted parameter names and their values.
	"""
	parameters = {}
	
	# Read and parse the file
	with open(parameter_file, "r") as file:
		for line in file:
			key, value = line.strip().split("=")
			key = key.strip()
			if key in keys:
				parameters[key] = float(value.strip())
	
	# Handle missing keys with default values
	if default_values:
		for key, default in default_values.items():
			if key not in parameters:
				parameters[key] = default
	
	return parameters

##############################################################################################
# Retrieve data
##############################################################################################

# Input parameters
Folder = "linsweep_202507281142"
dataPath = f"./DATA/{Folder}/" 

subfolders = [
    name for name in os.listdir(dataPath)
    if os.path.isdir(os.path.join(dataPath, name))
]

epsilon_lh = 0.2
epsilon_rh = 5.0

# Sweep parameters
Es, Rs, As, Fs, Ts, Vs = [], [], [], [], [], []

for subFolder in subfolders:

	# extract F, T, V from folder name
	parts = subFolder.split('_')
	F = float(parts[1])
	T = float(parts[3])
	V = int(parts[5])

	Fs.append(F)
	Ts.append(T)
	Vs.append(V)

	# load and analyse data
	input_file = f"{dataPath}{subFolder}/data.npz"
	data = np.load(input_file)

	phi_array = data["phi"][-1]
	n = np.shape(phi_array)[1]
	bx = 10

	config = fb.generate_config(n, bx, epsilon_lh, epsilon_rh)
	p, dpdx, dpdy = fb.run_sim_pressure(phi_array, config)
	
	Es.append(efficiency_ce(phi_array, p, dpdx, dpdy, bx, 1))
	Rs.append(maxhole(phi_array))
	As.append(np.mean(phi_array))

	print(f"{Fs[-1]:.3f} {Ts[-1]:.3f} {Vs[-1]:.3f} {Es[-1]:.3f} {Rs[-1]} {As[-1]:.3f}")

# convert to numpy arrays if needed
Es = np.array(Es)
Rs = np.array(Rs)
As = np.array(As)
Fs = np.array(Fs)
Ts = np.array(Ts)
Vs = np.array(Vs)

with open("./Results/all_simulations_folder.txt", "w") as f:
	f.write("F\tT\tV\tA\tR\tE\n")
	for F, T, V, A, R, E in zip(Fs, Ts, Vs, As, Rs, Es):
		f.write(f"{F:.2f}\t{T:.2f}\t{V:03d}\t{A:.6f}\t{R:.6f}\t{E:.6f}\n")