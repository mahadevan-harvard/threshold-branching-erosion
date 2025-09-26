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

# Curve shape depends quite a bit on fixed or non-fixed threshold, as well as the value of the threshols
# Double check why hole count is sometimes quite high, is it correcting for pixel size holes?
# def count_holes(phi_array,a):
# 	phi = np.copy(phi_array)
# 	phi[phi < a] = 0
# 	phi[phi > a] = 1

# 	phi = 1 - phi
# 	phi = remove_isolated_islands(phi)
	
# 	_, num_islands = ndi.label(1-phi)

# 	return num_islands

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
Folder = "side_202509041902"#linsweep_202505272242"
dataPath = f"./DATA/{Folder}/" 

F_range = np.array([1.0])#0.004])#np.array([0.2,0.4,0.6,0.8,1.0,1.2])
T_range = np.array([10])#2.1]) #np.array([5,7,9,11,13,15])
V_range = np.array([1])#,2,3])

epsilon_lh = 0.2
epsilon_rh = 5.0

# Sweep parameters
Es = np.zeros([len(F_range),len(T_range),len(V_range)])
Rs = np.zeros([len(F_range),len(T_range),len(V_range)])
As = np.zeros([len(F_range),len(T_range),len(V_range)])

for i,F in enumerate(F_range):
	for j,T in enumerate(T_range):
		for k,V in enumerate(V_range):

			subFolder = f"F_{F}_T_{T}_a_0.15_b_0"

			input_file = f"{dataPath}{subFolder}/data.npz"
			data = np.load(input_file)

			phi_array = data["phi"][-1] # np.load(f'{dataPath}phi_' + run + '.npy')
			n = np.shape(phi_array)[1]
			bx = 10

			config = fb.generate_config(n, bx, epsilon_lh, epsilon_rh)
			p, dpdx, dpdy = fb.run_sim_pressure(phi_array, config)
			
			Es[i,j,k] = efficiency_ce(phi_array, p, dpdx, dpdy, bx, 1)
			Rs[i,j,k] = maxhole(phi_array)
			As[i,j,k] = np.mean(phi_array)

			print(f"{Es[i,j,k]:.3f} {Rs[i,j,k]} {As[i,j,k]:.3f}")


with open("./Results/all_simulations_side.txt", "w") as f:
	f.write("F\tT\tV\tA\tR\tE\n")
	for i, F in enumerate(F_range):
		for j, T in enumerate(T_range):
			for k, V in enumerate(V_range):
				f.write(f"{F:.2f}\t{T}\t{V}\t{As[i,j,k]:.6f}\t{Rs[i,j,k]:.6f}\t{Es[i,j,k]:.6f}\n")

As_mean = As.mean(axis=2)
Rs_mean = Rs.mean(axis=2)
Es_mean = Es.mean(axis=2)

with open("./Results/averaged_simulations_side.txt", "w") as f:
	f.write("F\tT\tA\tR\tE\n")
	for i, F in enumerate(F_range):
		for j, T in enumerate(T_range):
			f.write(f"{F:.2f}\t{T}\t{As_mean[i,j]:.6f}\t{Rs_mean[i,j]:.6f}\t{Es_mean[i,j]:.6f}\n")