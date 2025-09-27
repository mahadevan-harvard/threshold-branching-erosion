"""
N.B. This code requires fenicsx
"""

import numpy as np

import scipy.ndimage as ndi
from scipy.optimize import minimize_scalar

from skimage import measure
from skimage import morphology

from erosion import simulate as fb

import matplotlib.pyplot as plt

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

def count_holes(phi_array,a):
	phi = np.copy(phi_array)
	phi[phi < a] = 0
	phi[phi > a] = 1

	phi = 1 - phi
	phi = remove_isolated_islands(phi)
	
	_, num_islands = ndi.label(1-phi)
	return num_islands

def maxhole(phi_array):
	"""
	Find binarization threshold that maximizes the number of holes
	"""
	result = minimize_scalar(lambda a: -count_holes(phi_array,a), bounds=(0, 0.5), method='bounded')
	x_max = result.x
	maxnum = count_holes(phi_array,x_max)
	
	a = x_max
	phi = np.copy(phi_array)
	phi[phi < a] = 0
	phi[phi > a] = 1

	phi = 1 - phi
	phi = remove_isolated_islands(phi)
	
	return maxnum

# Code for efficiency 

def efficiency_ce(phi_array, p, dpdx, dpdy, bx, deltaP):
	phi = np.clip(phi_array,a_min=0.1,a_max=1)
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

def kappa(phi):
    return (1-phi)**3/phi**2

##############################################################################################
# ROBUSTNESS FUNCTIONS 
##############################################################################################

def sample_nonzero_indices_2d(binary_array, N=1, seed=None, replace=True):
	"""
	Select indices in a channel
	"""

	# Set the seed if provided
	if seed is not None:
		np.random.seed(seed)
	
	# Find the indices of non-zero elements
	nonzero_indices = np.argwhere(binary_array)
	
	# If there are no non-zero elements, return None or raise an exception
	if len(nonzero_indices) == 0:
		return None  # or raise ValueError("No non-zero elements in the array")
	
	# Check if we can sample N indices
	if not replace and N > len(nonzero_indices):
		raise ValueError(f"Cannot sample {N} indices without replacement from {len(nonzero_indices)} non-zero elements")
	
	# Randomly select N indices
	sampled_indices = nonzero_indices[np.random.choice(len(nonzero_indices), size=N, replace=replace)]
	
	# Convert to list of tuples
	return [tuple(index) for index in sampled_indices]

def locate_channels(phi_array): # in-use
	"""
	Finds channels
	"""
	result = minimize_scalar(lambda a: -count_holes(phi_array,a), bounds=(0, 0.5), method='bounded')
	x_max = result.x
	maxnum = count_holes(phi_array,x_max)
	
	a = x_max
	phi = np.copy(phi_array)
	phi[phi < a] = 0
	phi[phi > a] = 1

	phi = 1 - phi
	phi = remove_isolated_islands(phi)

	return phi

def add_blockage(A, idx, k): # in-use
	# Set the kxk block to 1
	i, j = idx
	A[i-k:i+k+ 1, j-k:j+k+1] = 1
	return A

def robustness_test(phi, epsilon_lh, epsilon_rh, blockages=10):

	seeds = np.array([5])#

	res = np.zeros((blockages+1,len(seeds)))

	# Simulation parameters
	n = np.shape(phi)[1]
	bx = 10
	config = fb.generate_config(n, bx, epsilon_lh, epsilon_rh)

	# Simulation without blockages
	p, dp_dx0, dp_dx1 = fb.run_sim_pressure(phi, config)
	q = - kappa(phi) * dp_dx1
	res[0,:] = np.sum(q[:,-1])

	for j, seed in enumerate(seeds):
		phi_runner = np.copy(phi)
		phi_binary = locate_channels(np.copy(phi_runner))
		idx_pl = sample_nonzero_indices_2d(phi_binary, N=blockages, seed=seed)

		for i, idx in enumerate(idx_pl):
			phi_runner = add_blockage(phi_runner, idx, 5)

			p, dp_dx0, dp_dx1 = fb.run_sim_pressure(phi_runner, config)        
			q = - kappa(phi_runner)*dp_dx1

			res[i+1,j] = np.sum(q[:,-1])

	Sigma = kappa(phi_runner)
	jx = -Sigma * dp_dx0
	jy = -Sigma * dp_dx1

	flux_magnitude = np.sqrt(jx**2 + jy**2)

	
	return phi_runner, flux_magnitude, np.average(res,axis=1)

##############################################################################################
# CORE 
##############################################################################################

if __name__ == "__main__":

	epsilon_lh = 0.2
	epsilon_rh = 5.0

	# Input parameters
	Folder = "linsweep_202507281142"
	dataPath = f"./DATA/{Folder}/" 

	# Extract data for minimum and maximum number of loops
	subFolder_low = f"F_1.4_T_100.0_V_001"
	input_file = f"{dataPath}{subFolder_low}/data.npz"
	data = np.load(input_file)
	phi_min = data["phi"][-1]

	subFolder_high = f"F_0.8_T_9_V_001"
	input_file = f"{dataPath}{subFolder_high}/data.npz"
	data = np.load(input_file)
	phi_max = data["phi"][-1]

	############################
	# Robustness part
	############################

	phi_max, j_max, E_max = robustness_test(np.copy(phi_max),epsilon_lh, epsilon_rh,blockages=10)
	phi_plot = phi_max
	j_plot = j_max

	j_plot = np.log10(j_plot)

	##############################
	# Plotting
	##############################
	fig,ax = plt.subplots()

	ax.imshow(np.rot90(j_plot),cmap="magma",vmin=-2,vmax=1)

	ax.set_xticks([])
	ax.set_yticks([])
	plt.gca().set_xticks([])  # Remove x-ticks
	plt.gca().set_yticks([])  # Remove y-ticks

	# Save the current figure as an image in memory
	plt.savefig('jlog_obstacles.png', bbox_inches='tight', pad_inches=0)
	plt.show()
