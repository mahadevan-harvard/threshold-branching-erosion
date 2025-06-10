import numpy as np

import scipy.ndimage as ndi
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve

from skimage import measure
from skimage import morphology

#import FrangibleBranching_Pressure as fb
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
	Calculate Robustness
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
	A[i-k:i+k+ 1, j-k:j+k+1] = 1#np.amax(A)
	return A

def robustness_test(phi, epsilon_lh, epsilon_rh, blockages=10):

	seeds = np.array([47]) #np.arange(45,50).astype(int)

	res = np.zeros((blockages+1,len(seeds)))

	# Simulation parameters
	n = np.shape(phi)[1]
	bx = 10
	config = fb.generate_config(n, bx, epsilon_lh, epsilon_rh)

	# Simulation without blockages
	p, dp_dx0, dp_dx1 = fb.run_sim(phi, config)
	q = - kappa(phi) * dp_dx1
	res[0,:] = np.sum(q[:,-1])

	for j, seed in enumerate(seeds):
		phi_runner = np.copy(phi)
		phi_binary = locate_channels(np.copy(phi_runner))
		idx_pl = sample_nonzero_indices_2d(phi_binary, N=blockages, seed=seed)

		for i, idx in enumerate(idx_pl):
			phi_runner = add_blockage(phi_runner, idx, 5)

			p, dp_dx0, dp_dx1 = fb.run_sim(phi_runner, config)        
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
	  
	# Input parameters
	Folder = "data_n"
	dataPath = f"./DATA/{Folder}/" 

	# Extract data for minimum and maximum number of loops
	idx_min = str(40)
	idx_max = str(68)

	phi_min = np.load(f'{dataPath}phi_' + idx_min + '.npy')
	p_min = np.load(f'{dataPath}p_' + idx_min + '.npy')
	t_final_min,epsilon_lh_min,epsilon_rh_min = np.load(f"{dataPath}important_" + idx_min + ".npy")
	F_min, T_min = np.load(f'{dataPath}tunables_' + idx_min + ".npy")

	phi_max = np.load(f'{dataPath}phi_' + idx_max + '.npy')
	p_max = np.load(f'{dataPath}p_' + idx_max + '.npy')
	t_final_max,epsilon_lh_max,epsilon_rh_max = np.load(f"{dataPath}important_" + idx_max + ".npy")
	F_max, T_max = np.load(f'{dataPath}tunables_' + idx_max + ".npy")				 

	############################
	# Robustness part
	############################

	# phi_min, j_min, E_min = robustness_test(np.copy(phi_min),epsilon_lh_min, epsilon_rh_min,blockages=0)
	# phi_plot = phi_min
	# j_plot = j_min

	phi_max, j_max, E_max = robustness_test(np.copy(phi_max),epsilon_lh_max, epsilon_rh_max,blockages=0)
	phi_plot = phi_max
	j_plot = j_max

	j_plot = np.log10(j_plot)

#	eps = 1e-8
#	j_plot = np.where(np.isneginf(j_plot), np.log10(eps), j_plot)

	print(np.min(j_plot))
	print(np.max(j_plot))

	##############################
	# Plotting
	##############################

	fig,ax = plt.subplots()
#	ax.imshow(np.rot90(j_plot),cmap="magma",vmin=-2,vmax=1)
	ax.imshow(np.rot90(phi_max),cmap="viridis",vmin=0,vmax=1)
#	ax.imshow(np.rot90(j_plot),cmap="magma",vmin=0,vmax=1)
	#ax.imshow(np.rot90(colored_components), alpha=0.6)  # Overlay connected components in color
	ax.set_xticks([])
	ax.set_yticks([])
	plt.gca().set_xticks([])  # Remove x-ticks
	plt.gca().set_yticks([])  # Remove y-ticks

	# Save the current figure as an image in memory
	plt.savefig('j_plot_max.png', bbox_inches='tight', pad_inches=0)
	plt.show()



############

	# start = 36
	# count = 42

	# # Sweep parameters
	# coors = np.zeros((count,2))
	# Es = np.zeros(count)
	# Rs = np.zeros(count)
	# As = np.zeros(count)

	# for r in range(start,start + count):
	# 	ri = int(r - start)
	# 	run = str(r)
	# 	#print(run)

	# 	phi_array = np.load(f'{dataPath}phi_' + run + '.npy')
	# 	p = np.load(f"{dataPath}p_" + run + ".npy")
	# 	t_final,epsilon_lh,epsilon_rh = np.load(f"{dataPath}important_" + run + ".npy")
	# 	F, T = np.load(f"{dataPath}tunables_" + run + ".npy")				                       

	# 	n = np.shape(phi_array)[1]
	# 	bx = 10

	# 	config = fb.generate_config(n, bx, epsilon_lh, epsilon_rh)
	# 	p, dpdx, dpdy = fb.run_sim(phi_array,config)
		
	# 	coors[ri,:] = np.array([F,T])
	# 	Es[ri] = efficiency_ce(phi_array, p, dpdx, dpdy, bx, 1)
	# 	Rs[ri] = maxhole(phi_array)
	# 	As[ri] = np.mean(phi_array)

	# 	idx_max = str(int(np.argmax(Rs)+start))
	# 	idx_min = str(int(np.argmin(Rs)+start))

	# 	print(f"{Es[ri]:.3f} {Rs[ri]} {As[ri]:.3f}")
	# 	count += 1


# Fs = coors[:,0]
# Ts = coors[:,1]

# import matplotlib.pyplot as plt
# fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(15,4))

# print(Es.shape)
# mask = np.ones(Es.shape[0], dtype=bool)
# mask[6::7] = False  # Set every 6th element (starting from index 5) to False

# Fs_plot = Fs[mask]
# Fs_plot = np.rot90(Fs_plot.reshape(6,6).T)

# Ts_plot = Ts[mask]
# Ts_plot = np.rot90(Ts_plot.reshape(6,6).T)

# Es_plot = Es[mask]
# Es_plot = np.rot90(Es_plot.reshape(6,6).T)
# im0 = ax[0].pcolormesh(Ts_plot, Fs_plot, Es_plot,cmap="Reds_r")
# fig.colorbar(im0, ax=ax[0], orientation='vertical')
# ax[0].set_title("Efficiency")
# ax[0].set_xlabel("T")
# ax[0].set_ylabel("Q")

# Rs_plot = Rs[mask]
# Rs_plot = np.rot90(Rs_plot.reshape(6,6).T)
# im1 = ax[1].pcolormesh(Ts_plot, Fs_plot, Rs_plot,cmap="Greens_r")
# fig.colorbar(im1, ax=ax[1], orientation='vertical')
# ax[1].set_title("Robustness ($N_{loops}$)")
# ax[1].set_xlabel("T")
# ax[1].set_ylabel("Q")

# As_plot = As[mask]
# As_plot = np.rot90(As_plot.reshape(6,6).T)
# im2 = ax[2].pcolormesh(Ts_plot, Fs_plot, As_plot,cmap="Blues_r")
# fig.colorbar(im2, ax=ax[2], orientation='vertical')
# ax[2].set_title("Mass fraction")
# ax[2].set_xlabel("T")
# ax[2].set_ylabel("Q")

# plt.savefig("functionality_space.png",dpi=600)
# plt.show()
