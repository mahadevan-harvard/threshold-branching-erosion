import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import minimize_scalar

from skimage import measure
from skimage import morphology

from erosion import simulate as fb


import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, LogLocator)

import PlotLibrary as plotlib

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
	
# 	a = x_max
# 	phi = np.copy(phi_array)
# 	phi[phi < a] = 0
# 	phi[phi > a] = 1

# 	# labels, num_islands = ndi.label(phi)
# 	# fig,ax = plt.subplots()
# 	# ax.imshow(labels,cmap="rainbow")
# 	# plt.show()

# 	print(a)
# 	binary_optimal = remove_disconnected_channels(1-phi)
# 	cleaned = morphology.remove_small_objects(np.logical_not(binary_optimal), min_size=10)

# 	labeled_components = measure.label(cleaned, connectivity=1)

# 	# Find unique component labels
# 	unique_labels = np.unique(labeled_components)
# 	unique_labels = unique_labels[unique_labels != 0]  # Exclude background

# 	# Create a mask of boundary pixels
# 	boundary_mask = np.zeros_like(labeled_components, dtype=bool)
# 	boundary_mask[0, :] = boundary_mask[-1, :] = True  # Top & bottom edges
# #	boundary_mask[:, 0] = boundary_mask[:, -1] = True  # Left & right edges
# 	boundary_mask[:, 0] = True  # Left & right edges

# 	# Find labels that touch the boundary
# 	boundary_labels = np.unique(labeled_components[boundary_mask])
# 	boundary_labels = boundary_labels[boundary_labels != 0]  # Remove background

# 	# Count components
# 	total_components = len(unique_labels)
# 	non_boundary_components = total_components - len(boundary_labels)
# 	print(non_boundary_components)

# 	# Assign new labels: 1 if touching boundary, 2 otherwise
# 	new_labels = np.where(np.isin(labeled_components, boundary_labels), -1, labeled_components)
# 	new_labels[labeled_components == 0] = 0
# 	fig,ax = plt.subplots()
# 	ax.imshow(new_labels,cmap="rainbow")
# 	plt.show()

# 	phi = 1 - phi
# 	phi = remove_isolated_islands(phi)
	
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


# Plot Figure settings
figSpecs = plotlib.FigureSettings()
figSpecs.set_journal('PhysicalReview')
figSpecs.set_figureHeight(45)

# What is the size of this figure relative to the journal specs
xFraction = 1/3
yFraction = 1.0

width = xFraction * figSpecs.doubleColumn
height = yFraction * figSpecs.figureHeight

# Define colormaps
cmap = plt.cm.viridis # plt.cm.inferno

fig = plt.figure(figsize=(width, height))
ax = plt.subplot()


##############################################################################################
# Retrieve data
##############################################################################################

# Input parameters
Folder = "linsweep_202505272242"
dataPath = f"./DATA/{Folder}/" 

F_range = np.array([0.2,0.4,0.6,0.8,1.0,1.2])
T_range = np.array([5,7,9,11,13,15])
V_range = np.array([1,2,3])

epsilon_lh = 0.2
epsilon_rh = 5.0

# Sweep parameters
Es = np.zeros([len(F_range),len(T_range),len(V_range)])
Rs = np.zeros([len(F_range),len(T_range),len(V_range)])
As = np.zeros([len(F_range),len(T_range),len(V_range)])

for i,F in enumerate(F_range):
	for j,T in enumerate(T_range):
		for k,V in enumerate(V_range):

			subFolder = f"F_{F}_T_{T}_V_{V:03d}"

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


##########################################
# Plot Data
##########################################
# I don't understand this filtering step: What is it's purpose
# print(Es.shape)
# mask = np.ones(Es.shape[0], dtype=bool)
# mask[6::7] = False  # Set every 6th element (starting from index 5) to False

# Compute edges for pcolormesh
def edges(arr):
	return np.concatenate(([arr[0] - (arr[1] - arr[0])/2],
	                       (arr[:-1] + arr[1:])/2,
	                       [arr[-1] + (arr[-1] - arr[-2])/2]))

F_edges = edges(F_range)
T_edges = edges(T_range)

Fs_plot, Ts_plot = np.meshgrid(F_edges, T_edges, indexing="ij")  # shape (6+1, 6+1)


# Es_plot = np.mean(Es,axis=2)
# im0 = ax.pcolormesh(Ts_plot, Fs_plot, Es_plot,cmap="Reds_r")
# cbar = fig.colorbar(im0, ax=ax, orientation='vertical')
# ax.set_title("Efficiency")

# Rs_plot = np.mean(Rs,axis=2)
# im1 = ax.pcolormesh(Ts_plot, Fs_plot, Rs_plot,cmap="Greens_r")
# cbar = fig.colorbar(im1, ax=ax, orientation='vertical')
# ax.set_title("Robustness ($N_{loops}$)")

As_plot = np.mean(As,axis=2)
im2 = ax.pcolormesh(Ts_plot, Fs_plot, As_plot,cmap="Blues_r")
cbar = fig.colorbar(im2, ax=ax, orientation='vertical')
ax.set_title("Mass fraction")

# Evaluate max robustness and efficiency

# Get flat index of max value
max_idx_flat = np.argmax(Es)
i, j, k = np.unravel_index(max_idx_flat, Es.shape)
print(f"Max Es at F={F_range[i]}, T={T_range[j]}, V={V_range[k]}, value={Es[i,j,k]}")

# Get flat index of max value
max_idx_flat = np.argmax(Rs)
i, j, k = np.unravel_index(max_idx_flat, Rs.shape)
print(f"Max Rs at F={F_range[i]}, T={T_range[j]}, V={V_range[k]}, value={Rs[i,j,k]}")

max_idx_flat = np.argmax(As)
i, j, k = np.unravel_index(max_idx_flat, As.shape)
print(f"Max As at F={F_range[i]}, T={T_range[j]}, V={V_range[k]}, value={As[i,j,k]}")


# Get flat index of max value
max_idx_flat = np.argmin(Rs)
i, j, k = np.unravel_index(max_idx_flat, Rs.shape)
print(f"Min Rs at F={F_range[i]}, T={T_range[j]}, V={V_range[k]}, value={Rs[i,j,k]}")

##########################################
# Final Lay-out
##########################################

# Final adjustments to the figure
plotlib.set_box(ax)
plotlib.set_box(cbar.ax)

ax.set_xlabel("T")
ax.set_ylabel("Q")

plotlib.set_position(ax,x=0.20, y=0.18, width=.60, height=0.70)
plotlib.set_position(cbar.ax,x=0.85, y=0.18, width=.10, height=0.70)


# Save the figure
plt.savefig("functionality_space_A_new.pdf")
plt.show()
