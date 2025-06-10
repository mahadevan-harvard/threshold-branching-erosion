import numpy as np

import scipy.ndimage as ndi
from scipy.optimize import minimize_scalar

from skimage import measure
from skimage import morphology

import frangiblebranching_ns as fb

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

#####################
# Code for efficiency
#####################

#  
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
# Figure Settings
##############################################################################################

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
# CORE 
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


option = 2

if option == 0:
	xOption = "R" 
	yOption = "E" 
elif option == 1:
	xOption = "R" 
	yOption = "A" 
elif option == 2:
	xOption = "E" 
	yOption = "A" 

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


# Setup marker and color maps
marker_styles = ['o', 's', '^', 'D', 'v', 'P', '*']
F_to_marker = {f: marker_styles[i % len(marker_styles)] for i, f in enumerate(F_range)}

# Normalize T for colormap
T_norm = plt.Normalize(vmin=T_range.min(), vmax=T_range.max())
cmap = plt.cm.viridis

# Precompute data
Es_plot = np.mean(Es, axis=2)
Rs_plot = np.mean(Rs, axis=2)
As_plot = np.mean(As, axis=2)

for i, F in enumerate(F_range):
	for j, T in enumerate(T_range):
		color = cmap(T_norm(T))
		marker = F_to_marker[F]
		if option == 0:
			ax.scatter(Rs_plot[i, j], Es_plot[i, j], color=color, marker=marker,
					  edgecolor='black', linewidth=0.5,s=8,zorder=3)
		elif option == 1:
			ax.scatter(Rs_plot[i, j], As_plot[i, j], color=color, marker=marker,
					  edgecolor='black', linewidth=0.5,s=8,zorder=3)
		elif option == 2:
			ax.scatter(Es_plot[i, j], As_plot[i, j], color=color, marker=marker,
					  edgecolor='black', linewidth=0.5,s=8,zorder=3)

for j, T in enumerate(T_range):
	color = cmap(T_norm(T))
	marker = F_to_marker[F]
	if option == 0:
		ax.plot(Rs_plot[:, j], Es_plot[:, j], color=color,
					linewidth=1.0)
	elif option == 1:
		ax.plot(Rs_plot[:, j], As_plot[:, j], color=color,
					linewidth=1.0)
	elif option == 2:
		ax.plot(Es_plot[:, j], As_plot[:, j], color=color,
					linewidth=1.0)


##########################################
# Final Lay-out
##########################################

# Final adjustments to the figure
plotlib.set_box(ax)

# Axis labels
ax.set_xlabel(xOption)
ax.set_ylabel(yOption)

if option == 0:
	ax.set_xlim([0,50])
	ax.set_ylim([0,350])
	ax.xaxis.set_major_locator(MultipleLocator(10))
	ax.xaxis.set_minor_locator(MultipleLocator(2))
	ax.yaxis.set_major_locator(MultipleLocator(100))
	ax.yaxis.set_minor_locator(MultipleLocator(20))
elif option == 1:
	ax.set_xlim([0,50])
	ax.set_ylim([0.40,0.80])
	ax.xaxis.set_major_locator(MultipleLocator(10))
	ax.xaxis.set_minor_locator(MultipleLocator(2))
	ax.yaxis.set_major_locator(MultipleLocator(0.10))
	ax.yaxis.set_minor_locator(MultipleLocator(0.02))
elif option == 2:
	ax.set_xlim([0,350])
	ax.set_ylim([0.40,0.80])
	ax.xaxis.set_major_locator(MultipleLocator(100))
	ax.xaxis.set_minor_locator(MultipleLocator(20))
	ax.yaxis.set_major_locator(MultipleLocator(0.10))
	ax.yaxis.set_minor_locator(MultipleLocator(0.02))

plotlib.set_position(ax,x=0.20, y=0.18, width=.70, height=0.70)

# Save the figure
plt.savefig(f"functionality_correlation_{xOption}{yOption}_new.pdf")
plt.show()


# # Add colorbar for T
# sm = plt.cm.ScalarMappable(norm=T_norm, cmap=cmap)
# sm.set_array([])
# cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
# cbar.set_label("T")

# # Add marker legend for F
# handles = [plt.Line2D([0], [0], marker=marker, color='w',
# 					  markerfacecolor='gray', markeredgecolor='black',
# 					  markersize=10, label=f"Q={f:.2f}")
# 		   for f, marker in F_to_marker.items()]
# fig.legend(handles=handles, loc='lower center', ncol=len(F_range), frameon=False)