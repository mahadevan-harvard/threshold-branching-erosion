import numpy as np
import matplotlib.pyplot as plt

from skimage import measure as skm
from skimage.morphology import remove_small_objects

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from matplotlib.ticker import MultipleLocator

import PlotLibrary as plotlib

def remove_isolated_islands(image, min_size=1):
    # Ensure image is a numpy array
    image = np.array(image)

    # Label connected components in the image
    labeled_image, num_labels = skm.label(image > 0, connectivity=2, return_num=True)

    # Create a mask to preserve components that touch the boundary
    boundary_mask = np.zeros_like(image, dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True

    # Find boundary-connected components
    boundary_connected_components = np.unique(labeled_image[boundary_mask])

    # Create a mask to preserve boundary-connected components
    preserve_mask = np.isin(labeled_image, boundary_connected_components)

    # Remove small objects (islands) not connected to the boundary
    cleaned_image = image * preserve_mask

    # Optionally, remove small objects below a certain size threshold
    if min_size > 1:
        small_obj_removed = remove_small_objects(labeled_image, min_size=min_size)
        cleaned_image[small_obj_removed == 0] = 0

    return cleaned_image

# Plot Figure settings
figSpecs = plotlib.FigureSettings()
figSpecs.set_journal('PhysicalReview')
figSpecs.set_figureHeight(30)

# What is the size of this figure relative to the journal specs
xFraction = 1.0
yFraction = 1.0

width = xFraction * figSpecs.singleColumn
height = yFraction * figSpecs.figureHeight

# Define colormaps
cmap = plt.cm.viridis # plt.cm.inferno

fig = plt.figure(figsize=(width, height))
ax = plt.subplot()

##########################################
# Retrieve data
##########################################
config = "linsweep_202508141727"

Q = 5.0 #0.05, 0.5, 1.0, 2.0, 5.0
T = 100.0
V = 1
input_file = f'./DATA/{config}/F_{Q}_T_{T}_V_{V:03d}/data.npz'
output_file = f'./Results/{config}_E_F_{Q:g}_T_{T:g}_V_{V:03d}_cbar.pdf'
data = np.load(input_file)

savior = data["phi"]
phi_array = savior[-1]

bounds = [0.0, 0.5, 0.75, 1.0]
norm = Normalize(vmin=0, vmax=1)
sm = ScalarMappable(norm=norm, cmap=cmap)

im_fig, im_ax = plt.subplots()
im = im_ax.imshow(np.rot90(phi_array), cmap="viridis", vmin=0.0, vmax=1.0)
plt.close(im_fig)  # We don't need to display this figure

print(np.min(phi_array), np.max(phi_array))

cbar = fig.colorbar(sm, ax=ax, label=r"$\phi$", orientation="vertical")

##########################################
# Final Lay-out
##########################################

ax.set_aspect('equal', adjustable='datalim')
ax.axis("off")
fig.patch.set_facecolor([0,0,0,0])

cbar.ax.yaxis.set_major_locator(MultipleLocator(1.0))

# Adjust the figure's position
plotlib.set_position(ax, x=.0, y=2.5/30, width=1.0, height=24/30)
fig.savefig(output_file, dpi=600)
plt.show()