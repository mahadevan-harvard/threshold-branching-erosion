import numpy as np
import matplotlib.pyplot as plt

import PlotLibrary as plotlib

# Plot Figure settings
figSpecs = plotlib.FigureSettings()
figSpecs.set_journal('PhysicalReview')
figSpecs.set_figureHeight(86)

# What is the size of this figure relative to the journal specs
xFraction = 1.0
yFraction = 1.0

width = xFraction * figSpecs.singleColumn
height = yFraction * figSpecs.figureHeight

# Define colormaps
cmap = plt.cm.viridis

fig = plt.figure(figsize=(width, height))
ax = plt.subplot()

##########################################
# Retrieve data
##########################################
config = "linsweeppressure_202509271359"

Q = 0.5 
T = 10.0
V = 1
input_file = f'./DATA/{config}/F_{Q}_T_{T}_V_{V:03d}/data.npz'
output_file = f'./Results/{config}_E_F_{Q:g}_T_{T:g}_V_{V:03d}.png'
data = np.load(input_file)

savior = data["phi"]

phi0 = savior[0]
phi_array = savior[-1]

ax.imshow(np.rot90(phi_array),cmap="viridis",vmin=0.0,vmax=1.0)

# savior = data["flux"]
# flux_array = np.log10(savior[-1])
# flux_array = savior[-1]
# flux_array = flux_array - np.min(flux_array)
# ax.imshow(np.rot90(flux_array),cmap='magma',vmin=0,vmax=5.0)

##########################################
# Final Lay-out
##########################################

ax.set_aspect('equal', adjustable='datalim')
ax.axis("off")
fig.patch.set_facecolor([0,0,0,0])

# Adjust the figure's position
plotlib.set_position(ax, x=0, y=0, width=1.0, height=1.0)
fig.savefig(output_file, dpi=600)
plt.show()