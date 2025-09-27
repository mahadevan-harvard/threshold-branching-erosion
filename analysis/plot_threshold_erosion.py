import numpy as np


import matplotlib.pyplot as plt
import matplotlib

import PlotLibrary as plotlib


from matplotlib.colors import LinearSegmentedColormap
cmap = matplotlib.colormaps['Greys']

# Plot Figure settings
figSpecs = plotlib.FigureSettings()
figSpecs.set_journal('PhysicalReview')
figSpecs.set_figureHeight(30)

# What is the size of this figure relative to the journal specs
xFraction = 30/86
yFraction = 1.0

width = xFraction*figSpecs.singleColumn
height = yFraction*figSpecs.figureHeight

# Figure Settings
fig = plt.figure(figsize=(width,height)) 
ax = plt.subplot()

######################################################################
# Parameters
######################################################################

aa = 0.2  # slope of first line
bb = 5# slope of second line
num_intermediate_lines = 5  # Number of interpolated lines

# Calculate angles
theta1 = np.arctan(aa)  # angle of first line
theta2 = np.arctan(bb)  # angle of second line
angles = np.linspace(theta1, theta2, num_intermediate_lines + 2)

# Create x values
x1 = 0.5*np.linspace(0, 2, 100)

# Create custom colormap for transition from blue to red
colors = [plt.get_cmap('tab10')(0), 'firebrick']  # This gets tab:blue and firebrick
custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=num_intermediate_lines + 2)
cmap = plt.cm.viridis_r
# Generate all lines
for i, theta in enumerate(angles):
    
    # Calculate slope from angle
    current_slope = np.tan(theta)
    y = current_slope * x1
    
    # Get color from colormap
    weight = (i / (num_intermediate_lines + 1))
    color = cmap(0.2+0.8*weight)
    
    # Calculate alpha (transparency)
    if i == 0 or i == num_intermediate_lines + 1:
        alpha = 1.0  # Original lines are fully opaque
    else:
        alpha = 0.3  # Intermediate lines are more transparent
    
    # Plot the line
    ax.plot(x1, y, color=color, linewidth=1, alpha=alpha)

######################################################################
# Final layout settings
######################################################################


# Final adjustments to the figure
plotlib.set_box(ax)

ax.set_xlabel(r'$\nabla p$')
ax.set_ylabel(r'$q$')

plt.xlim(0, 0.65)
plt.ylim(0, 0.65)

ax.set_xticks([])
ax.set_yticks([])

hfrac = 5/30
frac = 1 - 2.5/30 - hfrac

wfrac = 1 - 0.15 - 2.5/30

plotlib.set_position(ax,x=0.15,y=hfrac,width=wfrac,height=frac)

# Save the figure
fig.savefig('./threshold_erosion.pdf')

plt.show()
