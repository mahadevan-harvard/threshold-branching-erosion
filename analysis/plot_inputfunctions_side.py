import numpy as np


import matplotlib.pyplot as plt
import matplotlib

import PlotLibrary as plotlib

from matplotlib.ticker import (MultipleLocator)


cmap = matplotlib.colormaps['Greys']

# Plot Figure settings
figSpecs = plotlib.FigureSettings()
figSpecs.set_journal('PhysicalReview')
figSpecs.set_figureHeight(30)

# What is the size of this figure relative to the journal specs
xFraction = 30/85
yFraction = 1.0/2

width = xFraction*figSpecs.singleColumn
height = yFraction*figSpecs.figureHeight

# Figure Settings
fig = plt.figure(figsize=(width,height)) 
ax = plt.subplot()

######################################################################
# input parameters
t_rampup = 9
F = 1.2
a = 0.14
b = 0.0
epsilon_lh = 0.2
epsilon_rh = 5

bx = 10
q_max_lh = F/(2*epsilon_lh)
q_max_rh = F/(2*epsilon_rh)
q_max_side = F/(2*bx)

# arrays
t = np.arange(0,50,0.01)

factor_sink = np.ones(len(t))
factor_outside = np.zeros(len(t))

mask = t/t_rampup < 1.0
factor_sink[mask] = t[mask]/t_rampup
factor_outside[mask] = 0

mask = t/t_rampup < 2*a
factor_in = t/t_rampup			
factor_sink[mask] = b*factor_in[mask] + (1 - b)*(factor_in[mask] - a) + (1 - b)*(t[mask]/t_rampup - a) #
factor_outside[mask] = (1-b)*a - (1 - b)*(t[mask]/t_rampup - a) #ramp-down the sides

mask = t/t_rampup < a
factor_in = t/t_rampup
factor_sink[mask] = b*factor_in[mask]
factor_outside[mask] = (1-b)*factor_in[mask]

Q = F*factor_sink
Q_side = F*factor_outside										

ax.plot(t,Q,label=f"sink",c="C02",lw=1)
ax.plot(t,Q_side,label=f"side",c="C04",lw=1)

Q_sum = Q + Q_side

######################################################################
# Final layout settings
######################################################################

# Final adjustments to the figure
plotlib.set_box(ax)

ax.set_xlabel(r'$t$',labelpad=-1.5)
ax.set_ylabel(r'$Q$')

ax.set_xlim([0,15])
ax.set_ylim([-0.25,1.50])

leg = plotlib.set_legend(ax,pos=4)
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.2))

plotlib.set_position(ax, x=0.22, y=0.33, width=.70, height=0.60)

# Save the figure
fig.savefig('./function_combined.pdf')

plt.show()
