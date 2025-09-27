import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import PlotLibrary as plotlib

from matplotlib.ticker import (MultipleLocator)


cmap = matplotlib.colormaps['Greys']

# Plot Figure settings
figSpecs = plotlib.FigureSettings()
figSpecs.set_journal('PhysicalReview')
figSpecs.set_figureHeight(30.0)

# What is the size of this figure relative to the journal specs
xFraction = 30/85
yFraction = 1.0/2

width = xFraction*figSpecs.singleColumn
height = yFraction*figSpecs.figureHeight

# Figure Settings
fig = plt.figure(figsize=(width,height)) 
ax = plt.subplot()


def get_flux(t, m1, m2, width1, width2):
    """
    Compute the flux at time t with alternating slopes m1 and m2.
    
    Parameters:
    t (float): Time.
    m1 (float): Slope for the first interval.
    m2 (float): Slope for the second interval.
    width1 (float): Duration of the first slope interval.
    width2 (float): Duration of the second slope interval.
    
    Returns:
    float: The flux at time t.
    """
    # Compute the total period of alternation
    period = width1 + width2
    
    # Determine the number of completed periods
    complete_periods = int(t // period)
    
    # Determine the time within the current period
    t_within_period = t % period
    
    # Calculate the flux at the start of this period
    flux_base = complete_periods * (m1 * width1 + m2 * width2)
    
    # Determine the contribution from the current period
    if t_within_period <= width1:
        # Within the first slope interval
        return flux_base + m1 * t_within_period
    else:
        # Within the second slope interval
        return flux_base + m1 * width1 + m2 * (t_within_period - width1)


######################################################################
# input parameters
######################################################################
#F = 0.1 and T = 20

t_rampup = 2.1
F = 0.004
t_max = 18.5

# arrays
dt = 0.001
t = np.arange(0,25,dt)
factor = np.zeros(len(t))
quot = 5
rel  = 1/quot
broad_low = 1
broad_high = rel*broad_low

for i, t_i in enumerate(t):
    if t_i < t_max:
        factor[i] = np.exp(t[i]/t_rampup)
    else:
        factor[i] = 0.0

Q = F*factor
ax.plot(t,Q,label=f"non-linear",c="C02",lw=1)

t_rampup = 9
F = 1.2

Q_ramp = F*np.where(t/t_rampup < 1.0, t/t_rampup,1) 

######################################################################
# Final layout settings
######################################################################


# Final adjustments to the figure
plotlib.set_box(ax)

ax.set_xlabel(r'$t$',labelpad=-1.5)
ax.set_ylabel(r'$Q$')

ax.set_xlim([0,25])
ax.set_ylim([-5,30])

leg = plotlib.set_legend(ax,pos=4)
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(2))

plotlib.set_position(ax, x=0.22, y=0.33, width=.70, height=0.60)

# Save the figure
fig.savefig('./function_nonlinear.pdf')

plt.show()
