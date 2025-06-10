from __future__ import absolute_import, division, print_function

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
class FigureSettings:
	""" 
	Class used to store all fixed properties of a figure.
	"""
	mmPerInch = 25.4

	def __init__(self):
		self.singleColumn = 2.0
		self.doubleColumn = 4.0

	def set_journal(self,journalName):
		if journalName == 'Nature':
			# Sizes
			singleWidth = 89.00 # [mm]
			doubleWidth = 183.00 # [mm]

			# Specifications for generic text	 
			font = {'family' : 'sans-serif',
    	  		    'serif' : 'Helvetica',
    	    		'size'   : 10
				   }

			# Specifications for latex-based text 
			# set the normal latex text font here # load up the sansmath so that math -> helvet	
			# Note: amsmath does not have cursive sub/superscript. (see plot)
			preambleSpecs = r'\usepackage{arev} \usepackage{amsmath}'

		elif journalName == 'PhysicalReview':
			# Sizes
			singleWidth = 86.00 # [mm]
			doubleWidth = 172.00 # [mm]

			# Specifications for generic text	 
			font = {'family' : 'serif',
    	  		    'serif' : 'Times New Roman',
    	    		'size'   : 8
				   }

			# Specifications for latex-based text 
			# Specifications for latex-based text 
			# set the normal latex text font here # load up the sansmath so that math -> helvet	
			# Note: amsmath does not have cursive sub/superscript. (see plot)
			preambleSpecs = r'\usepackage{newtxtext} \usepackage{newtxmath} \usepackage{amsmath}'

			mpl.rcParams['xtick.major.pad']='4'	
			mpl.rcParams['xtick.minor.pad']='4'	
			mpl.rcParams['ytick.major.pad']='1'	
			mpl.rcParams['ytick.minor.pad']='1'	
			mpl.rcParams['axes.labelpad']='1'
			mpl.rcParams['legend.fontsize']='6'	
			mpl.rcParams['legend.title_fontsize']='6'	
			mpl.rcParams['legend.labelspacing']='0.3'
			mpl.rcParams["axes.axisbelow"] = False
			mpl.rcParams['legend.borderaxespad']='1'

		elif journalName == 'SoftMatter':
			# Sizes
			singleWidth = 83.00 # [mm]
			doubleWidth = 171.00 # [mm]

			# Specifications for generic text	 
			font = {'family' : 'sans-serif',
    	  		    'serif' : 'Helvetica',
    	    		'size'   : 8
				   }

			# Specifications for latex-based text 
			preambleSpecs =  [
						 	  r'\usepackage{arev}',    # set the normal latex text font here
						 	  r'\usepackage{amsmath}',  # load up the sansmath so that math -> helvet
							]  							# Note: amsmath does not have cursive sub/superscript. (see plot)

			mpl.rcParams['xtick.major.pad']='4'	
			mpl.rcParams['xtick.minor.pad']='4'	
			mpl.rcParams['ytick.major.pad']='1'	
			mpl.rcParams['ytick.minor.pad']='1'	
			mpl.rcParams['axes.labelpad']='1'
			mpl.rcParams['legend.fontsize']='6'
			mpl.rcParams['legend.title_fontsize']='6'	
			mpl.rcParams['legend.labelspacing']='0.3'
			mpl.rcParams["axes.axisbelow"] = False	

		elif journalName == 'Thesis':
			# Sizes
			singleWidth = 125.00/2.0 # [mm]
			doubleWidth = 125.00 # [mm]

			# Specifications for generic text	 
			font = {'family' : 'sans-serif',
    	  		    'serif' : 'Helvetica',
    	    		'size'   : 8
				   }

			# Specifications for latex-based text 
			preambleSpecs =  [
						 	  r'\usepackage{arev}',    # set the normal latex text font here
						 	  r'\usepackage{amsmath}',  # load up the sansmath so that math -> helvet
							]  							# Note: amsmath does not have cursive sub/superscript. (see plot)

			mpl.rcParams['xtick.major.pad']='4'	
			mpl.rcParams['xtick.minor.pad']='4'	
			mpl.rcParams['ytick.major.pad']='1'	
			mpl.rcParams['ytick.minor.pad']='1'	
			mpl.rcParams['axes.labelpad']='1'
			mpl.rcParams['legend.fontsize']='6'	
			mpl.rcParams['legend.title_fontsize']='6'	
			mpl.rcParams['legend.labelspacing']='0.3'
			mpl.rcParams["axes.axisbelow"] = False
			mpl.rcParams['legend.borderaxespad']='1'
		else:
			print("The settings for this journal are not specified.")
			return

		# Set the sizes in Inches
		self.singleColumn = singleWidth/self.mmPerInch
		self.doubleColumn = doubleWidth/self.mmPerInch

		# Fix the fonts and turn on latex formatting
		# !!!! Note I am still not sure how well this works.
		mpl.rcParams['ps.usedistiller']='xpdf'
		plt.rc('text', usetex=True)
		plt.rc('font', **font)
		plt.rc('text.latex', preamble=preambleSpecs)

	def set_figureHeight(self,height,mode='mm'):
		if mode=='mm':
			self.figureHeight = height/self.mmPerInch					
		elif mode=='inch':
			self.figureHeight = height

def set_box(ax):
	""" 
	Format the box around the graph
	"""
	ax.tick_params('both', left=True,right=True,bottom=True,top=True,direction='in',width=0.5,length=6,which='major')
	ax.tick_params('both', left=True,right=True,bottom=True,top=True,direction='in',width=0.5,length=3,which='minor')
	for axis in ['top','bottom','left','right']:
	    ax.spines[axis].set_linewidth(0.5)

def set_legend(ax,pos=1,bbox=None,htextpad=1.0):
	""" 
	Set and format legend
	"""
	if bbox != None:
		legend = ax.legend(frameon=False,loc=pos,handletextpad=htextpad,bbox_to_anchor=bbox,handlelength=0.6)
	else:
		legend = ax.legend(frameon=False,loc=pos,handletextpad=htextpad,handlelength=0.6)
	legend.get_frame().set_facecolor('none')
	return legend

def adjust_position(ax,x_add=0,y_add=0,width_add=0,height_add=0):
	"""
	Make adjustments to the placement of any axes object.
	"""
	pos = ax.get_position() # get the original position 
	pos_new = [pos.x0+x_add, pos.y0+y_add,  pos.width+width_add, pos.height+height_add] 
	ax.set_position(pos_new) # set a new position

def set_position(ax,x=0,y=0,width=0,height=0):
	"""
	Make adjustments to the placement of any axes object.
	"""
	pos_new = [x, y,  width, height] 
	ax.set_position(pos_new) # set a new position

# Define function for string formatting of scientific notation
def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if exponent is None:
        exponent = int(np.floor(np.log10(abs(num))))
    coeff = np.round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    #return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)
    return coeff, exponent
