import numpy as np

from sklearn.decomposition import PCA

from scipy.spatial import cKDTree

import matplotlib.pyplot as plt

from matplotlib.ticker import (MultipleLocator)


import PlotLibrary as plotlib

figSpecs = plotlib.FigureSettings()
figSpecs.set_journal('PhysicalReview')
figSpecs.set_figureHeight(55)

# What is the size of this figure relative to the journal specs
xFraction = 1/3 #1.0
yFraction = 1.0

width = xFraction*figSpecs.doubleColumn
height = yFraction*figSpecs.figureHeight

cmap = plt.get_cmap('viridis')

# Figure Settings
fig = plt.figure(figsize=(width,height))
ax = plt.subplot()

##############################################################################################
# FUNCTIONS 
##############################################################################################

def dominates(p, q, tol=1e-8):
	"""Return True if p dominates q (maximize all objectives) with tolerance."""
	# weakly better in all objectives:
	weakly_better = np.all(p >= q - tol)
	# strictly better in at least one objective:
	strictly_better = np.any(p > q + tol)

	return weakly_better and strictly_better


def simple_cull_with_mask(Y, tol=1e-8):
	n = len(Y)
	is_pareto = np.ones(n, dtype=bool)
	P = []       # Pareto point values
	P_idx = []   # Their original indices

	for idx, y in enumerate(Y):
		to_remove = []
		dominated = False
		for i, d in enumerate(P):
			if dominates(y, d, tol):
				to_remove.append(i)
			elif dominates(d, y, tol):
				dominated = True
				break
		if dominated:
			is_pareto[idx] = False   # <-- important
		else:
			for i in reversed(to_remove):
				is_pareto[P_idx[i]] = False
				P.pop(i)
				P_idx.pop(i)
			P.append(y)
			P_idx.append(idx)

	return np.array(P), is_pareto
##############################################################################################
# Retrieve data
##############################################################################################

# Input parameters
#Folder = "linsweep_202507281142"
dataPath = f"./Results/all_simulations_folder.txt"

data = np.genfromtxt(dataPath,skip_header=1)
F = data[:,0]
T = data[:,1]
A = data[:,3]
R = data[:,4]
E = data[:,5]

# Scale the objectives 
# see if I can define them so they get minimized)
# Scale to 0 mean and unit variance
A_scaled = (A - A.mean()) / A.std()
R_scaled = (R - R.mean()) / R.std()
E_scaled = (E - E.mean()) / E.std()
F_scaled = (F - np.min(F))/(np.max(F)-np.min(F))

# Stack objectives into one array (rows are points, columns are objectives)
Y = np.column_stack([A_scaled, R_scaled, E_scaled])
pareto_points, is_pareto = simple_cull_with_mask(Y)

# Assume `pareto_points` is (n_points, 3)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(Y[is_pareto])

# Step 2: Project all points using that PCA basis
Y_pca_from_pareto = pca.transform(Y)

# Step 3 (optional): Extract PC1
pc1 = Y_pca_from_pareto[:, 0]

# Full dataset and front in objective space
tree = cKDTree(Y[is_pareto])  # Pareto points in 3D
d_p,_ = tree.query(Y)  # Distance of all points to closest Pareto point
d_p_norm = d_p / np.mean(d_p)

#Folder = "linsweep_202507281142"
dataPath = f"./Results/all_simulations_side.txt"

data = np.genfromtxt(dataPath,skip_header=1, ndmin=2)
F_side = data[:,0]
T_side = data[:,1]
A_side = data[:,3]
R_side = data[:,4]
E_side = data[:,5]

A_side_scaled = (A_side - A.mean()) / A.std()
R_side_scaled = (R_side - R.mean()) / R.std()
E_side_scaled = (E_side - E.mean()) / E.std()

#Folder = "linsweep_202507281142"
dataPath = f"./Results/all_simulations_exp.txt"

data = np.genfromtxt(dataPath,skip_header=1, ndmin=2)
F_exp = data[:,0]
T_exp = data[:,1]
A_exp = data[:,3]
R_exp = data[:,4]
E_exp = data[:,5]

A_exp_scaled = (A_exp - A.mean()) / A.std()
R_exp_scaled = (R_exp - R.mean()) / R.std()
E_exp_scaled = (E_exp - E.mean()) / E.std()

# Input parameters
dataPath = f"./Results/all_simulations_single.txt"

data = np.genfromtxt(dataPath,skip_header=1)
F_org = data[:,0]
T_org = data[:,1]
A_org = data[:,3]
R_org = data[:,4]
E_org = data[:,5]

# Scale the objectives 
A_org_scaled = (A_org - A.mean()) / A.std()
R_org_scaled = (R_org - R.mean()) / R.std()
E_org_scaled = (E_org - E.mean()) / E.std()

##########################################
# Final Lay-out
##########################################

plotlib.set_box(ax)
plotlib.set_position(ax,x=0.15, y=0.15, width=.80, height=0.80)

# leg = plotlib.set_legend(ax,pos=2)

mode = "AR" # "AE", "RE"
high_idx = 46 # index of high robustness simulation
low_idx = 220 # index of low robustness simulation

logT = np.log10(T)
scaled_T = (logT - np.min(logT))/(np.max(logT)-np.min(logT))

logF = np.log10(F)
scaled_F = (logF - np.min(logF))/(np.max(logF)-np.min(logF))

colors = np.zeros([len(logT),3])
colors[:,1] = scaled_T
colors[:,2] = scaled_F
#colors[is_pareto,2] = 1.0

if mode == "AR":
	ax.scatter(A_scaled[high_idx], R_scaled[high_idx], facecolor='darkorange', marker="^",alpha=1,zorder=10, linewidth=1,color="k",s=20)
	ax.scatter(A_scaled[low_idx], R_scaled[low_idx], facecolor='royalblue',marker="o",alpha=1,zorder=10, linewidth=1,color="k",s=20)

	ax.scatter(A_scaled, R_scaled, color=colors,s=15)
	ax.scatter(A_scaled[is_pareto], R_scaled[is_pareto], color='firebrick', linewidth=1,s=2)


	ax.scatter(A_side_scaled, R_side_scaled,color='black', marker="*",s=25,facecolor="grey",alpha=1.0)
	ax.scatter(A_exp_scaled, R_exp_scaled, color='black', marker="H",s=20,facecolor="grey",alpha=1.0)

	ax.set_xlabel(r"$(A - \langle A \rangle) / \sigma_A$")
	ax.set_ylabel(r"$(R - \langle R \rangle) / \sigma_R$")

	ax.set_xlim(-2.5,2.5)
	ax.set_ylim(-2.5,2.5)

	ax.xaxis.set_major_locator(MultipleLocator(1))
	ax.xaxis.set_minor_locator(MultipleLocator(.2))
	ax.yaxis.set_major_locator(MultipleLocator(1.0))
	ax.yaxis.set_minor_locator(MultipleLocator(0.2))

	plt.savefig("functionality_2Dproject_AR.pdf",dpi=600)
	plt.show()

elif mode == "AE":
	ax.scatter(A_scaled[high_idx], E_scaled[high_idx], facecolor='darkorange', marker="^",alpha=1,zorder=10, linewidth=1,color="k",s=20)
	ax.scatter(A_scaled[low_idx], E_scaled[low_idx], facecolor='royalblue',marker="o",alpha=1,zorder=10, linewidth=1,color="k",s=20)

	ax.scatter(A_scaled, E_scaled, color=colors,s=15)
	ax.scatter(A_scaled[is_pareto], E_scaled[is_pareto], color='firebrick', linewidth=1,s=2)

	ax.scatter(A_side_scaled, E_side_scaled,color='black', marker="*",s=25,facecolor="grey",alpha=1.0)
	ax.scatter(A_exp_scaled, E_exp_scaled, color='black', marker="H",s=20,facecolor="grey",alpha=1.0)

	ax.set_xlabel(r"$(A - \langle A \rangle) / \sigma_A$")
	ax.set_ylabel(r"$(E - \langle E \rangle) / \sigma_E$")

	ax.set_xlim(-2.5,2.5)
	ax.set_ylim(-1.5,3.5)

	ax.xaxis.set_major_locator(MultipleLocator(1))
	ax.xaxis.set_minor_locator(MultipleLocator(.2))
	ax.yaxis.set_major_locator(MultipleLocator(1.0))
	ax.yaxis.set_minor_locator(MultipleLocator(0.2))

	plt.savefig("functionality_2Dproject_AE.pdf",dpi=600)
	plt.show()

elif mode == "RE":
	ax.scatter(R_scaled[high_idx], E_scaled[high_idx], facecolor='darkorange', marker="^",alpha=1,zorder=10, linewidth=1,color="k",s=20)
	ax.scatter(R_scaled[low_idx], E_scaled[low_idx], facecolor='royalblue',marker="o",alpha=1,zorder=10, linewidth=1,color="k",s=20)

	ax.scatter(R_scaled, E_scaled, color=colors,s=15)
	ax.scatter(R_scaled[is_pareto], E_scaled[is_pareto], color='firebrick', linewidth=1,s=2)

	ax.scatter(R_side_scaled, E_side_scaled,color='black', marker="*",s=25,facecolor="grey",alpha=1.0)
	ax.scatter(R_exp_scaled, E_exp_scaled, color='black', marker="H",s=20,facecolor="grey",alpha=1.0)

	ax.set_xlabel(r"$(R - \langle R \rangle) / \sigma_R$")
	ax.set_ylabel(r"$(E - \langle E \rangle) / \sigma_E$")

	ax.set_xlim(-2.5,2.5)
	ax.set_ylim(-1.5,3.5)

	ax.xaxis.set_major_locator(MultipleLocator(1))
	ax.xaxis.set_minor_locator(MultipleLocator(.2))
	ax.yaxis.set_major_locator(MultipleLocator(1.0))
	ax.yaxis.set_minor_locator(MultipleLocator(0.2))

	plt.savefig("functionality_2Dproject_RE.pdf",dpi=600)
	plt.show()

elif mode == "QT":
	ax.scatter(F[high_idx], T[high_idx], facecolor='darkorange', marker="^",alpha=1,zorder=10, linewidth=1,color="k",s=20)
	ax.scatter(F[low_idx], T[low_idx], facecolor='royalblue',marker="o",alpha=1,zorder=10, linewidth=1,color="k",s=20)

	ax.scatter(F, T, color=colors,s=15)

	ax.scatter(F[is_pareto], T[is_pareto], color='firebrick', linewidth=1,s=2)

	ax.set_xlabel(r"$Q$")
	ax.set_ylabel(r"$T$")

	ax.set_xscale("log")
	ax.set_yscale("log")	
	ax.set_xlim(3e-2,7e0)
	ax.set_ylim(7e-1,1.5e2)

	plt.savefig("functionality_2Dproject_QT.pdf",dpi=600)
	plt.show()