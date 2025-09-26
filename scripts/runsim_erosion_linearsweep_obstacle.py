############################
############################
#
#    Code to reproduce the results of the paper "Flow-Driven Branching in a Frangible Porous Medium", by Derr et al.
#    Author: Paolo Fischer
#	 Refactored by: Justin Tauber
#    Link: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.158002
#    DOI: https://doi.org/10.1103/PhysRevLett.125.158002
#
############################
############################

import numpy as np

from mpi4py import MPI
import dolfinx as dx

from erosion import simulate as fb

import os

class BoundaryFlux:
	def __init__(self, F, T, config):
		"""
		Initialize the ramp-up boundary flux class.
		
		Parameters:
		- t_rampup: Ramp-up time constant
		- q_max_lh: Maximum flux on the left boundary
		- q_max_rh: Maximum flux on the right boundary
		- config: Configuration object with `n` (grid size) and other parameters
		- a: Power factor for ramp-up scaling
		"""
		q_max_lh = F/(2*config.epsilon_lh)
		q_max_rh = F/(2*config.epsilon_rh)

		self.t_rampup = T
		self.q_max_lh = q_max_lh
		self.q_max_rh = q_max_rh
		self.config = config

		self.dq1 = np.zeros((self.config.nx + 1, self.config.ny + 1))  # No flux through top boundary
		self.dq2 = np.zeros((self.config.nx + 1, self.config.ny + 1))  # No flux through bottom boundary
		self.dq5 = np.zeros((self.config.nx + 1, self.config.ny + 1))  # No flux through left boundary outside the source
		self.dq6 = np.zeros((self.config.nx + 1, self.config.ny + 1))  # No flux through right boundary outside the sink

	def __call__(self, t):
		"""
		Compute the boundary fluxes for a given time t.

		Parameters:
		- t: Current simulation time

		Returns:
		- dq1, dq2, dq3, dq4: Boundary fluxes for top, bottom, left, and right
		"""
		if t/self.t_rampup < 1.0:
			factor = t/self.t_rampup
		else:
			factor = 1.0 #np.sin(2*np.pi*t/self.t_rampup)

		# Boundary flux arrays
		dq3 = +self.q_max_lh * factor * np.ones((self.config.nx + 1, self.config.ny + 1))  # Left boundary flux
		dq4 = -self.q_max_rh * factor * np.ones((self.config.nx + 1, self.config.ny + 1))  # Right boundary flux

		return self.dq1, self.dq2, dq3, dq4, self.dq5, self.dq6

class CenterRectangle:
	def __init__(self, width, height, value=1.0-1e-6, inplace=False, require_square=True):
		self.width = int(width)
		self.height = int(height)
		self.value = value
		self.inplace = inplace
		self.require_square = require_square

	def __call__(self, A):
		if not isinstance(A, np.ndarray) or A.ndim != 2:
			raise ValueError("A must be a 2D numpy array")
		nr, nc = A.shape
		if self.require_square and nr != nc:
			raise ValueError("A must be square when require_square=True")
		if self.width > nc or self.height > nr:
			raise ValueError("Rectangle dimensions exceed array size")

		out = A if self.inplace else A.copy()
		r0 = (nr - self.height) // 2
		c0 = (nc - self.width) // 2
		out[r0:r0 + self.height, c0:c0 + self.width] = self.value
		return out

class CenterCircle:
	def __init__(self, radius, value=1.0-1e-6, inplace=False, center=None, trim=0.5):
		self.radius = int(radius)
		self.value = value
		self.inplace = inplace
		self.center = center  # (i_center, j_center) or None for array center
		self.trim = float(trim)

	def __call__(self, A):
		if not isinstance(A, np.ndarray) or A.ndim != 2:
			raise ValueError("A must be a 2D numpy array")
		nr, nc = A.shape
		r = self.radius
		if r <= 0:
			return A

		ic, jc = self.center if self.center is not None else (nr // 2, nc // 2)

		i0, i1 = max(0, ic - r), min(nr, ic + r + 1)
		j0, j1 = max(0, jc - r), min(nc, jc + r + 1)

		out = A if self.inplace else A.copy()

		ii = np.arange(i0, i1)[:, None]
		jj = np.arange(j0, j1)[None, :]
		ds2 = (ii - ic)**2 + (jj - jc)**2

		mask = ds2 <= r*r
		if self.trim > 0:
			rt = max(0.0, r - self.trim)
			mask &= ds2 < rt*rt

		out[i0:i1, j0:j1][mask] = self.value
		return out

#########################################################################################
#   MAIN BODY
#########################################################################################

if __name__ == "__main__":

	# BASE FOLDER
	Folder = "linsweep_202509081026"
	dataPath = f"./DATA/{Folder}/" 

	# Sweep parameters
	F = 0.3	# Flux
	T = 10.0	# Ramp-up time
	V = 1

	obstacle_type = "rectangle"
	obstacle_width = 10
	obstacle_height = 200

	# EXTRACT PHI
	subFolder = f"F_{F}_T_{T}_V_{V:03d}"
	input_file = f"{dataPath}{subFolder}/data.npz"
	data = np.load(input_file)
	phi_previous = data["phi"][-1]

	# PARAMETERS
	nx = 250                 # Specify grid size to be n x n.
	bx = 10                 # Boundary length of quadratic domain, so the domain will have shape [0,bx] x [0,bx].
	grid_spacing = bx/nx

	ny = nx
	by = bx

	epsilon_lh = 0.2        # half-width source
	epsilon_rh = 5.0          # half-width sink

	xi = 0.025               # communication length (mechancics)
	omega = 8               # threshold sharpness
	varphi_star = 0.8          # threshold transition point
	t_final = 2000           # Simulation time
	save_dt = 250

	# STORAGE

	# INITIALIZE DOMAIN AND BOUNDARIES
	domain = dx.mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([bx, by])], [nx, ny], dx.mesh.CellType.quadrilateral)
	V = dx.fem.functionspace(domain, ("Lagrange", 1))

	# Generate indices for transformation from 2D nump array to vector
	coords = domain.geometry.x[:,:2]
	indices = np.lexsort((coords[:,1], coords[:,0]))  

	s = fb.generate_s(V, nx, ny, indices)	# flux in the bulk
	ds = fb.generate_ds(domain, bx, by, epsilon_lh, epsilon_rh) # boundary flux

	config = fb.FEMConfig(V, s, ds, nx, ny, grid_spacing, indices, epsilon_lh, epsilon_rh)

	# Import pre-existing phi field
	phi0 = np.copy(phi_previous) 

	# add obstacle
	setter_function = CenterRectangle(width=obstacle_width, height=obstacle_height)
	phi0 = setter_function(phi0)

	# Start simulation
	print(f"\nrunning simulations with F={F}, T={T} ...\n")

	# Generate a subfolder for the current parameter combination
	param_folder = os.path.join(dataPath, subFolder)

	# Save parameters to a text file
	param_file = os.path.join(param_folder, "parameters_obstacle.txt")
	with open(param_file, "w") as f:
		f.write(f"F = {F}\n")
		f.write(f"T = {T}\n")
		f.write(f"nx = {nx}\n")
		f.write(f"ny = {ny}\n")		
		f.write(f"bx = {bx}\n")
		f.write(f"by = {by}\n")						
		f.write(f"epsilon_lh = {epsilon_lh}\n")
		f.write(f"epsilon_rh = {epsilon_rh}\n")		
		f.write(f"xi = {xi}\n")
		f.write(f"omega = {omega}\n")
		f.write(f"varphi_star = {varphi_star}\n")
		f.write(f"t_final = {t_final}\n")			
		f.write(f"obstacle_type = {obstacle_type}\n")	
		f.write(f"obstacle_width = {obstacle_width}\n")	
		f.write(f"obstacle_height = {obstacle_height}\n")	
	# specify boundary conditions
	boundary_function  = BoundaryFlux(F, T, config)

	# run simulation
	log_file = os.path.join(param_folder, "simulation_log_obstacle.txt")
	phi, p, times, phi_array, flux_array = fb.run_sim(phi0, t_final, xi, omega, varphi_star, config, boundary_function, log_file,save_dt=save_dt, dt_max=save_dt, setter_function=setter_function)

	# SAVE DATA
	data_file = os.path.join(param_folder, "data_obstacle.npz")
	np.savez(data_file, phi=phi_array, flux=flux_array)