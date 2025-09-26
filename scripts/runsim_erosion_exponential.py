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
from datetime import datetime

class NonlinearFluxRamp:
	def __init__(self, F, T, a, config):
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
		self.t_max = a

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
		if t < self.t_max:
			factor = np.exp(t/self.t_rampup)
		else:
			factor = 0.0

		# Boundary flux arrays
		dq3 = +self.q_max_lh * factor * np.ones((self.config.nx + 1, self.config.ny + 1))  # Left boundary flux
		dq4 = -self.q_max_rh * factor * np.ones((self.config.nx + 1, self.config.ny + 1))  # Right boundary flux

		# check if factor was meant to go from 0 to 1

		return self.dq1, self.dq2, dq3, dq4, self.dq5, self.dq6 

#########################################################################################
#   MAIN BODY
#########################################################################################

if __name__ == "__main__":

	# PARAMETERS
	nx = 250                 # Specify grid size to be n x n.
	bx = 10                 # Boundary length of quadratic domain, so the domain will have shape [0,bx] x [0,bx].
	grid_spacing = bx/nx

	ny = nx
	by = bx

	epsilon_lh = 0.2        # half-width source
	epsilon_rh = 5          # half-width sink

	phi_0 = 0.8             # mean of the Gaussian noise
	sigma_phi = 0.02        # variance of the Gaussian noise
	zeta = 0.15             # correlation length (structural)

	xi = 0.025               # communication length (mechancics)
	omega = 8               # threshold sharpness
	varphi_star = 0.8          # threshold transition point
	
	t_final = 25           # Simulation time
	save_dt = 0.125

	# Sweep parameters
	F_array = np.array([0.004])	# Flux
	T_array = np.array([2.1])	# Ramp-up time
	a_array = np.array([18.5])	# non-linear parameter

	# STORAGE
	# Base output folder
	base_data_folder = "DATA"
	os.makedirs(base_data_folder, exist_ok=True)

	# Timestamped folder
	timestamp = datetime.now().strftime("%Y%m%d%H%M")
	folder_name = f"exponeital_{timestamp}" 
	sweep_folder = os.path.join(base_data_folder, folder_name)
	os.makedirs(sweep_folder, exist_ok=True)

	# INITIALIZE DOMAIN AND BOUNDARIES
	domain = dx.mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([bx, by])], [nx, ny], dx.mesh.CellType.quadrilateral)
	V = dx.fem.functionspace(domain, ("Lagrange", 1))

	# Generate indices for transformation from 2D nump array to vector
	coords = domain.geometry.x[:,:2]
	indices = np.lexsort((coords[:,1], coords[:,0]))  

	s = fb.generate_s(V, nx, ny, indices)	# flux in the bulk
	ds = fb.generate_ds(domain, bx, by, epsilon_lh, epsilon_rh) # boundary flux

	config = fb.FEMConfig(V, s, ds, nx, ny, grid_spacing, indices, epsilon_lh, epsilon_rh)

	# INITIALIZE PHI
	seed = 486522830 #int(time.time() * 1e6) % (2**32) # Either hard-code or based on current time
	phi0 = fb.generate_phi0(phi_0, sigma_phi, zeta, config, seed=seed)

	# PARAMETER_SWEEP
	for F in F_array:
		for T in T_array:
			for a in a_array:

				print(f"\nrunning simulations with F={F}, T={T}, and a={a} ...\n")

				# Generate a subfolder for the current parameter combination
				param_folder = os.path.join(sweep_folder, f"F_{F}_T_{T}_a_{a}_b_{0}")
				os.makedirs(param_folder, exist_ok=True)

				# Save parameters to a text file
				param_file = os.path.join(param_folder, "parameters.txt")
				with open(param_file, "w") as f:
					f.write(f"F = {F}\n")
					f.write(f"T = {T}\n")
					f.write(f"a = {a}\n")
					f.write(f"nx = {nx}\n")
					f.write(f"ny = {ny}\n")		
					f.write(f"bx = {bx}\n")
					f.write(f"by = {by}\n")
					f.write(f"epsilon_lh = {epsilon_lh}\n")
					f.write(f"epsilon_rh = {epsilon_rh}\n")									
					f.write(f"phi_0 = {phi_0}\n")
					f.write(f"sigma_phi = {sigma_phi}\n")
					f.write(f"zeta = {zeta}\n")
					f.write(f"seed = {seed}\n")						
					f.write(f"xi = {xi}\n")
					f.write(f"omega = {omega}\n")
					f.write(f"varphi_star = {varphi_star}\n")
					f.write(f"t_final = {t_final}\n")			

				# specify boundary conditions
				boundary_function  = NonlinearFluxRamp(F, T, a, config)

				# run simulation
				log_file = os.path.join(param_folder, "simulation_log.txt")
				phi, p, times, phi_array, flux_array = fb.run_sim(phi0, t_final, xi, omega, varphi_star, config, boundary_function, log_file,save_dt=save_dt, dt_max=save_dt)

				# SAVE DATA
				data_file = os.path.join(param_folder, "data.npz")
				np.savez(data_file, phi=phi_array, flux=flux_array)