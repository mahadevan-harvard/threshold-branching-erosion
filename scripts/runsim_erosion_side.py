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

import time
import os
from datetime import datetime

class BoundaryFluxSide:
	def __init__(self, F, T, a, b, config):
		"""
		Initialize the ramp-up boundary flux class.
		
		Parameters:
		- t_rampup: Ramp-up time constant
		- q_max_lh: Maximum flux on the left boundary
		- q_max_rh: Maximum flux on the right boundary
		- config: Configuration object with `n` (grid size) and other parameters
		- a: Power factor for ramp-up scaling
		"""
		bx = config.grid_spacing*config.nx
		q_max_lh = F/(2*config.epsilon_lh)
		q_max_rh = F/(2*config.epsilon_rh)
		q_max_side = F/(2*bx)

		self.t_rampup = T
		self.q_max_lh = q_max_lh
		self.q_max_rh = q_max_rh
		self.q_max_side = q_max_side

		self.config = config
		self.a = a

		self.dq5 = np.zeros((self.config.nx + 1, self.config.ny + 1))  # No flux through left boundary outside the source
		self.dq6 = np.zeros((self.config.nx + 1, self.config.ny + 1))  # No flux through right boundary outside the sink 

		self.b = b

	def __call__(self, t):
		"""
		Compute the boundary fluxes for a given time t.

		Parameters:
		- t: Current simulation time

		Returns:
		- dq1, dq2, dq3, dq4: Boundary fluxes for top, bottom, left, and right
		"""
		if t/self.t_rampup < self.a:
			factor_in = t/self.t_rampup
			factor_out = self.b*factor_in
			factor_out_side = (1-self.b)*factor_in
		elif t/self.t_rampup < 2*self.a:
			factor_in = t/self.t_rampup			
			factor_out = self.b*factor_in + (1 - self.b)*(factor_in - self.a) + (1 - self.b)*(t/self.t_rampup - self.a) #
			factor_out_side = (1-self.b)*self.a - (1 - self.b)*(t/self.t_rampup - self.a) #ramp-down the sides
		elif t/self.t_rampup < 1.0:	
			factor_in = t/self.t_rampup			
			factor_out = factor_in
			factor_out_side = 0.0 #(1-self.b)*factor_in - self.b*(t/(self.t_rampup*self.a) - 1)*factor_in												
		else:
			factor_in = 1.0
			factor_out = 1.0			
			factor_out_side = 0.0

		# Boundary flux arrays
		dq1 = -self.q_max_side * factor_out_side * np.ones((self.config.nx + 1, self.config.ny + 1))  # Flux through bottom boundary
		dq2 = -self.q_max_side * factor_out_side * np.ones((self.config.nx + 1, self.config.ny + 1))  # Flux through top boundary
		dq3 = +self.q_max_lh * factor_in * np.ones((self.config.nx + 1, self.config.ny + 1))  # Left boundary flux
		dq4 = -self.q_max_rh * factor_out * np.ones((self.config.nx + 1, self.config.ny + 1))  # Right boundary flux

		# print(self.q_max_side * factor_out_side * 2 *  bx)
		# print(self.q_max_rh * factor_out * 2 * epsilon_rh)
		# print(self.q_max_side * factor_out_side * 2 *  bx + self.q_max_rh * factor_out * 2 * epsilon_rh)
		# print(self.q_max_lh * factor_in * 2 *  epsilon_lh)
		# print("\n")

		return dq1, dq2, dq3, dq4, self.dq5, self.dq6


#########################################################################################
#   MAIN BODY
#########################################################################################

if __name__ == "__main__":

	# PARAMETERS
	nx = 250                 # Specify grid size to be n x n. # Might want to change to nx and ny
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
	varphi_star = 0.8          # threshold transition point -> Is this not per definition the same as phi_0 for a stable config?
	
	t_final = 25           # Simulation time
	save_dt = 0.125

	# Sweep parameters
	F_array = np.array([1.0])	# Flux
	T_array = np.array([10])	# Ramp-up time
	a_array = np.array([0.15])	# non-linear parameter
	b_array = np.array([0])

	# STORAGE
	# Base output folder
	base_data_folder = "DATA"
	os.makedirs(base_data_folder, exist_ok=True)

	# Timestamped folder
	timestamp = datetime.now().strftime("%Y%m%d%H%M")
	folder_name = f"side_{timestamp}" 
	sweep_folder = os.path.join(base_data_folder, folder_name)
	os.makedirs(sweep_folder, exist_ok=True)

	# INITIALIZE DOMAIN AND BOUNDARIES
	domain = dx.mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([bx, by])], [nx, ny], dx.mesh.CellType.quadrilateral)
	V = dx.fem.functionspace(domain, ("Lagrange", 1))

	# Generate indices for transformation from 2D nump array to vector -> must be possible to do this in a more transpartent way 
	coords = domain.geometry.x[:,:2]
	indices = np.lexsort((coords[:,1], coords[:,0]))  

	s = fb.generate_s(V, nx, ny, indices)	# flux in the bulk
	ds = fb.generate_ds(domain, bx, by, epsilon_lh, epsilon_rh) # boundary flux

	config = fb.FEMConfig(V, s, ds, nx, ny, grid_spacing, indices, epsilon_lh, epsilon_rh)

	# INITIALIZE PHI
	seed = 42
	phi0 = fb.generate_phi0(phi_0, sigma_phi, zeta, config, seed=seed)

	# PARAMETER_SWEEP
	for F in F_array:
		for T in T_array:
			for a in a_array:
				for b in b_array:
					print(f"\nrunning simulations with F={F}, T={T}, and a={a} ...\n")

					# Generate a subfolder for the current parameter combination
					param_folder = os.path.join(sweep_folder, f"F_{F}_T_{T}_a_{a}_b_{b}")
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
					boundary_function  = BoundaryFluxSide(F, T, a, b, config)

					# run simulation
					log_file = os.path.join(param_folder, "simulation_log.txt")
					phi, p, times, phi_array, flux_array = fb.run_sim(phi0, t_final, xi, omega, varphi_star, config, boundary_function, log_file, save_dt=save_dt, dt_max=save_dt)

					# SAVE DATA
					data_file = os.path.join(param_folder, "data.npz")
					np.savez(data_file, phi=phi_array, flux=flux_array)