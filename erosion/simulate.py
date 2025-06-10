##########################################################################################
#
#    Code to reproduce the results of the paper "Flow-Driven Branching in a Frangible 
#    Porous Medium", by Derr et al.
#    Original implementation: Paolo Fischer
#	 Refactored: Justin Tauber
#
#    DOI: https://doi.org/10.1103/PhysRevLett.125.158002
#
#	Purpose: Simulates erosion-driven branching in porous media using FEM with 
# 	adaptive timestepping.
#
#	Notes:
#   The solver in fenicsx (a wrapper around PETSc) works with a 1D array for the values of 
#   phi on our 2D grid. On the other hand when we do things in numpy we want to work with a 
#   2D array (for the convolution for example).
#
#	Dependencies:
#	- numpy, scipy, mpi4py, dolfinx, ufl
#
#########################################################################################


import time
import os

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import erf

from mpi4py import MPI
import dolfinx as dx
import ufl
from dolfinx.fem.petsc import LinearProblem

from dolfinx.fem import form, set_bc
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from petsc4py import PETSc
from dolfinx import fem

#########################################################################################
#   Functions
#########################################################################################

class FEMConfig:
	"""
	Configuration container for FEM simulation setup.
	
	Holds mesh parameters, boundary conditions, and FEM-specific objects.
	Used to pass simulation state between functions.
	"""
	def __init__(self, V, s, ds, nx, ny, grid_spacing, indices, epsilon_lh, epsilon_rh, pressure_sink=False):
		# Mesh and discretization parameters
		self.V = V
		self.nx = nx
		self.ny = ny		
		self.grid_spacing = grid_spacing
		self.indices = indices

		# Source terms and boundary conditions
		self.s = s
		self.ds = ds
		self.epsilon_lh = epsilon_lh
		self.epsilon_rh = epsilon_rh
		self.pressure_sink = pressure_sink

		# FEM-specific objects (initialized later)
		self.u = None
		self.v = None
		self.phi = None
		self.phi_v = None
		self.bc1 = None
		self.bc2 = None
		self.bc3 = None
		self.bc4 = None
		self.bc5 = None
		self.bc6 = None

# class NeumannProblem:
# 	def __init__(self, solver, A, b, p_sol, nullspace):
# 		self.solver = solver
# 		self.A = A
# 		self.b = b
# 		self.p_sol = p_sol
# 		self.nullspace = nullspace

class NeumannProblem:
	def __init__(self, V, a_form, L_form, nullspace):
		self.A = assemble_matrix(a_form)
		self.A.assemble()
		self.A.setNullSpace(nullspace)

		self.b = assemble_vector(L_form)
		self.p_sol = dx.fem.Function(V)

		self.solver = PETSc.KSP().create(V.mesh.comm)
		self.solver.setOperators(self.A)
		self.solver.setType("cg")
		self.solver.getPC().setType("hypre")
		self.solver.getPC().setHYPREType("boomeramg")
		self.solver.setFromOptions()

		self.nullspace = nullspace

	def update_matrix(self, a_form):
		self.A.zeroEntries()
		assemble_matrix(self.A, a_form)
		self.A.assemble()
		self.A.setNullSpace(self.nullspace)
		self.solver.setOperators(self.A)

	def update_rhs(self, L_form):
		self.b.array[:] = 0.0
		assemble_vector(self.b, L_form)
		self.nullspace.remove(self.b)

	def solve(self):
		self.solver.solve(self.b, self.p_sol.x.petsc_vec)
		self.p_sol.x.scatter_forward()
		return self.p_sol

def generate_phi0(phi_0, sigma_phi, zeta, config, seed=42):
	"""
	Generates initial condition for phi at t = 0 using filtered noise (Eq. 6, erosion paper).

	Applies Gaussian filtering in Fourier space with correlation length zeta.
	"""
	# Fix random seed for reproducibility
	np.random.seed(seed)
	
	# Generate Fourier-space grid (kx, ky)
	kx = 2 * np.pi * np.fft.fftfreq(config.nx, d=config.grid_spacing)
	ky = 2 * np.pi * np.fft.fftfreq(config.ny, d=config.grid_spacing)
	KX, KY = np.meshgrid(kx, ky, indexing='ij')

	# Compute isotropic power spectrum filter
	K_sq = KX**2 + KY**2
	spectrum = np.exp(-0.25 * K_sq * zeta**2)

	# Generate white noise in Fourier space
	white_noise = np.random.normal(0, 1, (config.nx, config.ny)) + 1j * np.random.normal(0, 1, (config.nx, config.ny))

	# Apply the power spectrum to the white noise
	colored_noise_f = white_noise * np.sqrt(spectrum)

	# Inverse Fourier transform to get the correlated noise in real space
	colored_noise = np.fft.ifft2(colored_noise_f).real

	# Normalize the noise
	colored_noise = colored_noise / np.std(colored_noise) * sigma_phi

	# Add the noise to the constant background
	phi = phi_0 + colored_noise
		
	# Make sure that the initial state for phi is in [0,1] everywhere (such that it is physical).
	return np.clip(phi, 0, 1)

def gaussian_feather_mask(dist, radius, sigma):
	"""
	Returns a smooth radial mask using the Gaussian error function.
	"""
	return 0.5 * (1 - erf((dist - radius) / (np.sqrt(2) * sigma)))

def apply_half_circle_patch(phi, config, phi_patch=0.0, radius=1.0, sigma=0.2, side="bottom"):
	"""
	Applies a half-circle patch with a smooth transition (like feathering) to phi.
	Inside radius: phi sim phi_patch
	Outside: phi unchanged
	Transition: smooth interpolation over width sigma
	"""
	nx, ny = config.nx, config.ny
	h = config.grid_spacing

	x = np.linspace(0, nx*h, nx)
	y = np.linspace(0, ny*h, ny)
	X, Y = np.meshgrid(x, y, indexing='ij')

	cx = (nx * h) / 2
	cy = 0 if side == "bottom" else ny * h

	dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

	m = gaussian_feather_mask(dist, radius, sigma)

	# Blend field: smooth transition from phi_patch to phi
	phi = phi * (1 - m) + phi_patch * m

	return phi

def generate_s(V, nx, ny, indices):
	"""
	Generates source term s(x,y) based on FEM function space V.
	Currently set to zero everywhere. (q.3 in the erosion paper)
	"""	
	s_array = np.zeros((nx+1,ny+1))
	s_array = nptodx(s_array, indices)
	s = dx.fem.Function(V)
	s.x.array[:] = s_array
	
	return s

def generate_ds(domain, bx, by, epsilon_lh, epsilon_rh):
	"""
	Defines boundary subdomains for applying flux boundary conditions.

	The top and bottom boundaries are divided into regions:
	- Center regions (width 2 * epsilon_lh / epsilon_rh) where flux is applied.
	- Remaining regions where no flux is applied.
	Left and right vertical boundaries (x = 0 and x = bx) are tagged separately.
	"""
	boundaries = [
		(1, lambda x: np.isclose(x[0], 0)), # Left boundary (x = 0)
		(2, lambda x: np.isclose(x[0], bx)), # Right boundary (x = bx)
		(3, lambda x: np.isclose(x[1], 0) & (bx/2 - epsilon_lh <= x[0]) & (x[0] <= bx/2 + epsilon_lh)), # Bottom inflow
		(4, lambda x: np.isclose(x[1], 0) & ((x[0] < bx/2 - epsilon_lh) | (x[0] > bx/2 + epsilon_lh))), # Bottom no-flux
		(5, lambda x: np.isclose(x[1], by) & (bx/2 - epsilon_rh <= x[0]) & (x[0] <= bx/2 + epsilon_rh)), # Top outflow
		(6, lambda x: np.isclose(x[1], by) & ((x[0] < bx/2 - epsilon_rh) | (x[0] > bx/2 + epsilon_rh))) # Top no-flux
	]

	# Identify boundary facets and assign marker tags
	facet_indices, facet_markers = [], []
	fdim = domain.topology.dim - 1
	for (marker, locator) in boundaries:
		facets = dx.mesh.locate_entities(domain, fdim, locator)
		facet_indices.append(facets)
		facet_markers.append(np.full_like(facets, marker))

	# Flatten and sort facets for consistent indexing		
	facet_indices = np.hstack(facet_indices).astype(np.int32)
	facet_markers = np.hstack(facet_markers).astype(np.int32)
	sorted_facets = np.argsort(facet_indices)

	# Tag facets with subdomain markers for boundary integration
	facet_tag = dx.mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

	ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
	return ds

def generate_B(xi, grid_spacing):
	"""
	This function generates B_mat, which is the kernel with which the porosity phi is convoluted.
	"""
	extent = int(3 * xi / grid_spacing)  # 3 sigma on each side  
	x_range = np.linspace(-extent*grid_spacing,+extent*grid_spacing,2*extent+1)
	term1 = np.exp(-x_range**2/(2*xi**2))
	c_normalize = 1/np.sum(term1)
	term1 = c_normalize * term1
	mat1, mat2 = np.meshgrid(term1,term1)
	B = mat1 * mat2
	B_mat = np.flip(np.flip(B,axis=0),axis=1)  
	return B_mat

def convolve_optimized(A, B_mat, extent):
	"""
	Performs boundary-corrected convolution of array A with kernel B_mat.

	Implements the filtering described in Eqs. 54-58 of the erosion paper (SI), 
	using reflection padding to avoid boundary artifacts.
	"""
	# Pad A with reflected boundaries to prevent artificial erosion effects near edges.
	A_padded = np.pad(A, ((extent, extent), (extent, extent)), mode='reflect')

	# Perform FFT-based convolution; 'valid' accounts for pre-padding.
	output = fftconvolve(A_padded, B_mat, mode='valid')  # 'valid' because we already padded

	return output

def get_phi_v(phi, config):
	"""
	Extrapolates cell-centered phi values to cell corner nodes.

	Required to evaluate FEM terms where DoFs are defined at grid nodes.
	"""
	# Initialize corner node array
	phi_v = np.zeros((config.nx+1, config.ny+1))

	# Interior nodes: average 4 surrounding cell-centered values
	phi_v[1:-1, 1:-1] = (phi[:-1, :-1] + phi[:-1, 1:] + phi[1:, :-1] + phi[1:, 1:]) / 4

	# Edge nodes: average 2 adjacent cell-centered values
	phi_v[0, 1:-1] = (phi[0, :-1] + phi[0, 1:]) / 2  # Top edge
	phi_v[-1, 1:-1] = (phi[-1, :-1] + phi[-1, 1:]) / 2  # Bottom edge
	phi_v[1:-1, 0] = (phi[:-1, 0] + phi[1:, 0]) / 2  # Left edge
	phi_v[1:-1, -1] = (phi[:-1, -1] + phi[1:, -1]) / 2  # Right edge

	# Corner nodes: copy nearest cell-centered value
	phi_v[0, 0] = phi[0, 0]  # Top-left corner
	phi_v[0, -1] = phi[0, -1]  # Top-right corner
	phi_v[-1, 0] = phi[-1, 0]  # Bottom-left corner
	phi_v[-1, -1] = phi[-1, -1]  # Bottom-right corner

	# Convert numpy array to dolfinx function
	return nptodx(phi_v, config.indices)

def fem_initialize(phi, dq1, dq2, dq3, dq4, dq5, dq6, config):
	"""
	Staggered grid FEM formulation (ref: erosion SI p.8):
	- Extrapolates to corner nodes
	- Defines bilinear form (a) and linear form (L).
	- Handles nullspace (pure Neumann).
	- Returns initialized solver components.
	"""	
	# Extrapolate cell-centered phi to nodal DoFs
	config.phi_v = dx.fem.Function(config.V)
	config.phi_v.x.array[:] = get_phi_v(phi, config)

	# Bilinear form: Scaled by permeability kappa(phi)
	a = kappa(config.phi_v) * ufl.dot(ufl.grad(config.u), ufl.grad(config.v)) * ufl.dx 

	# Linear form L (source term), BCs added below
	L = - config.s * config.v * ufl.dx

	# --- Set up boundary flux functions ---
	dqs = [dq1, dq2, dq3, dq4, dq5, dq6]
	bcs = []
	for dq in dqs:
		bc_func = dx.fem.Function(config.V)
		bc_func.x.array[:] = nptodx(dq, config.indices)
		bcs.append(bc_func)

	config.bc1, config.bc2, config.bc3, config.bc4, config.bc5, config.bc6 = bcs

	# Add Neumann boundary conditionss to L
	ds_terms = [config.bc1, config.bc2, config.bc3, config.bc5, config.bc4, config.bc6]
	ds_ids   = [1, 2, 3, 4, 5, 6]

	for bc_func, tag in zip(ds_terms, ds_ids):
		L += bc_func * config.v * config.ds(tag)
			
	# --- Assemble system ---
	a_form = form(a)
	L_form = form(L)
	nullspace = PETSc.NullSpace().create(constant=True)

	problem = NeumannProblem(config.V, a_form, L_form, nullspace)

	return problem, config

def calculate_p(problem, phi, dq1, dq2, dq3, dq4, dq5, dq6, config):
	"""
	Update and solve pressure field:
	- Rebuilds FEM system with updated phi.
	- Updates boundary fluxes.
	- Applies constant nullspace removal.
	"""
	config.phi_v.x.array[:] = get_phi_v(phi, config)

	# --- Rebuild bilinear form with updated phi --
	a = kappa(config.phi_v) * ufl.dot(ufl.grad(config.u), ufl.grad(config.v)) * ufl.dx
	L = -config.s * config.v * ufl.dx

	for dq, bc in zip([dq1, dq2, dq3, dq4], [config.bc1, config.bc2, config.bc3, config.bc4]):
		bc.x.array[:] = nptodx(dq, config.indices)

	for bc, tag in zip(
		[config.bc1, config.bc2, config.bc3, config.bc5, config.bc4, config.bc6],
		[1, 2, 3, 4, 5, 6]):
		L += bc * config.v * config.ds(tag)

	problem.update_matrix(form(a))
	problem.update_rhs(form(L))
	result = problem.solve()

	return result, problem, config

def H(varphi, omega, varphi_star):
	return 1/2 * np.tanh(omega * (varphi-varphi_star)) + 1/2

def psi(conv, omega, varphi_star):
	Hconv = H(conv, omega, varphi_star)
	H0 = H(0, omega, varphi_star)
	H1 = H(1, omega, varphi_star)
	return (Hconv-H0)/(H1-H0)

def kappa(phi):
	"""
	Karman-Kozeny
	"""
	return (1-phi)**3/phi**2

def nptodx(array, indices):
	array_flattened_data = array.flatten()
	array_original_data = np.empty_like(array_flattened_data)
	array_original_data[indices] = array_flattened_data
	return array_original_data

def dxtonp(values, indices, nx, ny):
	"""
	Maps local DoF values back to global 2D numpy array.
	Assumes values[indices] corresponds to global ordering.
	"""	
	sorted_data = values[indices]
	return sorted_data.reshape(nx+1, ny+1)

def get_p(result, config):
	"""
	Extracts pressure field p from FEM result and computes its gradients on cell faces.
	"""
	# Convert FEM vector to 2D numpy array (nodal values)
	p = dxtonp(result.x.array, config.indices, config.nx, config.ny) # convert resulting p from dolfinx format to numpy format.
	
	# Compute dp/dx (x-direction gradients on vertical cell faces)
	dp_dx0 = 0.5 * ((p[1:, :-1] - p[:-1, :-1]) / config.grid_spacing + 
					(p[1:, 1:] - p[:-1, 1:]) / config.grid_spacing)
	
	# Compute dp/dy (y-direction gradients on horizontal cell faces)
	dp_dx1 = 0.5 * ((p[:-1, 1:] - p[:-1, :-1]) / config.grid_spacing + 
				 	(p[1:, 1:] - p[1:, :-1]) / config.grid_spacing)

	return p, dp_dx0, dp_dx1

def run_sim(phi0, t_final, xi, omega, varphi_star, config, boundary_function, log_file, save_dt=1, dt_max=1, e_max=1e-4):
	"""
	Run erosion simulation with adaptive timestep:
	- Advances phi using ReLU-based erosion model.
	- Solves pressure field using FEM each step.
	- Adaptive timestep control based on error e (p.12 of the SI of the erosion paper).
	"""
	# --- Adaptive time-stepping parameters ---
	#e_max = 1E-6 # Error tolerance for timestep control # Used to be 1e-4
	r_dec, r_inc = 0.9, 1.1 # Timestep reduction/increase factors
	r_min, r_max = 1/3, 3 # Timestep bounds
	dt = 0.001  # Initial timestep

	# --- FEM Functionspace initialization ---
	config.u = ufl.TrialFunction(config.V)
	config.v = ufl.TestFunction(config.V)

	# --- Precompute convolution kernel ---
	extent = int(3 * xi / config.grid_spacing)
	B_m = generate_B(xi, config.grid_spacing)

	# --- Simulation state initialization ---
	t, n = 0, 0
	phi = np.copy(phi0)

	dq1, dq2, dq3, dq4, dq5, dq6 = boundary_function(t)
	problem, config = fem_initialize(phi, dq1, dq2, dq3, dq4, dq5, dq6, config)
	
	# --- Prepare data storage & logging ---
	save_count = 0
	save_time = np.arange(0, t_final+save_dt, save_dt)
	Nsave = len(save_time) # np.floor(t_end / dt / Nstep).astype(int) + 1
	(W,L) = phi0.shape
	phi_array = np.zeros([Nsave, W, L])
	flux_array = np.zeros([Nsave, W, L])	
	flux_magnitude = np.zeros([W,L])

	file = open(log_file, "w")
	file.write("Step\tTime\tTimestep\tDelta_p\tPhi\tG2\tQ\tF\tErate\n")

	times = []
	tt1 = time.time()

	# --- Main simulation loop ---
	while t < t_final:		

		# --- Save output at specified times ---
		if save_count<Nsave and t >= save_time[save_count]:
			phi_array[save_count] = phi
			flux_array[save_count] = flux_magnitude
			tt2 = time.time()
			print("Step:", n, dt, t, tt2 -tt1)
			tt1 = time.time()	
			save_count += 1

		# --- Compute pressure field at current time ---
		dq1, dq2, dq3, dq4, dq5, dq6 = boundary_function(t)	
		result, problem, config = calculate_p(problem, phi, dq1, dq2, dq3, dq4, dq5, dq6, config)
		p, dp_dx0, dp_dx1 = get_p(result, config)

		# --- Compute erosion rate f from pressure gradients ---
		varphi = convolve_optimized(phi, B_m, extent)
		gradP2 = dp_dx0**2 + dp_dx1**2
		f = np.maximum(0, gradP2 - psi(varphi, omega, varphi_star))

		# --- Adaptive time-stepping loop ---
		while True:
			# Predictor step for phi (time = t + dt, and time = t + 0.5dt)
			dtf = dt * f
			phi_np1 = phi * (1 - dtf)
			phi_np12 = phi * (1 - 0.5 * dtf)		

			# Reject timestep if phi becomes unphysical
			if np.any(phi_np12 < 0):
				e = 100000
			else:
				# Recompute pressure at t + dt/2
				dq1, dq2, dq3, dq4, dq5, dq6 = boundary_function(t + dt/2) 
				result, problem, config = calculate_p(problem, phi_np12, dq1, dq2, dq3, dq4, dq5, dq6, config) # @@@ flag, used to just be phi, but it should be phi_np12
				p_np12, dp_dx0_np12, dp_dx1_np12 = get_p(result, config)

				# Recompute erosion rate at t + dt/2
				varphi_p12 = convolve_optimized(phi_np12, B_m, extent)
				gradP2_np12 = dp_dx0_np12**2 + dp_dx1_np12**2
				f_np12 = np.maximum(0, gradP2_np12 - psi(varphi_p12, omega, varphi_star))

				# Corrector step for phi
				dtf_np12 = dt * f_np12
				phihat_1 = phi_np12 * (1 - 0.5*dtf_np12)
				e = np.sqrt(1/(config.nx*config.ny*(e_max)**2) * np.sum((phi_np1 - phihat_1)**2))

			# --- Check timestep acceptance ---
			if e >= 1:
				dt = dt * np.maximum(r_min,np.minimum(r_max,r_dec/np.sqrt(e)))
				dt = min(dt, dt_max)
			else:
				# Final corrector step (Richardson extrapolation): combine predictor & corrector estimates
				phi_np1 = 2*phihat_1 - phi_np1
				if np.any(phi_np1) < 0:
					dt = r_dec*dt
					e = 100000
				else:
					# --- Compute output quantities for logging ---
					p_c = 0.25 * (p[:-1, :-1] +	p[:-1, 1:] + p[1:, :-1] + p[1:, 1:])
					dp = np.max(p_c) - np.min(p_c)	
					Phi = np.mean(phi)
					G2 = np.mean(gradP2)
					flux_magnitude = kappa(phi) * np.sqrt(dp_dx0**2 + dp_dx1**2)
					Q = np.mean(flux_magnitude)
					F = np.mean(f)
					ErosionRate = np.mean(phi * f)

					file.write(f"{n}\t{t}\t{dt}\t{dp}\t{Phi}\t{G2}\t{Q}\t{F}\t{ErosionRate}\n")

					# --- Advance time, increase timestep ---
					times.append(t)
					t = t + dt
					dt = r_inc*dt
					dt = min(dt, dt_max)
					n = n+1               				

					# Update phi field
					phi = phi_np1
					break

	# --- Save output at specified times ---
	if save_count<Nsave and t >= save_time[save_count]:
		phi_array[save_count] = phi
		flux_array[save_count] = flux_magnitude
		tt2 = time.time()
		print("Step:", n, dt, t, tt2 -tt1)
		tt1 = time.time()	
		save_count += 1

	file.close()

	return phi, p, times, phi_array, flux_array

##########################################################
# Pressure code
##########################################################

def generate_config(nx, bx, epsilon_lh, epsilon_rh):

	grid_spacing = bx/nx
	ny = nx
	by = bx

	# INITIALIZE DOMAIN AND BOUNDARIES
	domain = dx.mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([bx, by])], [nx, ny], dx.mesh.CellType.quadrilateral)
	V = dx.fem.functionspace(domain, ("Lagrange", 1))

	# Generate indices for transformation from 2D nump array to vector -> must be possible to do this in a more transpartent way 
	coords = domain.geometry.x[:,:2]
	indices = np.lexsort((coords[:,1], coords[:,0]))  

	s = generate_s(V, nx, ny, indices)	# flux in the bulk
	ds = generate_ds(domain, bx, by, epsilon_lh, epsilon_rh) # boundary flux

	config = FEMConfig(V, s, ds, nx, ny,  grid_spacing, indices, epsilon_lh, epsilon_rh)
	config.domain = domain

	return config

def fem_initialize_pressure(phi, config):
	"""  
	This is the function that executes the FEM in each step.
	To improve the numerical stability I implemented a staggered grid the following way (see p.8 in SI of erosion paper): 
		1) in fem_step() we determine the pressure p_res on the nodes of the grid. 
		2) in get_p_information() we calculate the spatial derivatives dp_dx0, dp_dx1, 
		   but we calculate them such that they represent the derivatives of p on the center of each facet of the grid.
		3) we use the gradient of p that we calculated this way to determine phi on the center of each facet (by an Euler-forward step) in the while-loop.
		4) at the beginning of fem_step we extrapolate phi from the cell centers to the cell corners, such that we can properly calculate p.
	Empirically this has slightly improved the stability of my code.
	"""

	# Construct phi_arrays
	config.phi_v = dx.fem.Function(config.V)
	config.phi_v.x.array[:] = get_phi_v(phi, config) 			# Phi defined on the corners of the mesh

	# Construct a
	a = kappa(config.phi_v) * ufl.dot(ufl.grad(config.u), ufl.grad(config.v)) * ufl.dx 

	# Construct L
   
	# Setup weak form (contributions to L of von Neumann boundary conditions missing at this point, added a few lines below).
	# Reference to check (just in case): eq. 44 in erosion

	L = - config.s * config.v * ufl.dx

	# Note: this mode is basically neglecting that it is a point source, does that matter
	config.bc1 = dx.fem.Constant(config.domain, dx.default_scalar_type(0))
	config.bc2 = dx.fem.Constant(config.domain, dx.default_scalar_type(0))

	boundary_conditions = [ config.bc1 * config.v * config.ds(1),
							config.bc2 * config.v * config.ds(2),
        ]

	# Incorporate boundary conditions into weak form.
	for bdc in boundary_conditions:
		L += bdc    

	bx = config.grid_spacing*config.nx

	# Set dirichlet bcs
	def boundary_dele_end(x):
		return np.isclose(x[1], bx)

	def boundary_dele_begin(x):
		return np.isclose(x[1], 0)

	def p_dele_begin(x):
		return 1*(1 + 0*x[0])

	def p_dele_end(x):
		return 0*(1 + 0*x[0])

	boundary_dofs_begin = dx.fem.locate_dofs_geometrical(config.V, boundary_dele_begin)
	p_bc_begin = dx.fem.Function(config.V)
	p_bc_begin.interpolate(p_dele_begin)
	bc_begin = dx.fem.dirichletbc(p_bc_begin, boundary_dofs_begin)

	boundary_dofs_end = dx.fem.locate_dofs_geometrical(config.V, boundary_dele_end)
	p_bc_end = dx.fem.Function(config.V)
	p_bc_end.interpolate(p_dele_end)
	bc_end = dx.fem.dirichletbc(p_bc_end, boundary_dofs_end)

	problem = LinearProblem(a, L, bcs=[bc_begin,bc_end], petsc_options={"ksp_type": "cg", "pc_type": "hypre"})

	return problem, config 

def calculate_p_pressure(problem, phi, config):
	config.phi_v.x.array[:] = get_phi_v(phi, config)
	result = problem.solve()

	return result, problem, config   

def run_sim_pressure(phi0, config):
	"""
	Get pressure profile with von neumann boundary conditions		
	"""

	# Initialize Functionspace and test/trial functions.
	config.u = ufl.TrialFunction(config.V)
	config.v = ufl.TestFunction(config.V)

	# Initialize the problem
	phi = np.copy(phi0)
	problem, config = fem_initialize_pressure(phi, config)		

	# calculate p at time = t, given the von Neumann boundary conditions (dqs).
	result, problem, config = calculate_p_pressure(problem, phi, config)
	p, dp_dx0, dp_dx1 = get_p(result, config)

	return p, dp_dx0, dp_dx1