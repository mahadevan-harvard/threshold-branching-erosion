# Threshold branching erosion

Simulates flow-driven branching through threshold-based erosion of a porous medium.

## Installation

This project uses Python 3 with fenicsx, installation through conda is recommended.

1. **Clone the repository:**
```bash
   git clone https://github.com/mahadevan-harvard/threshold-branching-erosion.git
   cd threshold-branching-erosion
```
   
2. **(Optional but recommended) Install fenicsx through conda environment:**
```bash
   conda create -n fenicsx-env
   conda activate fenicsx-env
   conda install -c conda-forge fenics-dolfinx mpich pyvista
```

3. **Install the package in editable mode::**
```bash
    pip install -e .
```
  
## Repository structure

- `erosion/`: Core simulation and solver code
- `scripts/`: Scripts to run simulations
- `analysis/`: Tools for analyzing simulation results

## Usage

The scripts in `scripts/` can be used to run simulations with different geometries and boundary conditions.  
Simulation parameters and parameter ranges are set directly in these scripts.

Each parameter sweep creates an output folder named:

{simtype}_{yyyymmddhhmm}

Inside, a subfolder is generated for each parameter set, e.g.:

F_0.3_T_10.0_V_001

for a linear ramp simulation with flow rate `F = 0.3`, ramp rate `T = 10.0`, and repeat index `V = 001`.

After a simulation completes, each subfolder contains:

- **`parameters.txt`** – all input parameters used for the simulation  
- **`data.npz`** – snapshots of the solid fraction and flux magnitude fields at specified time intervals  
- **`simulation_log.txt`** – global parameters of the simulation as a function of time (adaptive time stepping)

The log file contains:

- `Step` – simulation step  
- `time` – simulation time (adaptive time stepping)  
- `Delta_p` – pressure difference between source and sink  
- `Phi` – spatial average solid fraction  
- `G2` – spatial average squared pressure gradient magnitude  
- `Q` – Global influx  
- `F` – spatial average input to the threshold function  
- `erate` – spatial average erosion rate  

The analysis scripts in `analysis/` operate directly on this output folder structure.