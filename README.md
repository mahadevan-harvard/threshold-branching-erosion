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
