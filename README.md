# Decomposition Methods for Large-Scale Optimization (Python Package)

This Python package provides implementations of decomposition methods for solving large-scale optimization problems. It is being developed as part of my PhD research. 

Currently this package implements Dantzig-Wolfe Decomposition to solve an optimization problem with the block-diagonal structure. The optimization can be input as (1) MPS and DEC files or (2) Pandas dataframes that specific the A matrix, b bector, c_vector, inequalities, row & column orders (to permute A,b,c to mark which is the master problem and the subproblems).

## Current Implementation

- **Dantzig-Wolfe Decomposition (DWD):** A classic decomposition method that breaks down a large problem into smaller, more manageable subproblems. 
   - A parallelized version of DWD is available in the `cs-project-omp` branch.

## Planned Implementations

- **Lagrangian Relaxation (LR):** A relaxation technique that forms a dual problem to provide a lower bound on the optimal value of the original problem.
- **Alternating Direction Method of Multipliers (ADMM):** A versatile algorithm that blends the decomposability of dual ascent with the superior convergence properties of the method of multipliers.

## Installation
```bash
git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -e .
