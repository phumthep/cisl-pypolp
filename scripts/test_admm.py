'''
This script implements ADMM.

Assume we have the following problem

min f(x) + g(z)
s.t. Ax + Bz = rhs

x is the master-only variables, while z is the subproblem variables.

The augmented Lagrangian is
f(x) + g(z) + y.T @ (Ax + Bz - rhs) + rho/2 * ||Ax + Bz - rhs||^2

Note that rhs is "right-hand side".

'''
from __future__ import annotations
import os

from gurobipy import GRB
import gurobipy as gp
import numpy as np
import pandas as pd

from pypolp.problem_class import DWProblem
from pypolp.parser import parse_mps_with_orders, parse_mps, get_dataframe_orders


def form_x_model(
        dw_problem: DWProblem,
        y: np.array,
        z: np.array,
        rho: float
) -> gp.Model:
    ''' Create a model with master-only variables (x).
    The objective function is the augmented Lagrangian with x as the variable.

    Consider the augmented lagrangian function:

    f(x) + y.T @ (Ax + Bz - rhs) + rho/2 * ||Ax + Bz - rhs||^2

    Since we are only interested in the x part, we can ignore variable z, which
    we treat as constants. Also, remove 1/2 from the quadratic term 
    to simplify the expression. Now we have,

    f(x) + y.T @ (Ax) + rho * ||Ax + Bz - rhs||^2

    '''
    # ----- Create an optimization model with master-only variables or f(x)
    # Note: col_indices is a list of tuples (start_col, end_col), where the last tuple specify
    # the section of the A matrix that corresponds to the master-only variables
    col_id = dw_problem.col_indices[-1]

    # ----- Create f(x)
    obj_coeffs = dw_problem.obj_coeffs.iloc[col_id.start:]
    A = dw_problem.A.iloc[:, col_id.start:]

    # Ordering of var_info is different and begins with master-only variables
    var_info = dw_problem.var_info.iloc[col_id.start:]

    x_model = gp.Model('x_model')

    # Use only one thread
    x_model.setParam('Threads', 1)

    # Define variables
    x_model_vars = x_model.addMVar(
        shape=var_info.shape[0],
        name=var_info.index
    )
    x_model_vars.setAttr('lb', var_info.lower.values)
    x_model_vars.setAttr('ub', var_info.upper.values)
    x_model_vars.setAttr('vtype', var_info.type.values)

    # Define the objective term f(x) with the master-only variables
    fx = obj_coeffs.values.T @ x_model_vars

    # ----- Create y.T @ (Ax) term
    penalty_1 = y.T @ A.values @ x_model_vars

    # ----- Create the quadratic penalty term
    B = dw_problem.A.iloc[:, :col_id.start]
    temp = (
        A.values @ x_model_vars
        + B.values @ z
        - dw_problem.rhs.values.reshape(-1,)
    )
    penalty_2 = (temp * temp).sum()

    # ----- Update the objective function
    x_model.setObjective(
        expr=fx + penalty_1 + rho*penalty_2,
        sense=GRB.MINIMIZE
    )
    x_model.update()
    return x_model


def form_z_model(
        dw_problem: DWProblem,
        x: np.array,
        y: np.array,
        rho: float
) -> gp.Model:
    ''' Create a model with subproblem variables (z).
    The objective function is the augmented Lagrangian with z as the variable.

    Consider the augmented lagrangian function:

    f(x) + y.T @ (Ax + Bz - rhs) + rho/2 * ||Ax + Bz - rhs||^2

    Since we are only interested in the z part, we can ignore x variables,
    which we treat as constants. Also, remove 1/2 from the quadratic term to simplify the expression.

    y.T @ (Bz) + rho * ||Ax + Bz - rhs||^2

    '''
    # ----- Create an optimization model with master-only variables or f(x)
    # Note: col_indices is a list of tuples, where the last tuple specify
    # the section of the A matrix that corresponds to the master-only variables
    col_id = dw_problem.col_indices[-1]

    # ----- Create g(z)
    obj_coeffs = dw_problem.obj_coeffs.iloc[:col_id.start]
    B = dw_problem.A.iloc[:, :col_id.start]
    # Unfortunately, ordering of var_info begins with master-only variables
    var_info = dw_problem.var_info.iloc[:col_id.start]

    z_model = gp.Model('z_model')

    # Define variables
    z_model_vars = z_model.addMVar(
        shape=var_info.shape[0],
        name=var_info.index
    )
    z_model_vars.setAttr('lb', var_info.lower.values)
    z_model_vars.setAttr('ub', var_info.upper.values)
    z_model_vars.setAttr('vtype', var_info.type.values)

    # ----- Create y.T @ (Bz) term
    penalty_1 = y.T @ B.values @ z_model_vars

    # ----- Create the quadratic penalty term
    A = dw_problem.A.iloc[:, col_id.start:]
    temp = (
        A.values @ x
        + B.values @ z_model_vars
        - dw_problem.rhs.values.reshape(-1,)
    )
    penalty_2 = (temp * temp).sum()

    # ----- Update the objective function
    z_model.setObjective(
        expr=penalty_1 + rho*penalty_2,
        sense=GRB.MINIMIZE
    )
    z_model.update()
    return z_model


# ================= Load the problem =================
mps_file = '..//temp//dummy.mps'
dec_file = '..//temp//dummy.dec'
(_, A_df, _, _, col_df) = parse_mps(mps_file)
row_order, col_order = get_dataframe_orders(dec_file, A_df, col_df)
dw_problem = parse_mps_with_orders(mps_file, row_order, col_order)
# del A_df
del col_df

# ================= Solve with ADMM =================
# Note: col_indices is a list of tuples, where the last tuple specify
# the section of the A matrix that corresponds to the master-only variables
col_id = dw_problem.col_indices[-1]
A = dw_problem.A.iloc[:, col_id.start:]
B = dw_problem.A.iloc[:, :col_id.start]

x_opttime = []
z_opttime = []

# Initialize variables
rho = 0.5
col_id = dw_problem.col_indices[-1]
z = np.zeros(col_id.start)
y = np.zeros(dw_problem.A.shape[0])
b = dw_problem.rhs

for i in range(10):
    print(f'=== Iteration: {i+1} ===')
    # Perform argmin-x step
    x_model = form_x_model(dw_problem, y, z, rho)
    x_model.optimize()
    x_opttime.append(x_model.Runtime)

    # Use the estimated x variables to update the z variables
    x = np.array(x_model.getAttr('x'))
    # Perform argmin-z step
    z_model = form_z_model(dw_problem, x, y, rho)
    z_model.optimize()
    z_opttime.append(z_model.Runtime)

    # Use the estimated x, z variables to update the y variables
    z = np.array(z_model.getAttr('x'))
    # Update the multipliers y
    y += rho * (
        A.values @ x
        + B.values @ z
        - dw_problem.rhs.values.reshape(-1,)
    )


# ================= Solve with Gurobi =================
gp_model = gp.read(mps_file)
gp_model.optimize()
gp_model.Runtime

# ================= Post processing =================
# Recover the solution x and z
solution = {}
for xvar in x_model.getVars():
    solution[xvar.varName] = xvar.x
for zvar in z_model.getVars():
    solution[zvar.varName] = zvar.x
solution = pd.DataFrame.from_dict(solution, orient='index')

# Format the dataframe
solution = solution.reset_index()
solution.columns = ['varname', 'admm_value']

# Join with the Gurobi solution
gp_solution = pd.DataFrame(
    [(var.varName, var.x) for var in gp_model.getVars()],
    columns=['varname', 'gurobi_value']
)
solution = solution.merge(gp_solution, on='varname', how='left')
del gp_solution
