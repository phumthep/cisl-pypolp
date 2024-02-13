from gurobipy import GRB
import gurobipy as gp
import numpy as np

from pypolp.functions import get_temp_dir

'''
From Decomposition Techniques Example 5.1

min  x^2 + y^2
s.t.
    -x - y <= -4
    x, y >= 0
solution: x = 2, y = 2 and objval = 8.
'''

def update_with_subgradient(
        mu_k: float,
        current_x: gp.Var,
        current_y: gp.Var,
        k: int,
        a: float = 1,
        b: float = 0.1
        ) -> float:
    ''' Based on page 205.
    '''
    mismatch = -current_x.X -current_y.X + 4
    if mismatch > 0:
        return (
            mu_k
            + 1/(a + b*k)
            * (-current_x.X -current_y.X + 4) / abs(-current_x.X -current_y.X + 4)
            )
    else:
        return mu_k

# Define primal variables
model = gp.Model('test_lagrangian')
x = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name='x',)
y = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name='y')

# Initialize variables before Lagrangian relaxation
mu = 0
primal_lb = -np.inf
primal_ub = np.inf

for i in range(10):

    # Solve the primal problem
    objfunc_expr = x**2 + y**2 + mu * (-x - y + 4)
    model.setObjective(objfunc_expr, GRB.MINIMIZE)
    model.optimize()

    # Update the lower bound
    primal_lb = max(primal_lb, model.objVal)
    primal_ub = min(primal_ub, model.objVal)

    # Update the multiplier
    mu = update_with_subgradient(mu, x, y, i)
    
    print(f'\nIteration {i}:')
    print(f'    x = {x.X}')
    print(f'    y = {y.X}')
    print(f'    mu = {mu}')

