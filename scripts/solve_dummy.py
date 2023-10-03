# from pydopt.components.base_opt import gurobipyOptimizer as baseOpt
from pydopt.problems.p10 import problem
from pydopt.core.dw import dw_opt
from pydopt.core.optim import gurobipyOptimizer as baseOpt


(
 c_df, A_df, b_df, ineq_df, col_df, subp_indices, col_indices
 ) = problem()


# Vanilla solver
# base_model = baseOpt.create_model(c_df, A_df, b_df, ineq_df, col_df, debug=True)
# base_model.optimize()


# Dantzig-Wolfe solver
MAXITER = 10
DW_TOL = 1e-5

dw_obj, dw_x, master = dw_opt(
    c_df, A_df, b_df, ineq_df, col_df, subp_indices, col_indices, MAXITER, DW_TOL)

