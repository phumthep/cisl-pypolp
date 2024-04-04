''' Test the random problem generator
'''
from pypolp.dw.dw import DantzigWolfe
from pypolp.optim import GurobipyOptimizer
from pypolp.problems.dantzig_wolfe import generate_dw_problem
from pypolp.dw.record import DWRecord


opt_problem, dw_problem = generate_dw_problem(
    master_size=2,
    num_master_only_vars=6,
    n_subproblems=3,
    min_subp_vars=2,
    max_subp_vars=4,
    min_subp_constrs=2,
    max_subp_constrs=5
)


# Create an instance of the Dantzig-Wolfe algorithm
record = DWRecord()
record.fit(dw_problem)
dw_model = DantzigWolfe(
    dw_improve=50,
    dw_rmpgap=50,
    num_threads=1
)
dw_model.fit(dw_problem, record)
dw_model.solve(record)

dw_objval_lp, dw_solution_lp = dw_model.get_solution(record)

# Create an instance of the Gurobi optimizer
gp_model = GurobipyOptimizer.create(opt_problem, num_threads=7)
solution = gp_model.optimize()
