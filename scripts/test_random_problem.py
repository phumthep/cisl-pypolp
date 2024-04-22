''' Test the random problem generator
'''
from pypolp.dw.dw import DantzigWolfe
from pypolp.optim import GurobipyOptimizer
from pypolp.problems.dantzig_wolfe import generate_dw_problem
from pypolp.dw.record import DWRecord

if __name__ == '__main__':
    opt_problem, dw_problem = generate_dw_problem(
        master_size=40,
        num_master_only_vars=20,
        n_subproblems=200,
        min_subp_vars=200,
        max_subp_vars=400,
        min_subp_constrs=200,
        max_subp_constrs=500
    )

    import time

    # Create an instance of the Dantzig-Wolfe algorithm
    record = DWRecord()
    record.fit(dw_problem)
    begin = time.time()
    dw_model = DantzigWolfe(
        dw_improve=50,
        dw_rmpgap=50,
        num_threads=1,
        to_parallel=False
    )
    dw_model.fit(dw_problem, record)
    dw_model.solve(record)

    dw_objval_lp, dw_solution_lp = dw_model.get_solution(record)
    end = time.time()
    serial = ("serial", dw_objval_lp, end - begin, dw_solution_lp)

    ####################################################################
    record = DWRecord()
    record.fit(dw_problem)
    begin = time.time()
    dw_model = DantzigWolfe(
        dw_improve=50,
        dw_rmpgap=50,
        num_threads=1,
        to_parallel=True,
        num_processes=4
    )
    dw_model.fit(dw_problem, record)
    dw_model.solve(record)

    dw_objval_lp, dw_solution_lp = dw_model.get_solution(record)
    end = time.time()
    parallel = ("parallel", dw_objval_lp, end - begin, dw_solution_lp)

    ####################################################################

    ####################################################################
    # begin = time.time()
    # # Create an instance of the Gurobi optimizer
    # gp_model = GurobipyOptimizer.create(opt_problem, num_threads=7)
    # solution = gp_model.optimize()
    # end = time.time()
    # gurobipy = ("gold", solution.objval, end-begin)

    print(serial)
    print(parallel)
    # print(gurobipy)
