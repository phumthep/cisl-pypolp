from pypolp.dw.dw import DantzigWolfe
from pypolp.optim import GurobipyOptimizer
from pypolp.dw.record import Record
from pypolp.problems.dantzig_wolfe import problem_02 as problem


def main():
    #------- Load a problem instance
    opt_problem, dw_problem = problem()
    
    #------- Optimize with Dantzig-Wolfe
    record = Record()
    record.fit(dw_problem)
    
    dw_instance = DantzigWolfe()
    dw_instance.fit(dw_problem, record)
    dw_instance.solve(record)
    
    dw_objval, dw_solution = dw_instance.get_solution(record)
    master_time, subproblem_time = dw_instance.get_stats(mode='runtime')
    
    
    base_opt = GurobipyOptimizer.create(opt_problem, to_log=True)
    true_solution = base_opt.optimize()
    
    print('\n============================================')
    print('\n=== Completed solving with Dantzig-Wolfe ===')
    print(f'Opt time - Master Problem:   {round(master_time, 5)} s')
    print(f'Opt time - Subproblem:       {round(subproblem_time, 5)} s')
    print(f'Opt time - DW Total:          {round(master_time+subproblem_time, 5)} s')
    print(f'Opt time - Base Gurobi:      {round(base_opt.runtime, 5)} s')
    print(f'\nObj Val - DW:             {round(dw_objval, 2)} s')
    print(f'Obj Val - Base Gurobi:    {round(true_solution.objval, 2)} s')
    print('\n============================================')




if __name__ == '__main__':
    main()
