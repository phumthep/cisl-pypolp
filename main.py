from pypolp.dw.dw import DantzigWolfe
from pypolp.optim import GurobipyOptimizer
from pypolp.dw.record import DWRecord
from pypolp.problems.dantzig_wolfe import problem_02 as problem


def main():
    #------- Load a problem instance
    opt_problem, dw_problem = problem()
    
    #------- Optimize with Dantzig-Wolfe
    record = DWRecord()
    record.fit(dw_problem)
    
    dw_instance = DantzigWolfe()
    dw_instance.fit(dw_problem, record)
    dw_instance.solve(record)
    
    dw_objval, dw_solution = dw_instance.get_solution(record)
    master_time, subproblem_time = dw_instance.get_stats(mode='runtime')
    
    
    base_opt = GurobipyOptimizer.create(opt_problem, to_log=True)
    true_solution = base_opt.optimize()
    
    print('\n============================================')
    print(f'\n{"":<10} {"Completed Dantzig-Wolfe":^10}')
    print('\nOptimization time')
    print(f'{"Master Problem:":<20} {round(master_time, 3)} s')
    print(f'{"Subproblem:":<20} {round(subproblem_time, 3)} s')
    print(f'{"DW Total:":<20} {round(master_time+subproblem_time, 3)} s')
    print(f'{"Base Gurobi:":<20} {round(base_opt.runtime, 3)} s')
    print('\nObjective Value')
    print(f'{"DW:":<20} {round(dw_objval, 2)} s')
    print(f'{"Gurobi:":<20} {round(true_solution.objval, 2)} s')
    print('\n============================================')




if __name__ == '__main__':
    main()
