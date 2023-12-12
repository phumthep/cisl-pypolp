import numpy as np
import pandas as pd

from pypolp.config import (
    get_dw_max_iter,
    get_dw_improve,
    get_dw_optgap,
    get_dw_recover_integer,
    get_dw_verbose,
    get_master_timelimit
    )
from pypolp.dw.master import MasterProblem
from pypolp.dw.record import DWRecord
from pypolp.dw.subproblems import Subproblems
from pypolp.problem_class import DWProblem



class DantzigWolfe:
    ''' A manager of the Dantzig-Wolfe implementation.
    '''
    def __init__(
            self,
            max_iter: int = None,
            dw_improve: float = None,
            dw_optgap: float = None,
            recover_integer: bool = None,
            master_timelimit: int = None,
            ):
        self.dw_verbose = get_dw_verbose()
        # Use parameters from user_conf.ini if not provided
        if not max_iter:
            self.MAXITER: int = get_dw_max_iter()
        if not dw_improve:
            # in percentage
            self.DWIMPROVE: float = get_dw_improve()
        if not dw_optgap:
            self.DWOPTGAP: float = get_dw_optgap()
        # The default is not to recover an integer solution
        # We recover an integer solution by reoptimizing the master problem
        # with binary weights.
        if not recover_integer:
            self.RECOVER_INTEGER: bool = get_dw_recover_integer()

        self.subproblems: Subproblems = None
        self.n_subproblems: int = None
        
        self.master_problem: MasterProblem = None
        self.master_size: int = None

        self.phase: int = 1
        self.dw_iter: int = None # Record the final number of iterations
        


    def fit(self, dw_problem: DWProblem, record: DWRecord) -> None:
        # If there are master-only variables, then the master will contain those.
        # Otherwise, the master is an empty model.
        self.master_size = dw_problem.master_size
        self.master_problem = MasterProblem.fit(dw_problem)
        
        # We construct the master problem using proposals from the subproblems.
        # The master problem will collect proposals from subproblem as the first
        # step inside the following for loop.
        # Note that here is the zero-th iteration.
        self.subproblems = Subproblems()
        self.subproblems.fit(dw_problem)
        self.n_subproblems = self.subproblems.n_subproblems
        
        # In the beginning, solve the subproblems using their original
        # cost coefficients
        if self.dw_verbose:
            print(f'\n\n==== DW ITER 0 Phase {self.phase} ====')
        self.subproblems.solve(dw_iter=0, record=record)
        
        
    
    def solve(self, record: DWRecord) -> None:
        objval_old = np.inf
        total_reduced_cost = -np.inf
        dual_bound = -np.inf
        
        for dw_iter in range(1, self.MAXITER+1):
            # An iteration is counted when the master problem is optimize.
            if self.dw_verbose:
                print(f'\n\n==== DW ITER {dw_iter} Phase {self.phase} ====')
            # Populate the master problem with new columns
            self.master_problem.add_cols_from(record.current_PQs)
            _ = self.master_problem.solve()
            
            # Update the parameters
            self.phase = self.master_problem.phase
            duals = self.master_problem.get_duals()
            lambs = duals[:self.master_size].reshape(-1, 1)
            alphas = duals[self.master_size:]

            # Decide whether to terminate only when we are in Phase 2
            if self.phase == 2:
                record.add_primal_objval(self.master_problem.objval)
                
                # Terminate if a condition is met
                # 1) If the change in objval is below a threshold
                objval_new = abs(self.master_problem.objval)
                # Prevent division by zero
                if objval_new == 0:
                    objval_new += 1e-6
                percent_improve = abs((objval_new - objval_old)) / objval_new * 100
                objval_old = objval_new
                if percent_improve <= self.DWIMPROVE:
                    percent_improve = round(percent_improve, 3)
                    print(f'\nTerminate DW: Improvement is less than tolerance: {percent_improve} %')
                    break
                if self.dw_verbose:
                    print(f'{"DW Solve: Incre. improve:":<25} {round(percent_improve, 4)} %')
                
                # 2) If the lower bound improvement is less than threshold
                reduced_costs = [ck - alpha_k for ck,alpha_k in zip(record.subproblem_objvals, alphas)]
                # Only consider negative reduced costs when picking a variable to enter
                # reduced_costs = [rc for rc in reduced_costs if rc < 0]
                new_total_reduced_cost = sum(reduced_costs)
                
                # If we do not get new extreme points/rays, then break
                if total_reduced_cost != new_total_reduced_cost:
                    total_reduced_cost = new_total_reduced_cost
                else:
                    print('\nTerminate DW: No new proposal from subproblems')
                    break
                
                # dual_bound is zero at the first iteration
                if dw_iter > 1:
                    dual_bound = objval_new + total_reduced_cost
                record.add_dual_bound(dual_bound)
                
                # optgap is in percent. Add 0.001 to the denominator to prevent division by zero
                optgap = abs(dual_bound - objval_new)/(0.001 + abs(dual_bound))*100 
                if self.dw_verbose:
                    print(f'{"DW Solve: Optgap:":<25} {round(optgap, 4)} %')
                # total_reduced_cost is zero at the first iteration
                if optgap <= self.DWOPTGAP:
                    print(f'\nTerminate DW: Optgap is less than tolerance: {round(optgap*100, 4)} %')
                    break
                # Remove all current objective values from record
                record.reset_subproblem_objvals()
                
            if dw_iter == self.MAXITER:
                print(f'\nTerminate DW: Reached max iteration: {self.MAXITER}')
            self.subproblems.update_solve(self.phase, dw_iter, lambs, record)
        
        # Produce an error if we have not reached Phase 2 after reaching the max iteration.
        if not self.phase == 2:
            raise ValueError('DantzigWolfe has not entered phase II.')
        
        # Record the total iterations required
        self.dw_iter = dw_iter
        
    
    def get_solution(
            self, record: DWRecord,
            recover_integer: bool = False
            ) -> tuple[float, pd.DataFrame]:
        # We need to recover integer solutions
        if self.RECOVER_INTEGER or recover_integer:
            self.master_problem.convert_betas_to_int()
            # self.master_problem.model.setParam('OutputFlag',1)
            _ = self.master_problem.solve()
        
        # Get X from the master problem
        master_vars = self.master_problem.get_X()
        objval = self.master_problem.objval
        
        master_only_var_idx = ~master_vars['variable'].str.contains('B\(', regex=True)
        if master_only_var_idx.sum() == 0:
            master_only_vars = None
        else:
            master_only_vars = master_vars[master_only_var_idx]
            master_only_vars = master_only_vars.set_index('variable')
        
        betas = master_vars[~master_only_var_idx].copy()
    
        # Label each row of beta with its subproblem_id and iteration_id
        p = r'B\((?P<j>\d+),(?P<i>\d+)\)'
        betas[['j', 'i']] = betas['variable'].str.extract(p)
        betas = betas.astype({'j':int, 'i':int})
    
        # For each beta, extract the corresponding solution X from the record
        betas['X'] = betas.apply(
            lambda row: record.get_proposal(row['i'], row['j']).X,
            axis = 1
            )
        
        # Weight each solution by its beta and then do group sum
        betas['weighted_X'] = betas['X'].multiply(betas['value'])
        temp_x = betas.groupby('j').agg({'weighted_X': 'sum'}).values.flatten()
    
        final_solution = []
        for j, x in enumerate(temp_x):
            final_solution.extend(x)
        
        master_size = len(final_solution)
        final_solution = pd.DataFrame(
            final_solution, 
            index = record.varnames[:master_size])
        
        final_solution.index = final_solution.index.rename('variable')
        final_solution.columns = ['value']
        
        final_solution = pd.concat([final_solution, master_only_vars], axis=0)
            
        return objval, final_solution
    
    
    def get_runtimes_dict(self) -> dict[int|str: list[float]]:
        runtimes = self.subproblems.runtimes
        runtimes['master'] = self.master_problem.runtimes
        return runtimes
    

    def get_itercounts_dict(self) -> dict[int|str: list[float]]:
        itercounts = self.subproblems.itercounts
        itercounts['master'] = self.master_problem.itercounts
        return itercounts
    
    
    def get_stats(self, mode) -> tuple[float|int, float|int]:
        ''' Return the total runtime/itercounts for (master, subproblem)
        '''
        stats: list | dict[int, list] = None
        if mode == 'runtime':
            stats = self.get_runtimes_dict()
        elif mode == 'itercount':
            stats = self.get_itercounts_dict()
        
        master_stats = None
        subproblem_stats = 0
        for k, v in stats.items():
            if k == 'master':
                master_stats = sum(v)
            else:
                subproblem_stats += sum(v)
        return master_stats, subproblem_stats
    
    
    def get_objvals(self) -> list:
        return self.primal_values