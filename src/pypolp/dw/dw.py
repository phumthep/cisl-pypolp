from collections import defaultdict
import configparser

import numpy as np
import pandas as pd

from pypolp.dw.master import MasterProblem
from pypolp.dw.record import Record
from pypolp.dw.subproblems import Subproblems
from pypolp.problem_class import DWProblem
from pypolp.tools.functions import get_config



class DantzigWolfe:
    def __init__(self):
        config: configparser.ConfigParser = get_config()
        self.MAXITER: int = int(config['DEFAULT']['MAXITER'])
        self.DWTOL: float = float(config['DEFAULT']['DWTOL']) # in percentage
        
        self.subproblems: Subproblems = None
        self.master_problem: MasterProblem = None

        self.master_size: int = None
        self.n_subproblems: int = None

        self.phase: int = 1
        self.duals: np.array = None
        self.primal_objvals = None
        


    def fit(self, dw_problem: DWProblem, record: Record) -> None:
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
        self.subproblems.solve(dw_iter=0, record=record)
        
        
    
    def solve(self, record: Record) -> None:
        self.primal_objvals = []
        objval_old = np.inf
        
        for dw_iter in range(1, self.MAXITER+1):
            # An iteration is counted when the master problem is optimize.
            print(f'\n\n==== DW ITER {dw_iter} Phase {self.phase} ====')
            # Populate the master problem with new columns
            self.master_problem.add_cols_from(record)
            _ = self.master_problem.solve()
            
            # Update the parameters

            self.phase = self.master_problem.phase

            if self.phase == 2:
                self.primal_objvals.append(self.master_problem.objval)
                # Terminate if a condition is met
                # 1) Check if the improvement is below a threshold
                objval_new = abs(self.master_problem.objval)
                if objval_new == 0:
                    objval_new += 1e-6
                percent_improve = abs((objval_new - objval_old)) / objval_new * 100
                objval_old = objval_new
            
                if percent_improve <= self.DWTOL:
                    percent_improve = round(percent_improve, 3)
                    print(f'\nTerminate DW: Improvement is less than tolerance: {percent_improve} %')
                    break

            duals = self.master_problem.get_duals()
            lambs = duals[:self.master_size].reshape(-1, 1)
            self.subproblems.update_solve(self.phase, dw_iter, lambs, record)

    
    
    def get_solution(self, record: Record) -> type[float, pd.DataFrame]:
        objval = None

        if self.phase == 2:
            # Get X1 from the master problem
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
            temp_x = betas.groupby('j').agg({'weighted_X': np.sum}).values.flatten()
        
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
    
        else:
            raise ValueError('DantzigWolfe has not entered phase II.')
            
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