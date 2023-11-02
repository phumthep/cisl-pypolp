from collections import defaultdict

from gurobipy import GRB
import numpy as np
import pandas as pd


from pypolp.optim import GurobipyOptimizer, Solution, Proposal, get_model_from
from pypolp.problem_class import OptProblem,DWProblem
from pypolp.dw.record import Record
from pypolp.tools.functions import get_config



def convert_solution_to_proposal(
        solution: Solution, 
        dw_iter: int, 
        block_id: int) -> Proposal:
    return Proposal(
        X = solution.X,
        objval = solution.objval,
        is_ray = solution.is_ray,
        dw_iter = dw_iter,
        block_id = block_id)


class Subproblem(GurobipyOptimizer):
    def __init__(self, opt_problem: OptProblem, to_log: bool): 
        self.block_id: int = None
        
        model = get_model_from(opt_problem, to_log)
        # The MIPGap is required to achieve good solution
        config = get_config()
        mipgap = float(config['DWSUBPROBLEM']['MIPGAP'])
        model.setParam('MIPGap', mipgap)
        super().__init__(model)
        
        
        
    def set_id(self, block_id: int) -> None:
        self.block_id = block_id
    
    
    def set_c_A(self, cost_coeffs: pd.DataFrame, A_master: pd.DataFrame) -> None:
        self.cost_coeffs = cost_coeffs
        self.A_master = A_master
    
    
    def _update_farkas(self, lambs: np.array, x: np.array) -> None:
        new_c = np.matmul(lambs.T, self.A_master.values)
        new_c = new_c.flatten()
        self.model.setObjective(
            expr = new_c @ x,
            sense = GRB.MINIMIZE)
    
    
    def _update_reduced_cost(self, lambs: np.array, x: np.array) -> None:
        x = np.array(self.model.getVars())
        new_c = self.cost_coeffs.values.T - np.matmul(lambs.T, self.A_master.values)
        new_c = new_c.flatten()
        self.model.setObjective(
            expr = new_c @ x,
            sense = GRB.MINIMIZE)
    

    def update_c(self, dw_phase, lambs):
        """
        Update the cost cofficients either with Farkas pricing or reduced cost pricing
        """
        x = np.array(self.model.getVars())
        if dw_phase == 1:
            self._update_farkas(lambs, x)
        else:
            self._update_reduced_cost(lambs, x)
        # Gurobi requires manually updating the model
        self.model.update()


    @classmethod
    def create(cls, opt_problem: OptProblem, to_log: bool) -> 'Subproblem':
        return cls(opt_problem, to_log)


class Subproblems():
    def __init__(self):
        self.all_subproblems: dict[int: GurobipyOptimizer] = None
        self.n_subproblems: int = None
        
        # Track the optimization progress
        self.runtimes: dict[int, list[float]] = None
        self.itercounts: dict[int, list[int]] = None
        
    
    def _record_opt_stats(self, block_id: int, runtime: int, itercount: float) -> None:
        self.runtimes[block_id].append(runtime)
        self.itercounts[block_id].append(itercount)


    def fit(self, dw_problem: DWProblem) -> None:
        ''' Chop the constraint matrix and create subproblems
        '''
        
        config = get_config()
        subp_debug = int(config['DWSUBPROBLEM']['DEBUG'])
        
        # Call subproblem creator
        self.all_subproblems = []
        self.runtimes = defaultdict(list)
        self.itercounts = defaultdict(list)
        
        master_size = dw_problem.master_size
        
        # Loop thru to create subproblems
        for block_id in range(dw_problem.n_subproblems):

            row_id = dw_problem.row_indices[block_id]
            col_id = dw_problem.col_indices[block_id]
            
            # Prepare a subproblem and append it to all_subproblems
            opt_problem = OptProblem(
                dw_problem.obj_coeffs.iloc[col_id.start: col_id.end],
                dw_problem.A.iloc[row_id.start: row_id.end, col_id.start: col_id.end],
                dw_problem.rhs.iloc[row_id.start: row_id.end],
                dw_problem.inequalities.iloc[row_id.start: row_id.end],
                dw_problem.var_info.iloc[col_id.start: col_id.end]
                )
            
            subproblem = Subproblem.create(opt_problem, subp_debug)
            
            subproblem.set_c_A(
                cost_coeffs = dw_problem.obj_coeffs.iloc[col_id.start:col_id.end],
                A_master = dw_problem.A.iloc[:master_size, col_id.start:col_id.end]
                )
            subproblem.set_id(block_id)
            
            self.all_subproblems.append(subproblem)


    def solve(self, dw_iter, record: Record) -> None:
        for block_id, subproblem in enumerate(self.all_subproblems):
            # print(f'\nDW Solve: Subproblem {block_id}\n')
            solution = subproblem.optimize()
            #TODO: Record the stats only when in the analytical mode to save memory and computation
            self._record_opt_stats(block_id, subproblem.runtime, subproblem.itercount)
            
            proposal = convert_solution_to_proposal(
                    solution, 
                    dw_iter, 
                    block_id)
            record.update(proposal)
    
    
    def update_solve(
            self, 
            dw_phase: int, 
            dw_iter: int,
            lambs: np.array, 
            record: Record
            ) -> None:
        ''' This method is called after the second DW iteration. We coupled
        the update and optimize steps together to save the overhead from using
        two for loops.
        '''
        for block_id, subproblem in enumerate(self.all_subproblems):
            subproblem.update_c(dw_phase, lambs)
            solution = subproblem.optimize()
            record.add_subproblem_objval(solution.objval)
            
            #TODO: Record the stats only when in the analytical mode to save memory and computation
            self._record_opt_stats(block_id, subproblem.runtime, subproblem.itercount)
            
            proposal = convert_solution_to_proposal(
                    solution, 
                    dw_iter, 
                    block_id)
            record.update(proposal)


                    