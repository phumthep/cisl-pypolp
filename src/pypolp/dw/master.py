from __future__ import annotations

from gurobipy import GRB
import gurobipy as gp
import pandas as pd

from pypolp.config import (
    get_master_timelimit,
    get_master_verbose,
    get_master_mipgap
    )
from pypolp.dw.record import ProposalPQ
from pypolp.functions import generate_convex_names
from pypolp.optim import GurobipyOptimizer, Solution
from pypolp.problem_class import DWProblem



class MasterProblem(GurobipyOptimizer):
    def __init__(
            self,
            model: gp.Model,
            mipgap: float,
            timelimit: int,
            verbose: bool,
            ):
        # Inherit methods from GurobipyOptimizer class
        super().__init__(
            model = model,
            warmstart = True,
            mipgap = mipgap,
            verbose = verbose,
            to_record = True
            )
        if not timelimit:
            self.model.setParam('timelimit', get_master_timelimit())
        else:
            self.model.setParam('timelimit', timelimit)
        # Other add-on attributes from GurobipyOptimizers
        self.has_mov: bool = None
        self.phase: int = None
        self.runtimes: list[float, ...] = None
        self.itercounts: list[int, ...] = None
    
    
    def _add_col(self, proposal_pq: ProposalPQ) -> None:
        ''' Add a new column from a proposal. P is a scalar of the objective coefficient.
        Q is a vector of coefficients in the A matrix.
        '''
        if not (len(proposal_pq.Q) == self.model.getAttr('numconstrs')):
            raise ValueError(
                'The shape of Q does not match the number of constraints in the master problem.')
        
        varname = f'B({proposal_pq.block_id},{proposal_pq.dw_iter})'
        # A ray may get scaled by the weight variable without an upperbound.
        if not proposal_pq.is_ray:
            self.model.addVar(
                lb = 0, 
                obj = proposal_pq.P.iloc[0], 
                vtype = GRB.CONTINUOUS, 
                name = varname,
                column = gp.Column(proposal_pq.Q, self.model.getConstrs())
                )
        else:
            # Defining 1 as the upper bound will mess up the dual variables
            self.model.addVar(
                lb = 0,
                obj = proposal_pq.P, 
                vtype = GRB.CONTINUOUS, 
                name = varname,
                column = gp.Column(proposal_pq.Q, self.model.getConstrs())
                )
        
        
    def add_cols_from(self, current_PQs: list[ProposalPQ, ...]) -> None:
        ''' Add new columns using extreme points/rays from the subproblems.
        '''
        while current_PQs:
            proposal_pq = current_PQs.pop()
            self._add_col(proposal_pq)
        self.model.update()
        

    def solve(self) -> Solution:
        if self.verbose:
            print('\nDW Solve: Master Problem\n')
        solution = self.optimize()

        if self.model.status == 3: # Infeasible
            self.phase = 1
        elif self.model.status == 2: # Optimal
            self.phase = 2
        elif self.model.status == 9: # Hit time limit
            pass
        else:
            raise RuntimeError(f'Gurobi terminated with status {self.model.status}')
        
        # GurobiOptimizer has the record attribute
        if self.to_record:
            if not self.runtimes:
                self.runtimes = [self.runtime]
                self.itercounts = [self.itercount]
            else:
                self.runtimes.append(self.runtime)
                self.itercounts.append(self.itercount)
        
        return solution
    
    
    def convert_betas_to_int(self) -> None:
        ''' Change the beta variables to integer and re-optimize.
        '''
        for gp_var in self.model.getVars():
            if gp_var.varname.startswith('B('):
                gp_var.setAttr('VType', GRB.INTEGER)
                
    
    @staticmethod
    def _get_rhs_inequality_dataframes(dw_problem: DWProblem) -> tuple[pd.DataFrame]:
        ''' Return two dataframes to define the convexity constraints:
            b-vector (or right-hand side) and the inequality signs.
        '''
        convex_names: list[str, ...] = generate_convex_names(dw_problem.n_subproblems)
        # The RHS of convexity constraints is 1
        rhs_convex = pd.DataFrame(
            [1 for _ in range(dw_problem.n_subproblems)],
            index = convex_names,
            columns = ['value']
            )
        # The sign of convexity constraints is the equality
        ineq_convex = pd.DataFrame(['E']*dw_problem.n_subproblems, index=convex_names, columns=['value'])
        
        rhs = pd.concat(
            [dw_problem.rhs[:dw_problem.master_size], rhs_convex],
            axis = 0
            )
        inequality = pd.concat(
            [dw_problem.inequalities[:dw_problem.master_size], ineq_convex],
            axis = 0
            )
        return rhs, inequality
    
    
    @classmethod
    def _get_master_model(cls, dw_problem: DWProblem) -> gp.Model:
        ''' The master model contains blank objective function and blank A matrix.
        Although the constraints are originally empty, we will add new columns to them.
        Here, we need placeholders for the original constraints and the convexity constraints.
        Note that we must define the  signs and the RHS for these placeholders.
        '''
        rhs, inequality = cls._get_rhs_inequality_dataframes(dw_problem)
        
        model = gp.Model()
        model.addConstrs(0==0 for _ in range(dw_problem.n_subproblems))
        model.setAttr('rhs', model.getConstrs(), rhs)
        model.setAttr('sense', model.getConstrs(), inequality)
        model.setAttr('constrname', model.getConstrs(), rhs.index)
        model.update()
        return model
    
    
    @classmethod
    def _get_master_model_with_mov(cls, dw_problem: DWProblem) -> gp.Model:
        ''' Create a model of the master problem given master-only variables.
        The objective function and the constraint matrix contain master-only variables.
        Note that we also have the convexity constraints.
        '''
        
        # The master-only variables are specified by the last component of col_indices
        col_id = dw_problem.col_indices[-1]
        
        obj_coeffs = dw_problem.obj_coeffs.iloc[col_id.start:]
        A = dw_problem.A.iloc[:dw_problem.master_size, col_id.start:]
        var_info = dw_problem.var_info.iloc[col_id.start:]
        
        # Create the section for convexity constraints
        rhs, inequality = cls._get_rhs_inequality_dataframes(dw_problem)
        
        A_convex = pd.DataFrame(
            0,
            index = generate_convex_names(dw_problem.n_subproblems),
            columns = A.columns
            )
        A = pd.concat(
            [A, A_convex],
            axis=0
            )
        return cls._get_gp_model_from_dataframes(
            obj_coeffs = obj_coeffs,
            A = A,
            rhs = rhs,
            inequalities = inequality,
            var_info = var_info
            )
    
    
    @classmethod
    def fit(
            cls,
            dw_problem: DWProblem,
            timelimit: int = None
            ) -> MasterProblem:
        # When there are no master-only variables, the master problem is empty
        # but with placeholder for constraints.
        has_mov = dw_problem.check_has_check_master_only_vars()
        if not has_mov:
            model: gp.Model = cls._get_master_model(dw_problem=dw_problem)
        else:
            model: gp.Model = cls._get_master_model_with_mov(dw_problem=dw_problem)
        
        return cls(
            model = model,
            mipgap = get_master_mipgap(),
            timelimit = timelimit,
            verbose = get_master_verbose(),
            )

