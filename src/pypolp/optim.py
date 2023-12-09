from __future__ import annotations
from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

from pypolp.config import (
    get_gp_warmstart,
    get_gp_mipgap,
    get_gp_record,
    get_gp_verbose,
    )
from pypolp.problem_class import OptProblem



@dataclass(slots=True)
class Solution:
    ''' Class containing info on the solution of an optimization problem.
    '''
    X: list
    objval: float
    is_ray: bool
    
    def __eq__(self, other):
        return (
            (self.X == other.X).all()
            and (self.objval == other.objval) 
            and (self.is_ray == other.is_ray)
            )
 

@dataclass(slots=True)
class Proposal(Solution):
    ''' A Solution class with information on block_id and the DW iteration.
    '''
    dw_iter: int
    block_id: int
    
    def __eq__(self, other):
        return (
            (self.X == other.X).all()
            and (self.is_ray == other.is_ray)
            )
    
    @classmethod
    def from_solution(
            cls,
            solution: Solution, 
            dw_iter: int, 
            block_id: int
            )-> Proposal:
        return Proposal(
            X = solution.X,
            objval = solution.objval,
            is_ray = solution.is_ray,
            dw_iter = dw_iter,
            block_id = block_id)


class GurobipyOptimizer:
    ''' Class that wraps around gurobipy model.
    '''
    def __init__(
            self,
            model: gp.Model,
            warmstart: bool,
            mipgap: float,
            verbose: bool,
            to_record: bool,
            ):
        self.status_map = {2: 'optimal', 3: 'infeasible', 5:'unbounded', 9:'time_limit'}
        self.model: gp.Model = model
        
        # Default to Gurobi parameters in user_config.ini
        if not warmstart:
            self.model.setParam('LPWarmStart', get_gp_warmstart())
        else:
            self.model.setParam('LPWarmStart', warmstart)
            
        if not mipgap:
            self.model.setParam('mipgap', get_gp_mipgap())
        else:
            self.model.setParam('mipgap', mipgap)
        
        if not verbose:
            self.model.setParam('outputflag', get_gp_verbose())
        else:
            self.model.setParam('outputflag', verbose)
            
        # If want to record gurobi statistics
        if not to_record:
            self.to_record = get_gp_record()
        else:
            self.to_record = to_record
            
        # Need to compute additional information when a model is 
        # infeasible to get the Farkas ray
        self.model.setParam('infunbdinfo', 1)
        
        self.status: str = None
        self.farkas_duals: np.array = None

        self.objval: float = None
        self.runtime: float = None # in seconds
        self.itercount: int = None
        
        
    def __repr__(self):
        return f'GurobipyOptimizer(model={self.model})'
        

    def optimize(self) -> Solution:
        self.model.optimize()
        
        # Store the Gurobi status code
        status: int = self.model.Status
        
        self.status = self.status_map.get(status)

        if self.to_record:
            self.objval = self.model.objval
            self.runtime = self.model.runtime
            self.itercount = int(self.model.itercount)

        
        if self.status == 'optimal':
            return Solution(
                X = np.array(self.model.x), 
                objval = self.model.objval, 
                is_ray = False
                )
        elif self.status == 'infeasible':
            self.farkas_duals = np.array(self.model.farkasdual)
            return None
        elif self.status == 'unbounded':
            # Returns an extreme ray
            return Solution(
                X = np.array(self.model.UnbdRay), 
                objval = None, 
                is_ray = True
                )
        elif self.status == 'time_limit':
            return Solution(
                X = None,
                objval = None,
                is_ray = None
                )
        else:
            raise RuntimeError(
                '\nOptimization terminates with status ' + str(status))
        

    def get_duals(self) -> np.array:
        if self.status == 'optimal':
            duals = np.array([constr.pi for constr in self.model.getConstrs()])
        else:
            duals = self.farkas_duals
        return duals
    
    
    def get_X(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                'variable': self.model.getAttr('varname'),
                'value': self.model.getAttr('X')}
            )


    @classmethod
    def create(
            cls,
            opt_problem: OptProblem,
            warmstart: bool = None,
            mipgap: float = None,
            verbose: bool = None,
            to_record: bool = None
            ) -> GurobipyOptimizer:
        return cls(
            model = cls.get_gp_model_from_opt_problem(opt_problem),
            warmstart = warmstart,
            mipgap = mipgap,
            verbose = verbose,
            to_record = to_record
            )
    
    
    @classmethod
    def from_file(
            cls, 
            filename: str, 
            warmstart: bool = None,
            mipgap: float = None,
            verbose: bool = None,
            to_record: bool = None
            ) -> GurobipyOptimizer:
        ''' Create a GurobipyOptimizer from a file instead of
        getting from dataframes.
        '''
        return cls(
            model = gp.read(filename),
            warmstart = warmstart,
            mipgap = mipgap,
            verbose = verbose,
            to_record = to_record
            )
    

    @staticmethod
    def get_gp_model_from_dataframes(
            obj_coeffs: pd.DataFrame,
            A: pd.DataFrame,
            rhs: pd.DataFrame,
            inequalities: pd.DataFrame,
            var_info: pd.DataFrame
            ) -> gp.Model:
        ''' Create a gurobipy model from a set of five dataframes.
        '''
        model = gp.Model()
        # Define variables
        model_vars = model.addMVar(
            shape = var_info.shape[0], 
            name = var_info.index
            )
        model_vars.setAttr('lb', var_info.lower.values)
        model_vars.setAttr('ub', var_info.upper.values)
        model_vars.setAttr('vtype', var_info.type.values)
        # Define the objective value
        model.setObjective(
            expr = obj_coeffs.values.T @ model_vars,
            sense = GRB.MINIMIZE)
        # Add constraints
        model_constrs = model.addMConstr(
            A.values, model_vars, inequalities['value'].values, rhs.values.reshape(-1)
            )
        model_constrs.setAttr('constrname', A.index)
        model.update()
        return model
    
    
    @staticmethod
    def get_gp_model_from_opt_problem(opt_problem: OptProblem) -> gp.Model:
        ''' Create a gurobipy model from OptProblem class.
        '''
        return GurobipyOptimizer.get_gp_model_from_dataframes(
            *opt_problem.get_dataframes()
            )