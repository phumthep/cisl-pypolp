from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

from pypolp.problem_class import OptProblem, DWProblem
from pypolp.tools.functions import get_config
from pypolp.tools.parser import parse_mps, parse_mps_dec



STATUS_MAP = {2: 'optimal', 3: 'infeasible', 5:'unbounded', 9:'time_limit'}

CONFIG = get_config()
gp.setParam('LPWarmStart', int(CONFIG['GUROBI']['WARMSTART']))

# Required to get Farkas ray
gp.setParam('infunbdinfo', 1)
gp.setParam('MIPGap', 0)



#----- Classes
def get_model_from(opt_problem: OptProblem, to_log: bool) -> gp.Model:
    
    # Extract parameters required to set-up a Gurobi model
    (
        obj_coeffs,
        A,
        rhs,
        inequalities,
        var_info
        ) = opt_problem.get_dataframes()
    
    model = gp.Model()
    model.setParam('outputflag', to_log)
    
    # Define variables
    model_vars = model.addMVar(
        shape = var_info.shape[0], 
        name = var_info.index)
    
    model_vars.setAttr('lb', var_info.lower.values)
    model_vars.setAttr('ub', var_info.upper.values)
    model_vars.setAttr('vtype', var_info.type.values)
    
    # Define the objective value
    model.setObjective(
        expr = obj_coeffs.values.T @ model_vars,
        sense = GRB.MINIMIZE)
    
    # Identify and add the constraints
    model_constrs = model.addMConstr(
        A.values, model_vars, inequalities['sign'].values, rhs.values.reshape(-1))

    model_constrs.setAttr('constrname', A.index)
    model.update()
    return model



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
    ''' Similar to Solution but has a name or ID.
    '''
    dw_iter: int
    block_id: int
    
    def __eq__(self, other):
        return (
            (self.X == other.X).all()
            and (self.is_ray == other.is_ray)
            )


class GurobipyOptimizer:
    ''' Class that acts as a wrapper around Gurobipy model.
    '''
    def __init__(self, model):
        self.model: gp.Model = model
        self.status: str = None
        self.farkas_duals: np.array = None
        
        self.to_log: bool = None
        
        self.objval: float = None
        self.runtime: float = None # in seconds
        self.itercount: int = None
        
        
    def __repr__(self):
        return f'GurobipyOptimizer(model={self.model})'
        

    def optimize(self) -> Solution:
        
        self.model.optimize()
        self.objval = self.model.objval
        #TODO: Save only when doing analysis
        self.runtime = self.model.runtime
        self.itercount = int(self.model.itercount)
        
        status = self.model.Status
        self.status = STATUS_MAP.get(status)
        
        
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
            duals = [constr.pi for constr in self.model.getConstrs()]
        else:
            duals = self.farkas_duals
        return np.array(duals)
    
    
    def get_X(self) -> pd.DataFrame:
        var_names = self.model.getAttr("varname")
        var_vals = self.model.getAttr("x")
        return pd.DataFrame({'variable':var_names, 'value':var_vals})


    @classmethod
    def create(
            cls,
            opt_problem: OptProblem, 
            to_log: bool
            ) -> 'GurobipyOptimizer':
        model = get_model_from(opt_problem, to_log)
        return cls(model)
    
    
    @classmethod
    def from_file(cls, filename: str, to_log: bool) -> 'GurobipyOptimizer':
        ''' Directly create a GurobipyOptimizer from a file instead of
        getting them from dataframes.
        '''
        model = gp.read(filename)
        model.setParam('outputflag', to_log)
        return cls(model)
    
    
    
    
#----- Supporting functions
def create_opt_problem(mps_file: str) -> OptProblem:
    return OptProblem.from_dataframes(*parse_mps(mps_file))


def create_dw_problem(mps_file: str, dec_file: str) -> DWProblem:
    return DWProblem.from_tuple(*parse_mps_dec(mps_file, dec_file))
