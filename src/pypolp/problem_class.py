from dataclasses import dataclass
import typing

import pandas as pd

@dataclass(slots=True)
class OptProblem:
    ''' Class representing an optimization problem.
    User can manually create the dataframes.
    '''
    obj_coeffs: pd.DataFrame
    A: pd.DataFrame
    rhs: pd.DataFrame
    inequalities: pd.DataFrame
    var_info: pd.DataFrame
    
    def get_dataframes(self) -> tuple[pd.DataFrame, ...]:
        return (
            self.obj_coeffs,
            self.A,
            self.rhs,
            self.inequalities,
            self.var_info)



@dataclass(slots=True)
class DWProblem(OptProblem):
    ''' Class representing a Dantzig-Wolfe decomposition problem.
    '''
    row_indices: typing.Any = None
    col_indices: typing.Any = None
    n_subproblems: int = None
    master_size: int = None
    
    def get_opt_problem(self) -> OptProblem:
        return OptProblem(
            obj_coeffs = self.obj_coeffs,
            A = self.A,
            rhs = self.rhs,
            inequalities = self.inequalities,
            var_info = self.var_info
            )