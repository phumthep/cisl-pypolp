from dataclasses import dataclass
import typing

import pandas as pd



@dataclass(slots=True)
class OptProblem:
    ''' Class representing an optimization problem. This class is a container
    for dataframes, which a user can manually create the dataframes.
    '''
    obj_coeffs: pd.DataFrame
    A: pd.DataFrame
    rhs: pd.DataFrame
    inequalities: pd.DataFrame
    var_info: pd.DataFrame
    
    def get_dataframes(self) -> tuple[pd.DataFrame]:
        return (
            self.obj_coeffs,
            self.A,
            self.rhs,
            self.inequalities,
            self.var_info
            )



@dataclass(slots=True)
class DWProblem(OptProblem):
    ''' Class representing a Dantzig-Wolfe decomposition problem.
    '''
    # If there are no master-only variables, then the last tuple
    # in row_indices is (None, None)
    row_indices: typing.Any = None
    col_indices: typing.Any = None
    
    n_subproblems: int = None
    master_size: int = None

    def get_opt_problem(self) -> OptProblem:
        ''' Return as an OptProblem class.
        '''
        return OptProblem(
            obj_coeffs = self.obj_coeffs,
            A = self.A,
            rhs = self.rhs,
            inequalities = self.inequalities,
            var_info = self.var_info
            )
    
    def check_has_check_master_only_vars(self) -> bool:
        ''' If there is a master-only variable, then the final member of
        row_indices is (None, None)
        '''
        return (self.row_indices[-1].start is None)
    
    
