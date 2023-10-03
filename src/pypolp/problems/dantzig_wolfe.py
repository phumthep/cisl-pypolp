#%% A toy problem with master-only variables

import numpy as np
import pandas as pd

from pypolp.problem_class import OptProblem, DWProblem
from pypolp.problems.functions import convert_to_namedtuples


def get_dataframes(c, A, b):
    row_names = [f'constr_{j}' for j in range(A.shape[0])]
    col_names = [f'var_{k}' for k in range(A.shape[1])]
    
    c_df = pd.DataFrame(c, index=col_names, columns=['value'])
    A_df = pd.DataFrame(A, index=row_names, columns=col_names)
    b_df = pd.DataFrame(b, index=row_names, columns=['value'])
    
    ineq_df = pd.DataFrame(['L']*A.shape[0], index=row_names, columns=['sign'])
    
    col_df = pd.DataFrame({
        'type': ['C']*A.shape[1],
        'lower': [0] * A.shape[1],
        'upper': [np.inf] * A.shape[1]},
        index = col_names)
    
    return c_df, A_df, b_df, ineq_df, col_df


def df_2_problems(c_df, A_df, b_df, ineq_df, col_df, row_indices, col_indices, n_subproblems, master_size):
    opt_problem = OptProblem(c_df, A_df, b_df, ineq_df, col_df)
    dw_problem = DWProblem(
        c_df, A_df, b_df, ineq_df, col_df, row_indices, col_indices, n_subproblems, master_size)
    return opt_problem, dw_problem


def problem_01() -> tuple[OptProblem, DWProblem]:
    # x* = (4, 0, 2, 0) and z = -14
    # Source: https://www.youtube.com/watch?v=wxz0NJvKZNM
    
    # Define the problem characteristics
    master_size = 2
    n_subproblems = 2
    
    # Define the start/end rows in the A matrix of each subproblem
    row_indices = convert_to_namedtuples(
        [(2, 4), (4, 6)]
        )
    
    col_indices = convert_to_namedtuples(
        [(0, 2), (2, 4)]
        )
    
    # Specify the parameters A, b, c
    c = np.array([-2, -1, -3, -1])
    A = np.array([
        [1, 1, 1, 1],
        [0, 1, 2, 1],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 1],
        [0, 0, 1, 1]    
        ])
    b = np.r_[6, 4, 6, 2, 3, 5]
    
    (c_df, A_df, b_df, ineq_df, col_df) = get_dataframes(c, A, b)
    opt_problem, dw_problem = df_2_problems(
        c_df, A_df, b_df, ineq_df, col_df, 
        row_indices, col_indices, 
        n_subproblems, master_size
        )
    
    return opt_problem, dw_problem


def problem_02() -> tuple[OptProblem, DWProblem]:
    # Define the problem characteristics
    master_size = 1
    n_subproblems = 2
    
    # Define the start/end rows in the A matrix of each subproblem
    row_indices = convert_to_namedtuples([(1, 3), (3, 6), (None, None)])
    col_indices = convert_to_namedtuples([(0, 2), (2, 4), (4, 7)])
    
    
    # Specify the parameters A, b, c
    c = np.array([-1, -8, -5, -6, -1, -3, -1])
    A = np.array([
        [1, 4,  5,   2,    1, 2, 1],
        [2, 3,  0,   0,    0, 0, 0],
        [5, 1,  0,   0,    0, 0, 0],
        [0, 0, -3,  -4,    0, 0, 0],
        [0, 0,  1,   0,    0, 0, 0],
        [0, 0,  0,   1,    0, 0, 0],
        ])
    
    b = np.r_[15, 6, 5, -12, 4, 3]

    (c_df, A_df, b_df, ineq_df, col_df) = get_dataframes(c, A, b)
    opt_problem, dw_problem = df_2_problems(
        c_df, A_df, b_df, ineq_df, col_df, 
        row_indices, col_indices, 
        n_subproblems, master_size
        )
    
    return opt_problem, dw_problem


