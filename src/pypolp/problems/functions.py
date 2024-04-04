from collections import namedtuple

import numpy as np
import pandas as pd

from pypolp.problem_class import OptProblem, DWProblem


Index = namedtuple('Index', 'start end')


def convert_df_to_optproblem(
        c_df: pd.DataFrame,
        A_df: pd.DataFrame,
        b_df: pd.DataFrame,
        ineq_df: pd.DataFrame,
        col_df: pd.DataFrame,
) -> OptProblem:
    ''' Return OptProblem (and DWProblem) from a set of dataframes.
    '''
    return OptProblem(c_df, A_df, b_df, ineq_df, col_df)


def convert_df_to_dwproblem(
        c_df: pd.DataFrame,
        A_df: pd.DataFrame,
        b_df: pd.DataFrame,
        ineq_df: pd.DataFrame,
        col_df: pd.DataFrame,
        row_indices: pd.DataFrame,
        col_indices: pd.DataFrame,
        n_subproblems: pd.DataFrame,
        master_size: int
) -> DWProblem:
    ''' Return OptProblem (and DWProblem) from a set of dataframes.
    '''
    return DWProblem(
        c_df, A_df, b_df, ineq_df, col_df,
        row_indices, col_indices, n_subproblems, master_size)


def convert_df_to_problems(
        c_df: pd.DataFrame,
        A_df: pd.DataFrame,
        b_df: pd.DataFrame,
        ineq_df: pd.DataFrame,
        col_df: pd.DataFrame,
        row_indices: pd.DataFrame,
        col_indices: pd.DataFrame,
        n_subproblems: pd.DataFrame,
        master_size: int
) -> tuple[OptProblem, DWProblem]:
    ''' Return both OptProblem and DWProblem from a set of dataframes.
    '''
    return (
        OptProblem(c_df, A_df, b_df, ineq_df, col_df),
        DWProblem(c_df, A_df, b_df, ineq_df, col_df, row_indices,
                  col_indices, n_subproblems, master_size)
    )


def convert_to_namedtuples(
        indices: list[tuple[int], ...]
) -> list[Index, ...]:
    ''' Return a list of named tuples. Each tuple is (start_idx, end_idx).
    '''
    return [Index(*item) for item in indices]


def get_dataframes(
        c: np.array,
        A: np.array,
        b: np.array
):
    ''' Return a set of five dataframes to represent an optimization problem.
    This problem assumes that variables are non-negative.
    The sign is less than or equal to.
    '''
    row_names = [f'constr_{j}' for j in range(A.shape[0])]
    col_names = [f'var_{k}' for k in range(A.shape[1])]

    c_df = pd.DataFrame(c, index=col_names, columns=['value'])
    A_df = pd.DataFrame(A, index=row_names, columns=col_names)
    b_df = pd.DataFrame(b, index=row_names, columns=['value'])

    ineq_df = pd.DataFrame(
        ['L']*A.shape[0], index=row_names, columns=['value'])

    col_df = pd.DataFrame(
        {
            'type': ['C']*A.shape[1],
            'lower': [0] * A.shape[1],
            'upper': [np.inf] * A.shape[1]
        },
        index=col_names
    )
    return c_df, A_df, b_df, ineq_df, col_df
