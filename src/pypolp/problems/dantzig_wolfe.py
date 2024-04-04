import numpy as np

from scipy.sparse import random as create_sparse_mat
from scipy.linalg import block_diag
from scipy import stats

from pypolp.problem_class import OptProblem, DWProblem
from pypolp.problems.functions import (
    convert_to_namedtuples,
    get_dataframes,
    convert_df_to_problems
)


def create_index(num_items: int, start_index: int, append_none: bool):
    ''' Return a list of tuples of start_idx and end_idx 
    when given a list of number of items in each subproblem.
    '''
    # start_index is the size of the master problem
    indices = []
    for n in num_items:
        indices.append((start_index, start_index + n))
        start_index = start_index + n
    if append_none:
        indices.append((None, None))
    return indices


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

    opt_problem, dw_problem = convert_df_to_problems(
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
    opt_problem, dw_problem = convert_df_to_problems(
        c_df, A_df, b_df, ineq_df, col_df,
        row_indices, col_indices,
        n_subproblems, master_size
    )

    return opt_problem, dw_problem


def generate_dw_problem(
        master_size: int,
        num_master_only_vars: int,
        n_subproblems: int,
        min_subp_vars: int,
        max_subp_vars: int,
        min_subp_constrs: int,
        max_subp_constrs: int,
        master_density: float = 0.8,
        subp_density: float = 0.8
) -> tuple[OptProblem, DWProblem]:
    '''
    Generate a Dantzig-Wolfe problem with the specified parameters.
    The optimization problem has the form:
    min c^T x
    s.t. Ax <= b
    x >= 0
    '''
    # Each subproblem has between min_subp_vars to max_subp_vars.
    nvars_in_subp = np.random.randint(
        min_subp_vars, max_subp_vars+1, size=n_subproblems
    )
    col_indices = convert_to_namedtuples(
        create_index(
            num_items=nvars_in_subp,
            start_index=0,
            append_none=False
        )
    )

    # Sample coefficients from Poisson distribution
    # because the distribution produces integers
    rvs = stats.poisson(1, loc=1.5).rvs

    # Create constraints for the master problem
    Amp = create_sparse_mat(
        master_size,
        nvars_in_subp.sum(),
        density=master_density,
        random_state=123,
        data_rvs=rvs
    ).A

    # Section of the A matrix that corresponds to the master-only variables
    Amp_minor = create_sparse_mat(
        master_size,
        num_master_only_vars,
        density=master_density,
        random_state=123,
        data_rvs=rvs
    ).A

    # Randomly assign number of constraints in the subproblems
    ncons_in_subp = np.random.randint(
        min_subp_constrs, max_subp_constrs+1, size=n_subproblems
    )
    blocks = []

    for m, n in zip(ncons_in_subp, nvars_in_subp):
        constraints = create_sparse_mat(
            m, n, density=subp_density, random_state=123, data_rvs=rvs
        ).A
        # Filter rows of zeros (if any) to prevent issues
        constraints = constraints[~np.all(constraints == 0, axis=1)]
        blocks.append(constraints)

    # Update ncons_in_subp because we have removed rows with all zeros
    ncons_in_subp = np.array([block.shape[0] for block in blocks])

    has_master_only_vars = num_master_only_vars > 0
    row_indices = convert_to_namedtuples(
        create_index(
            ncons_in_subp,
            master_size,
            append_none=has_master_only_vars
        )
    )

    # Create system of the form Ax <= b. To ensure feasibility,
    # we have b being dependent on A and x
    A = np.r_[Amp, block_diag(*blocks)]

    Amp_minor = np.r_[Amp_minor, np.zeros(
        (A.shape[0] - master_size, num_master_only_vars))]

    A = np.c_[A, Amp_minor]

    xfeasible = np.random.randint(0, 6, size=A.shape[1])
    b = np.matmul(A, xfeasible) + np.random.randint(1, 3, size=A.shape[0])
    c = np.random.randint(-10, 10, size=A.shape[1])

    # Format the data into OptProblem and DWProblem objects
    (c_df, A_df, b_df, ineq_df, col_df) = get_dataframes(c, A, b)
    opt_problem, dw_problem = convert_df_to_problems(
        c_df, A_df, b_df, ineq_df, col_df,
        row_indices, col_indices,
        n_subproblems, master_size
    )

    return opt_problem, dw_problem


def test_create_index():
    assert create_index([2, 3, 4], 0, False) == [(0, 2), (2, 5), (5, 9)]
    assert create_index([2, 3, 4], 1, False) == [(1, 3), (3, 6), (6, 10)]
    assert create_index([2, 3, 4], 2, False) == [(2, 4), (4, 7), (7, 11)]


if __name__ == '__main__':
    # Unit test for create_index
    test_create_index()
