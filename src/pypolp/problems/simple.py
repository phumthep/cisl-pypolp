# This file contains simple optimization problems
# to test optimization models

import pandas as pd
import numpy as np

from pypolp.data_class import OptProblem



def problem_01(to_dw: bool) -> OptProblem:
    varnames = ['w', 'x', 'y', 'z']
    constr_names = ['c1', 'c2', 'c3', 'c4']
    
    c_df = pd.DataFrame({'value': [5, 2, -4, -1]}, index=varnames)
    
    A_df = pd.DataFrame(
        {
            'w': [2, 0, 0, 0],
            'x': [2, 1, 1, 0],
            'y': [1, 0, 0, 0],
            'z': [0, 0, 0, 2]
            },
        index=constr_names
        )
    
    b_df = pd.DataFrame({'value': [15, -3, 5, 10]}, index=constr_names)
    
    ineq_df = pd.DataFrame({'sign': ['<', '>', '<', '=']}, index=constr_names)
    
    col_df = pd.DataFrame(
        {
            'type': ['Continuous', 'Continuous', 'Continuous', 'Continuous'],
            'lower': [0, float(-np.inf), float(-np.inf), float(-np.inf)],
            'upper': [float(np.inf), float(np.inf), float(np.inf), float(np.inf)]},
            index = varnames
        )
    

    return OptProblem.from_dataframes(c_df, A_df, b_df, ineq_df, col_df)



def problem_02():
    '''
    min    -2x1 - 3x2 - 4x3 - 5x4
    s.t.
           x1 +   x2 +   x3 +   x4   <=   30
                        3x3 +  4x4   <=   12
          2x1 +  2x2                 <=   24
        
           x1, x2, x3, x4 >= 0
        
    '''
    col_names = ['var1', 'var2', 'var3', 'var4']
    row_names = ['foo', 'bar', 'baz']
    
    c_df = pd.DataFrame([-2, -3, -4, -5], index=col_names, columns=['value'])
    
    A_df = pd.DataFrame({
        'var1': [1, 0, 2],
        'var2': [1, 0, 2],
        'var3': [1, 3, 0],
        'var4': [1, 4, 0]},
        index = row_names)
    
    b_df = pd.DataFrame([10, 12, 24], index=row_names, columns=['value'])
    
    ineq_df = pd.DataFrame(['<']*3, index=row_names, columns=['sign'])
    col_df = pd.DataFrame({
        'type': ['Continuous'] * 4,
        'lower': [0] * 4,
        'upper': [np.inf] * 4},
        index = col_names)
    
    return c_df, A_df, b_df, ineq_df, col_df


def unbounded_problem():
    '''
    min    -2x1 - 3x2 - 4x3 - 5x4  - 10x5
    s.t.
           x1 +   x2 +   x3 +   x4  <=   30
                        3x3         <=   12
          2x1                       <=   24
        
           x1, x2, x3, x4 >= 0
        
    '''
    col_names = ['var1', 'var2', 'var3', 'var4', 'var5']
    row_names = ['foo', 'bar', 'baz']
    
    c_df = pd.DataFrame([-2, -3, -4, -5, -10], index=col_names, columns=['value'])
    
    A_df = pd.DataFrame({
        'var1': [1, 0, 2],
        'var2': [1, 0, 0],
        'var3': [1, 3, 0],
        'var4': [1, 0, 0],
        'var5': [0, 0, 0]},
        index = row_names)
    
    b_df = pd.DataFrame([10, 12, 24], index=row_names, columns=['value'])
    
    ineq_df = pd.DataFrame(['<']*3, index=row_names, columns=['sign'])
    col_df = pd.DataFrame({
        'type': ['Continuous'] * 5,
        'lower': [0] * 5,
        'upper': [np.inf] * 5},
        index = col_names)
    
    return c_df, A_df, b_df, ineq_df, col_df


def infeasible_problem():
    '''
    min -x1 + -x2 + -x3
    s.t.
    x1 +  x2 +  x3  = 6
    2x1 + 3x2 +  x3  = 8
    2x1 +  x2 + 3x3  = 0
    
    (4, -1, -1 ) is a certificate of feasibility
    '''
    # Define the problem
    col_names = ['var1', 'var2', 'var3']
    row_names = ['foo', 'bar', 'baz']
    
    c_df = pd.DataFrame([-1, -1, -1], index=col_names, columns=['value'])
    
    A_df = pd.DataFrame({
        'var1': [1, 2, 2],
        'var2': [1, 3, 1],
        'var3': [1, 1, 3]
        },
        index = row_names)
    
    b_df = pd.DataFrame([6, 8, 0], index=row_names, columns=['value'])
    
    ineq_df = pd.DataFrame(['=']*3, index=row_names, columns=['sign'])
    
    col_df = pd.DataFrame({
        'type': ['Continuous'] * 3,
        'lower': [0] * 3,
        'upper': [np.inf] * 3},
        index = col_names)
    
    return c_df, A_df, b_df, ineq_df, col_df
    