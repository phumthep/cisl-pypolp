import os

import gurobipy as gp
import numpy as np
import pandas as pd


def get_pypolp_dir() -> str:
    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                    os.path.realpath(__file__))))


def get_temp_dir() -> str:
    return os.path.join(get_pypolp_dir(), 'temp')


def check_is_binary_from_df(
        df: pd.DataFrame,
        target_varnames: list[str],
        atol: float = 1e-5,
        return_non_binary: bool = False
        ) -> bool:
    ''' Return true if all target variables are binary.
    '''
    # PowNet's variable naming convention is 'variable[node, time]'
    target_variables = df.copy()
    target_variables['variable'] = target_variables['name'].str.split('[', expand=True).iloc[:, 0]
    target_variables = target_variables.loc[target_variables['variable'].isin(target_varnames), :]
    
    # To check whether values are binary,
    # we need to first prevent numerical instability
    target_variables.loc[
        np.isclose(target_variables['value'], 0, atol=atol), 'value'
                   ] = 0
    target_variables.loc[
        np.isclose(target_variables['value'], 1, atol=atol), 'value'
                   ] = 1
    
    # Values that are not exactly 0 or 1 are non-binary values
    non_int_vars = target_variables[target_variables['value'] > 0]
    non_int_vars = non_int_vars[non_int_vars['value'] < 1]
    num_non_int = len(non_int_vars)
    
    if return_non_binary:
        return (num_non_int == 0, non_int_vars)
    else:
        return num_non_int == 0


def check_is_binary_from_model(
        model: gp.Model,
        target_varnames: list[str],
        atol: float = 1e-5,
        return_non_binary: bool = False
        ) -> bool | pd.DataFrame:
    ''' Check if target variables are binary. Return non-binary
    variables otherwise.
    '''
    variables = model.getVars()
    filtered_vars = {}
    for v in variables:
        if v.varname.split('[')[0] in target_varnames:
            filtered_vars[v.varname] = v.X

    target_variables = pd.DataFrame.from_dict(filtered_vars, orient='index')
    target_variables = target_variables.reset_index()
    target_variables.columns = ['name', 'value']
    
    # To check whether values are binary,
    # we need to first prevent numerical instability
    target_variables.loc[
        np.isclose(target_variables['value'], 0, atol=atol), 'value'
                   ] = 0
    target_variables.loc[
        np.isclose(target_variables['value'], 1, atol=atol), 'value'
                   ] = 1
    
    # Values that are not exactly 0 or 1 are non-binary values
    non_int_vars = target_variables[target_variables['value'] > 0]
    non_int_vars = non_int_vars[non_int_vars['value'] < 1]
    num_non_int = len(non_int_vars)
    
    if return_non_binary:
        return (num_non_int == 0, non_int_vars)
    else:
        return num_non_int == 0
    



def generate_convex_names(n_subproblems: int):
    ''' The name of convexity contrainsts are in the format convex_[block_id].
    We recover the final solution using this regex pattern.
    '''
    return [f'convex_{j}' for j in range(n_subproblems)]


def separate_master_vars(master_vars: pd.DataFrame) -> tuple[pd.DataFrame]:
    ''' Return a dataframe with only master-only variables and a dataframe
    with weighting variables, respectively.
    '''
    master_only_vars = master_vars[~master_vars['variable'].str.contains('B\(', regex=True)]
    master_only_vars = master_only_vars.set_index('variable')
    betas = master_vars[master_vars['variable'].str.contains('B\(', regex=True)].copy()
    return master_only_vars, betas
