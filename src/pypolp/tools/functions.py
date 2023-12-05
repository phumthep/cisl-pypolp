import configparser
import os

import gurobipy as gp
import numpy as np
import pandas as pd


def get_root_dir() -> str:
    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.realpath(__file__)))))


def get_config() -> configparser.ConfigParser:
    config_file = os.path.join(get_root_dir(), 'user_config.ini')
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


def check_is_binary(
        model: gp.Model,
        target_varnames: list[str],
        atol: float = 1e-5
        ) -> bool:
    ''' Return true if target variables are binary.
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
    
    return num_non_int == 0
    
