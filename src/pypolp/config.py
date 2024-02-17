import configparser
import os

from pypolp.functions import get_pypolp_dir


config_file = os.path.join(get_pypolp_dir(), 'user_config.ini')
CONFIG = configparser.ConfigParser()
CONFIG.read(config_file)


def get_config() -> configparser.ConfigParser:
    return CONFIG



#---- Dantzig-Wolfe
def get_dw_max_iter() -> int:
    ''' The number of iterations impact the solution quality.
    '''
    return CONFIG.getint('DW', 'MAXITER')


def get_dw_improve() -> float:
    ''' Terminate DW when incremental improvement in
    the objective value of the master problem is below a threshold.
    '''
    return CONFIG.getfloat('DW', 'DWIMPROVE')


def get_dw_rmpgap() -> float:
    ''' Terminate DW when the gap between the current objval of the RMP
    and the current lower bound is larger than a threshold.
    '''
    return CONFIG.getfloat('DW', 'DWRMPGAP')


def get_dw_recover_integer() -> bool:
    ''' If we need to reoptimize the master problem with binary variables.
    '''
    return CONFIG.getboolean('DW', 'RECOVER_INTEGER')


def get_dw_verbose() -> bool:
    ''' Print new extreme points/rays added at each iteration.
    '''
    return CONFIG.getboolean('DW', 'VERBOSE')



#---- Master Problem
def get_master_verbose() -> bool:
    ''' Print Gurobi log for the master problem.
    '''
    return CONFIG.getboolean('DWMASTER', 'VERBOSE')


def  get_master_timelimit() -> float:
    ''' The time to solve the master problem depends on 
    the number of proposals so far. 
    '''
    return CONFIG.getfloat('DWMASTER', 'TIMELIMIT')


def get_master_mipgap() -> float:
    ''' Specify mipgap of the master problem.
    '''
    return CONFIG.getfloat('DWMASTER', 'MIPGAP')



#---- Subproblems
def get_subp_warmstart() -> bool:
    return CONFIG.getboolean('DWSUBPROBLEM', 'WARMSTART')

def get_subp_verbose() -> bool:
    ''' Print Gurobi log for subproblems.
    '''
    return CONFIG.getboolean('DWSUBPROBLEM', 'VERBOSE')


def get_subp_mipgap() -> bool:
    ''' Specify mipgap of all subproblems.
    '''
    return CONFIG.getfloat('DWSUBPROBLEM', 'MIPGAP')


def get_subp_relax() -> bool:
    ''' Specify whether to convert the subproblems to linear programs 
    by relaxing the integer variables.
    '''
    return CONFIG.getboolean('DWSUBPROBLEM', 'RELAX')



#---- Gurobi
def get_gp_warmstart() -> bool:
    return CONFIG.getboolean('GUROBI', 'WARMSTART')


def get_gp_mipgap() -> float:
    return CONFIG.getfloat('GUROBI', 'MIPGAP')


def get_gp_verbose() -> bool:
    return CONFIG.getboolean('GUROBI', 'VERBOSE')


def get_gp_record() -> bool:
    ''' If the GurobipyOptimizer should keep track of statistics.
    '''
    return CONFIG.getboolean('GUROBI', 'RECORD')
