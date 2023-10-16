from collections import namedtuple
import re

import gurobipy as gp
import pandas as pd
import numpy as np

from pypolp.problem_class import OptProblem, DWProblem



Index = namedtuple('Index', ['start', 'end'])



######################################
##### Functions for mps file
######################################

def parse_mps(path: str) -> tuple[pd.DataFrame, ...]:
    print('\nParsing the MPS file...')
    model = gp.read(path)
    
    # Loop through variables
    varnames = []
    lowerbounds = []
    upperbounds = []
    obj_coeffs = []
    vtypes = []
    for v in model.getVars():
        varnames.append(v.varname)
        lowerbounds.append(v.lb)
        upperbounds.append(v.ub)
        obj_coeffs.append(v.obj)
        vtypes.append(v.vtype)
    
    # Loop through constraints
    b = []
    ineq = []
    constr_names = []
    for constr in model.getConstrs():
        b.append(constr.rhs)
        ineq.append(constr.sense)
        constr_names.append(constr.constrname)
    
    
    c_df = pd.DataFrame(obj_coeffs, index=varnames, columns=['value'])
    
    A = model.getA()
    A_df = pd.DataFrame(A.toarray(), index=constr_names, columns=varnames)
    
    b_df = pd.DataFrame(b, index=constr_names, columns=['value'])
    
    inequalities = pd.DataFrame(ineq, index=constr_names, columns=['sign'])
    
    col_df = pd.DataFrame(
        {'type':vtypes, 'lower':lowerbounds, 'upper':upperbounds},
        index = varnames
        )
    
    return (
        c_df, A_df, b_df, inequalities, col_df)
    


######################################
##### Functions for dec file
# Info: https://gcg.or.rwth-aachen.de/doc-3.5.0/own-dec-file.html
######################################

def add_item(constr_dict, key, value) -> None:
    ''' Add unique keys to a dictionary {blockid: [constr_name, constr_name]}.
    '''
    if key not in constr_dict:
        constr_dict[key] = value
    else:
        raise ValueError(f'The block number of {key} is not unique.')


def parse_section(line) -> int:
    ''' Return the section ID if this line describes 
    the block number or the master section.
    '''
    if line == 'MASTERCONSS':
        return 0
    
    else:
        p = r'\d+'
        num_block = re.search(p, line).group()
        num_block = int(num_block)
        return num_block
    

def get_row_order(lines: list[str, ...], num_blocks: int) -> pd.DataFrame:
    ''' Return a dataframe describing constr_name and block_id pairs.
    The rows are sorted in an ascending order by their block_ids.
    '''
    # Set flag to true when we the current line is at 'BLOCK XX'
    flag = False
    sections = ['BLOCK ' + str(i) for i in range(1, num_blocks+1)]
    sections.append('MASTERCONSS')
    
    # Label each line with its section, Block 1, 2, ... or master.
    cur_section = None
    constr_dict = {}
    
    for line in lines:
        line = line.rstrip()
        
        # If the current line is the section header, then update the section label.
        # Otherwise, we will label the following lines under the current section.
        flag = (line in sections)
        if flag:
            cur_section = parse_section(line)
            
        # The current line might not belong to a block or the master
        # if it is at the beginning of the DEC file.
        # If a line belongs to 
        elif (cur_section is not None) and (not flag):
            add_item(constr_dict, line, cur_section)
        
        # The current line does not belong to a block or the master.
        else:
            pass
            # print('\nIrrelevant line:\n', line)
    
    row_order = pd.DataFrame(constr_dict, index=['block_id']).transpose()
    row_order = row_order.sort_values(by='block_id', ascending=True)
    
    return row_order
    

def get_col_order(A_df, row_order, col_df) -> pd.DataFrame:
    ''' Return a dataframe describing var_name and block_id pairs.
    The rows are sorted in an ascending order by their block_ids.
    '''
    df = A_df.join(row_order)
    variables = list(col_df.index)
    col_order = pd.DataFrame(0, index=variables, columns=['block_id'])
    
    # For each variable, find constraints (rows) in which it belongs
    for variable in variables:
    
        # var_df is a column
        var_df = df.loc[:, [variable, 'block_id']]
        
        mask = np.where(
            (var_df[variable] != 0) & (var_df['block_id'] != 0))
        
        # There is an edge case where a variable does not belong to any block
        # Here, that variable is pushed to the last box. Subproblem has no constraints
        if len(mask[0]) > 0:
            memberships = var_df.iloc[mask]['block_id'].unique()
            # Check that this variable is not a linking variable
            if len(memberships) > 1:
                raise ValueError(f'{variable} is in multiple blocks.')
            col_order.loc[variable] = memberships[0]
        elif len(mask[0]) == 0:
            col_order.loc[variable] = np.nan
    
    if (col_order.block_id == 0).any():
        print('\nWarning: some variables are unlabeled in the dec file.\n')
    
    col_order = col_order.sort_values(by='block_id', ascending=True)
    
    return col_order


def get_orders(path, A_df, col_df) -> tuple[pd.DataFrame, pd.DataFrame]:
    print('\nParsing the DEC file...')
    with open(path, 'r') as f:
        lines = f.readlines()
    # A DEC file states the number of blocks in the line 
    # after the NBLOCKS line
    num_blocks = int(lines[lines.index('NBLOCKS\n')+1])
    row_order = get_row_order(lines, num_blocks)
    col_order = get_col_order(A_df, row_order, col_df)
    return row_order, col_order


def get_indices(order_df) -> list[Index[int, int]]:
    ''' Create a list of tuples containing the starting index 
    and ending index of each block. The input dataframe must be ordered
    by blocks in an ascending fashion.
    '''
    df = order_df.copy()
    df = df.reset_index(names='name')
    
    indices = []
    # The first tuple pertains to Block 1. Master-only variables are
    # pushed to the end section of order_df
    max_k = int(df.block_id.max())
    for k in range(1, max_k+1):
        idx = df.loc[df['block_id'] == k].index
        # The ending indice needs + 1 to support Python indexing
        indices.append(
            Index(idx.min(), idx.max()+1)
            )
    return indices


def parse_mps_dec(path_mps, path_dec) -> tuple[pd.DataFrame, ...]:
    """Return a tuple of five dataframes that define an instance"""
    
    c_df, A_df, b_df, ineq_df, col_df = parse_mps(path_mps)
    row_order, col_order = get_orders(path_dec, A_df, col_df)
    
    # Permute the rows and columns of the dataframes
    # such that we can use row_indices and col_indices to subset
    # the dataframes according to blocks.
    c_df = c_df.loc[col_order.index]
    A_df = A_df.loc[row_order.index, col_order.index]
    b_df = b_df.loc[row_order.index]
    ineq_df = ineq_df.loc[row_order.index]
    
    col_df = col_df.loc[col_order.index]
    
    row_indices = get_indices(row_order)
    col_indices = get_indices(col_order)
    
    # Find the number of subproblems or blocks
    n_subproblems = len(col_indices)
    
    # If there are master-only variables, then we need to tag them
    # by adding a special pair of indexes to row_indices and col_indices
    # Master-only variables are permuted
    # to the last section of the dataframes (think of the end columns of A_df)
    last_var_idx_in_subp = col_indices[-1].end
    num_total_vars = len(col_df)
    
    # There is a master-only variable, when the last block stops short from
    # covering the rest of the variables.
    if last_var_idx_in_subp != num_total_vars:
        row_indices.append(Index(None, None))
        col_indices.append(Index(last_var_idx_in_subp, num_total_vars))
        

    return DWProblem(
        obj_coeffs = c_df, 
        A = A_df, 
        rhs = b_df, 
        inequalities = ineq_df, 
        var_info = col_df, 
        row_indices = row_indices, 
        col_indices = col_indices, 
        n_subproblems = n_subproblems,
        master_size = row_indices[0].start)


def get_dataframe_orders(path_dec, A_df, col_df) -> tuple[pd.DataFrame, pd.DataFrame]:
    print('\nParsing the DEC file...')
    with open(path_dec, 'r') as f:
        lines = f.readlines()
    # A DEC file states the number of blocks in the line 
    # after the NBLOCKS line
    num_blocks = int(lines[lines.index('NBLOCKS\n')+1])
    row_order = get_row_order(lines, num_blocks) 
    col_order = get_col_order(A_df, row_order, col_df)
    return row_order, col_order


def parse_mps_with_orders(path_mps, row_order, col_order) -> DWProblem:
    
    c_df, A_df, b_df, ineq_df, col_df = parse_mps(path_mps)
    
    # Permute the rows and columns of the dataframes
    # such that we can use row_indices and col_indices to subset
    # the dataframes according to blocks.
    c_df = c_df.loc[col_order.index]
    A_df = A_df.loc[row_order.index, col_order.index]
    b_df = b_df.loc[row_order.index]
    ineq_df = ineq_df.loc[row_order.index]
    
    col_df = col_df.loc[col_order.index]
    
    row_indices = get_indices(row_order)
    col_indices = get_indices(col_order)
    
    # Find the number of subproblems or blocks
    n_subproblems = len(col_indices)
    
    # If there are master-only variables, then we need to tag them
    # by adding a special pair of indexes to row_indices and col_indices
    # Master-only variables are permuted
    # to the last section of the dataframes (think of the end columns of A_df)
    last_var_idx_in_subp = col_indices[-1].end
    num_total_vars = len(col_df)
    
    # There is a master-only variable, when the last block stops short from
    # covering the rest of the variables.
    if last_var_idx_in_subp != num_total_vars:
        row_indices.append(Index(None, None))
        col_indices.append(Index(last_var_idx_in_subp, num_total_vars))
    
    return DWProblem(
        obj_coeffs = c_df, 
        A = A_df, 
        rhs = b_df, 
        inequalities = ineq_df, 
        var_info = col_df, 
        row_indices = row_indices, 
        col_indices = col_indices, 
        n_subproblems = n_subproblems,
        master_size = row_indices[0].start)





    