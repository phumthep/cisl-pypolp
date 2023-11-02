from gurobipy import GRB
import gurobipy as gp
import pandas as pd

from pypolp.dw.record import Record, ProposalPQ
from pypolp.optim import OptProblem, DWProblem, GurobipyOptimizer, get_model_from
from pypolp.tools.functions import get_config



def has_master_only_vars(row_indices) -> bool:
    ''' If there is a master-only variable, then the final member of
    row_indices is (None, None)
    '''
    return (row_indices[-1].start is None)


def get_num_blocks(row_indices) -> int:
    if has_master_only_vars(row_indices):
        # The final member is just a placeholder for master-only variables.
        return len(row_indices) - 1
    else:
        return len(row_indices)


def separate_master_vars(master_vars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    master_only_vars = master_vars[~master_vars['variable'].str.contains('B\(', regex=True)]
    master_only_vars = master_only_vars.set_index('variable')
    betas = master_vars[master_vars['variable'].str.contains('B\(', regex=True)].copy()
    return master_only_vars, betas


def generate_convex_names(n_subproblems: int):
    ''' The name of convexity contrainsts are in the format convex_[block_id].
    We recover the final solution using this regex pattern.
    '''
    convex_names = [f'convex_{j}' for j in range(n_subproblems)]
    return convex_names


def build_convex_rhs_ineq(dw_problem: DWProblem) -> tuple[pd.DataFrame, ...]:
    n_subproblems = dw_problem.n_subproblems
    convex_names = generate_convex_names(n_subproblems)
    # The RHS of convexity constraints is 1
    rhs_convex = [1 for _ in range(dw_problem.n_subproblems)]
    rhs_convex = pd.DataFrame(rhs_convex, index=convex_names, columns=['value'])
    # The sign of convexity constraints is the equality
    ineq_convex = pd.DataFrame(['E']*n_subproblems, index=convex_names, columns=['sign'])
    
    return rhs_convex, ineq_convex
    
    

def create_empty_model(dw_problem: DWProblem, to_log: bool) -> gp.Model:
    ''' Create an empty Gurobi model from a list of constraint names
    '''
    master_size = dw_problem.master_size
    n_subproblems = dw_problem.n_subproblems

    constr_names = list(dw_problem.A.index[:dw_problem.master_size])
    convex_names = generate_convex_names(n_subproblems)
    constr_names = constr_names + convex_names

    rhs_convex, ineq_convex = build_convex_rhs_ineq(dw_problem)

    rhs_master = dw_problem.rhs[:master_size]
    rhs_master = pd.concat([rhs_master, rhs_convex], axis=0)

    ineq_master = pd.concat(
        [dw_problem.inequalities[:master_size], ineq_convex],
        axis = 0)
    
    model = gp.Model()
    model.setParam('outputflag', to_log)
    for idx, constr_name in enumerate(constr_names):
        model.addConstr(0 == 0, name=constr_name)
    model.update()
    
    for idx, constr in enumerate(model.getConstrs()):
        constr
        constr.setAttr("RHS", rhs_master.value[idx])
        constr.setAttr("Sense", ineq_master.sign[idx])
    model.update()
    model.display()
    
    return model


class MasterProblem(GurobipyOptimizer):
    def __init__(self, model):
        super().__init__(model)
        self.objval: float = None
        self.phase: int = None
        
        self.runtimes: list[float] = None
        self.itercounts: list[int] = None
    
    @staticmethod
    def _get_opt_problem(dw_problem: DWProblem) -> OptProblem:
        ''' In case there are master-ony variables, put those variables
        in the master problem in their original form. Also, need to add
        the convexity constraints.
        '''
        col_id = dw_problem.col_indices[-1]
        
        obj_coeffs = dw_problem.obj_coeffs.iloc[col_id.start:]
        A = dw_problem.A.iloc[:dw_problem.master_size, col_id.start:]
        rhs = dw_problem.rhs.iloc[:dw_problem.master_size]
        inequalities = dw_problem.inequalities.iloc[:dw_problem.master_size]
        var_info = dw_problem.var_info.iloc[col_id.start:]
        
        # Create the section for convexity constraints
        convex_names = generate_convex_names(dw_problem.n_subproblems)
        
        A_convex = pd.DataFrame(0, index=convex_names, columns=A.columns)
        rhs_convex, ineq_convex = build_convex_rhs_ineq(dw_problem)
        
        A = pd.concat([A, A_convex], axis=0)
        
        rhs = pd.concat([rhs, rhs_convex], axis=0)
    
        inequalities = pd.concat([inequalities, ineq_convex], axis = 0)
        
        opt_problem = OptProblem(
            obj_coeffs, A, rhs, inequalities, var_info)
        return opt_problem
    
    
    def _add_col(self, proposal_pq: ProposalPQ) -> None:
        varname = f'B({proposal_pq.block_id},{proposal_pq.dw_iter})'
        
        if not (len(proposal_pq.Q) == self.model.getAttr('numconstrs')):
            raise ValueError(
                'The shape of Q does not match the number of constraints in the master problem.')
        
        constrs = self.model.getConstrs()
        
        # A ray may get scaled by the weight variable without an upperbound.
        if not proposal_pq.is_ray:
            self.model.addVar(
                lb = 0, #ub = 1,
                obj = proposal_pq.P.iloc[0], 
                vtype = GRB.CONTINUOUS, 
                name = varname,
                column = gp.Column(proposal_pq.Q, constrs)
                )
        else:
            self.model.addVar(
                lb = 0,
                obj = proposal_pq.P, 
                vtype = GRB.CONTINUOUS, 
                name = varname,
                column = gp.Column(proposal_pq.Q, constrs)
                )
        
        
    def add_cols_from(self, record: Record) -> None:
        ''' Add new columns using extreme points/rays from the subproblems.
        '''
        while record.current_PQs:
            proposal_pq = record.current_PQs.pop()
            self._add_col(proposal_pq)
        self.model.update()
        

    def solve(self):
        # print('\nDW Solve: Master Problem\n')
        solution = self.optimize()
        
        # We did not initiate the class with an empty list of runtimes
        if self.runtimes is None:
            self.runtimes = []
            self.itercounts = []
        
        if self.model.status == 3:
            self.phase = 1
        else:
            self.phase = 2
            
        #TODO: Save only when doing analysis
        self.runtimes.append(self.runtime)
        self.itercounts.append(self.itercount)
        
        return solution
    
    
    def convert_betas_to_int(self) -> None:
        '''
        Change the beta variables to integer and re-optimize.
        '''
        master_vars = self.model.getVars()
        # The names of beta variables are in the format B(block_id, dw_iter)
        betas = [gp_var for gp_var in master_vars if gp_var.varname.startswith('B(')]
        for v in betas:
            v.setAttr('VType', GRB.INTEGER)
        
    
    
    @classmethod
    def fit(cls, dw_problem: DWProblem) -> None:
        '''
        '''
        config = get_config()
        debug = int(config['DWMASTER']['DEBUG'])
        
        has_mov = has_master_only_vars(dw_problem.row_indices)
        
        if not has_mov:
            model = create_empty_model(dw_problem, to_log=debug)
            model.setParam('outputflag', debug)
        else:
            opt_problem = cls._get_opt_problem(dw_problem)
            model = get_model_from(opt_problem, to_log=debug)
        
        return cls(model)
    



