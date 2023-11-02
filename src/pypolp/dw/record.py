import typing
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pypolp.optim import Proposal
from pypolp.problem_class import DWProblem


@dataclass
class ProposalPQ:
    P: np.array
    Q: np.array
    dw_iter: int
    block_id: int
    is_ray: bool



class Record:
    ''' Class to keep track of the proposals. Records only unique proposals.
    
    Example: subproblem_1 yields new solution on dw_ter 1, 3, and 4.
    subproblem_2 yields new solution on the first dw iteration.
    
    Structure of proposals:
    {
     subproblem_1: [Proposal, Proposal, Proposal],
     subproblem_2: [Proposal]
     }
    
    Structure of proposal_ids:
    {
     subproblem_1: [1, 3, 4],
     subproblem_2: [1]
     }
    
    '''
    def __init__(self):
        # Keep the proposal objects of each subproblem
        self.proposals: dict[int: list[Proposal, ...]] = None
        
        # Track the dw iteration number of each proposal
        self.proposal_ids: dict[list[int, ...]] = None
        
        # The master will collect the new_proposals from this list.
        # The list is emptied afterwards.
        # TODO: Modify this to Pool when do multiprocessing
        self.current_PQs: list[ProposalPQ, ...] = None
        
        # Need info to reformulate proposal into Ps and Qs
        self.cjs: dict[int: list[pd.DataFrame]] = None
        self.Ajs: dict[int: list[pd.DataFrame]] = None
        self.n_subproblems: int = None
        self.col_indices: list[typing.NamedTuple[int, int]]
        self.varnames: list['str', ...]
        
        # Save the reduced costs to calculate the lower bound of RMP
        self.subproblem_objvals: list[float, ...] = None
        self.dual_bounds: list[float, ...] = None
        self.primal_objvals: list[float, ...] = None
        

    def _update_proposals(self, proposal: Proposal) -> None:
        self.proposals[proposal.block_id].append(proposal)
        self.proposal_ids[proposal.block_id].append(proposal.dw_iter)
        
        
    def _convert_to_pq(self, proposal: Proposal) -> ProposalPQ:
        X = proposal.X
        block_id = proposal.block_id
        
        cj = self.cjs[block_id]
        Aj = self.Ajs[block_id]
        
        P = np.matmul(cj.T, X)
        
        # The convex coeff is 1 for a point and 0 for a ray
        convex_coeffs = np.zeros(self.n_subproblems)
        if not proposal.is_ray:
            convex_coeffs[proposal.block_id] = 1
        else:
            convex_coeffs[proposal.block_id] = 0
        Q = np.matmul(Aj, X)
        Q = np.r_[Q, convex_coeffs]
        
        proposal_pq = ProposalPQ(
            P = P,
            Q = Q,
            dw_iter = proposal.dw_iter,
            block_id = proposal.block_id,
            is_ray = proposal.is_ray
            )
        return proposal_pq
        
        
    def _add_pq(self, proposal = Proposal) -> None:
        proposal_pq = self._convert_to_pq(proposal)
        print(f'DW Solve: Added B({proposal.block_id}, {proposal.dw_iter})')
        self.current_PQs.append(proposal_pq)
        

    def fit(self, dw_problem: DWProblem) -> None:
        self.proposals = defaultdict(list)
        self.proposal_ids = defaultdict(list)
        
        # TODO: Modify this to Pool when do multiprocessing
        self.current_PQs = []
        
        self.subproblem_objvals = []
        self.dual_bounds = []
        self.primal_objvals = []
        
        self.cjs = []
        self.Ajs = []
        self.n_subproblems = dw_problem.n_subproblems
        self.col_indices = dw_problem.col_indices
        self.varnames = list(dw_problem.A.columns)
        
        # Decompose
        for block_id in range(dw_problem.n_subproblems):
            col_id = dw_problem.col_indices[block_id]
            cj = dw_problem.obj_coeffs.iloc[col_id.start:col_id.end]
            Aj = dw_problem.A.iloc[:dw_problem.master_size, col_id.start:col_id.end]
            self.cjs.append(cj)
            self.Ajs.append(Aj)
            

    def check_unique(self, proposal: Proposal) -> bool:
        block_id = proposal.block_id
        if proposal in self.proposals[block_id]:
            return False
        else:
            return True
    
    
    def update(self, proposal: Proposal) -> None:
        ''' Update the record only when this is the first DW interation
        or the subproblem's proposal is unique.
        '''
        if proposal.dw_iter == 0 or self.check_unique(proposal):
            self._update_proposals(proposal)
            self._add_pq(proposal)
            
        
    def add_subproblem_objval(self, objval_j: float) -> None:
        self.subproblem_objvals.append(objval_j)
        
        
    def reset_subproblem_objvals(self) -> None:
        self.subproblem_objvals = []
        
        
    def add_primal_objval(self, objval_master: float) -> None:
        self.primal_objvals.append(objval_master)
        
        
    def add_dual_bound(self, dual_bound: float) -> None:
        self.dual_bounds.append(dual_bound)
        
        
    def get_proposal(self, dw_iter:int, block_id:int) -> Proposal:
        # Get the position of i in the list
        proposal_idx = self.proposal_ids[block_id].index(dw_iter)
        return self.proposals[block_id][proposal_idx]


