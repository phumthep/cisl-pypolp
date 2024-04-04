from __future__ import annotations
from collections import defaultdict
import multiprocessing as mp
from gurobipy import GRB
import gurobipy as gp
import numpy as np
import pandas as pd

from pypolp.optim import Solution, Proposal

from pypolp.config import (
    get_subp_warmstart,
    get_subp_verbose,
    get_subp_mipgap,
    get_subp_relax,
    get_gp_record,
)
from pypolp.dw.record import DWRecord
from pypolp.optim import GurobipyOptimizer, Proposal
from pypolp.problem_class import DWProblem


# -------------- Section break
def _build_and_solve(problem_tuple):
    with gp.Env() as env, gp.Model(env=env) as model:
        # Change this to subproblem.fit
        '''subproblem = Subproblem.get_gp_model_from_dataframes(
            obj_coeffs=problem_tuple[0],
            A=problem_tuple[1],
            rhs=problem_tuple[2],
            inequalities=problem_tuple[3],
            var_info=problem_tuple[4],
            model_name=problem_tuple[5]
        )'''

        # model.setParam('Threads', 1)
        model.setParam('outputflag', get_subp_verbose())

        # Extract the problem_tuple
        obj_coeffs = problem_tuple[0]
        A = problem_tuple[1]
        rhs = problem_tuple[2]
        inequalities = problem_tuple[3]
        var_info = problem_tuple[4]
        model_name = problem_tuple[5]

        # Add variables to the model
        x = model.addMVar(
            shape=var_info.shape[0],
            name=var_info.index
        )
        x.setAttr('lb', var_info.lower.values)
        x.setAttr('ub', var_info.upper.values)
        x.setAttr('vtype', var_info.type.values)

        # Define the objective value
        model.setObjective(
            expr=obj_coeffs.values.T @ x,
            sense=GRB.MINIMIZE)

        # Add constraints
        model_constrs = model.addMConstr(
            A.values, x, inequalities['value'].values, rhs.values.reshape(
                -1)
        )
        model_constrs.setAttr('constrname', A.index)

        # After optimize, check if the optimization is bounded or unbounded
        model.optimize()
        if model.Status == 2:
            is_ray = False
        elif model.Status == 5:
            is_ray = True
        else:
            raise ValueError(f'Unsupported status: {model.Status}')
        # Workflow requires the Solution object
        solution = Solution(
            X=np.array(model.x),
            objval=model.objval,
            is_ray=is_ray
        )
        # outputs
        runtime = model.Runtime
        itercount = model.IterCount
        return runtime, itercount, solution


def _update_and_solve(problem_tuple, A_master, dw_phase, lambs):
    with gp.Env() as env, gp.Model(env=env) as model:
        # Change this to subproblem.fit
        '''subproblem = Subproblem.get_gp_model_from_dataframes(
            obj_coeffs=problem_tuple[0],
            A=problem_tuple[1],
            rhs=problem_tuple[2],
            inequalities=problem_tuple[3],
            var_info=problem_tuple[4],
            model_name=problem_tuple[5]
        )'''

        # model.setParam('Threads', 1)
        model.setParam('outputflag', get_subp_verbose())

        # Extract the problem_tuple
        obj_coeffs = problem_tuple[0]
        A = problem_tuple[1]
        rhs = problem_tuple[2]
        inequalities = problem_tuple[3]
        var_info = problem_tuple[4]
        model_name = problem_tuple[5]

        # Add variables to the model
        x = model.addMVar(
            shape=var_info.shape[0],
            name=var_info.index
        )
        x.setAttr('lb', var_info.lower.values)
        x.setAttr('ub', var_info.upper.values)
        x.setAttr('vtype', var_info.type.values)

        # Define the objective value based on the Phase of Dantzig-Wolfe
        if dw_phase == 1:
            new_c = np.matmul(lambs.T, A_master.values)
            new_c = new_c.flatten()
            model.setObjective(
                expr=new_c @ x,
                sense=GRB.MINIMIZE)
        else:
            new_c = obj_coeffs.values.T - \
                np.matmul(lambs.T, A_master.values)
            new_c = new_c.flatten()
            model.setObjective(
                expr=new_c @ x,
                sense=GRB.MINIMIZE)

        # Add constraints
        model_constrs = model.addMConstr(
            A.values, x, inequalities['value'].values, rhs.values.reshape(
                -1)
        )
        model_constrs.setAttr('constrname', A.index)

        # After optimize, check if the optimization is bounded or unbounded
        model.optimize()
        if model.Status == 2:
            is_ray = False
        elif model.Status == 5:
            is_ray = True
        else:
            raise ValueError(f'Unsupported status: {model.Status}')
        # Workflow requires the Solution object
        solution = Solution(
            X=np.array(model.x),
            objval=model.objval,
            is_ray=is_ray
        )
        # outputs
        runtime = model.Runtime
        itercount = model.IterCount
        return runtime, itercount, solution


class Subproblem(GurobipyOptimizer):
    ''' This class extends GurobipyOptimizer to keep block-specific information
    and provide functionalities to update the objective coefficients.
    '''

    def __init__(
            self,
            model: gp.Model,
            warmstart: bool,
            mipgap: float,
            verbose: bool,
            to_relax: bool,
            num_threads: int,
    ):
        # If the user did not specify, then use values from user_config.ini
        if warmstart is None:
            warmstart = get_subp_warmstart()

        if mipgap is None:
            mipgap = get_subp_mipgap()

        if verbose is None:
            self.verbose = get_subp_verbose()
        else:
            self.verbose = verbose

        if to_relax is None:
            to_relax = get_subp_relax()

        # Inherit methods from GurobipyOptimizer class
        super().__init__(
            model=model,
            warmstart=warmstart,
            mipgap=mipgap,
            verbose=self.verbose,
            to_record=True,
            to_relax=to_relax,
            num_threads=num_threads
        )
        # Add-on attribute
        self.block_id: int = None
        self.cost_coeffs = None  # original cost coefficients
        self.A_master = None  # constraint coeffs in the master problem

        # Set number of threads to 1
        self.model.setParam('Threads', 1)

    def set_id(self, block_id: int) -> None:
        ''' Label Subproblem class with an ID and change the gp.Model's name
        using the format 'subproblem_{block_id}.'
        '''
        self.block_id = block_id
        self.model.setAttr('ModelName', f'subproblem_{block_id}')

    def set_c_A(self, cost_coeffs: pd.DataFrame, A_master: pd.DataFrame) -> None:
        ''' These are coefficients in the master problem section.
        '''
        self.cost_coeffs = cost_coeffs
        self.A_master = A_master

    def _update_farkas(self, lambs: np.array) -> None:
        x = np.array(self.model.getVars())
        new_c = np.matmul(lambs.T, self.A_master.values)
        new_c = new_c.flatten()
        self.model.setObjective(
            expr=new_c @ x,
            sense=GRB.MINIMIZE)

    def _update_reduced_cost(self, lambs: np.array) -> None:
        x = np.array(self.model.getVars())
        new_c = self.cost_coeffs.values.T - \
            np.matmul(lambs.T, self.A_master.values)
        new_c = new_c.flatten()
        self.model.setObjective(
            expr=new_c @ x,
            sense=GRB.MINIMIZE)

    def update_c(self, dw_phase: int, lambs: np.array) -> None:
        """
        Update the cost cofficients either with Farkas pricing or reduced cost pricing
        """
        if dw_phase == 1:
            self._update_farkas(lambs)
        else:
            self._update_reduced_cost(lambs)
        # Gurobi requires manually updating the model
        self.model.update()

    @classmethod
    def fit(
            cls,
            model: gp.Model,
            warmstart: bool,
            mipgap: float,
            verbose: bool,
            to_relax: bool,
            num_threads: int,
    ) -> Subproblem:
        return cls(
            model=model,
            warmstart=warmstart,
            mipgap=mipgap,
            verbose=verbose,
            to_relax=to_relax,
            num_threads=num_threads
        )


# -------------- Section break

class Subproblems():
    ''' This class is a collection of subproblems.
    '''

    def __init__(
            self,
            verbose: bool = None,
            to_record: bool = None,
            to_relax: bool = None,
            num_threads: int = None
    ):
        self.to_relax: bool = to_relax
        self.num_threads: int = num_threads

        # Use parameters from user_conf.ini if not provided
        if verbose is None:
            self.verbose: bool = get_subp_verbose()
        else:
            self.verbose: bool = verbose

        if to_record is None:
            self.to_record: bool = get_gp_record()
        else:
            self.to_record: bool = to_record

        self.all_subproblems: list[Subproblem, ...] = None
        self.n_subproblems: int = None

        # Track the statistics of each subproblem
        self.runtimes: dict[int, list[float, ...]] = None
        self.itercounts: dict[int, list[int, ...]] = None

        # Store problem_tuples to build subproblems
        self.problem_tuples: list[tuple[pd.DataFrame, pd.DataFrame,
                                        pd.DataFrame, pd.DataFrame, pd.DataFrame, str]] = None

        # Store original parameters to update_sole subproblems
        self.original_c: list[pd.DataFrame, ...] = None
        self.original_A_master: list[pd.DataFrame, ...] = None

    def _record_opt_stats(self, block_id: int, runtime: int, itercount: float) -> None:
        self.runtimes[block_id].append(runtime)
        self.itercounts[block_id].append(itercount)

    def fit(
            self,
            dw_problem: DWProblem,
            warmstart: bool = None,
            mipgap: float = None,
            verbose: bool = None,
            to_record: bool = None,
    ) -> None:
        ''' Chop the constraint matrix and create subproblems
        '''
        # Call subproblem creator
        self.all_subproblems = []
        self.runtimes = defaultdict(list)
        self.itercounts = defaultdict(list)
        self.problem_tuples = []
        self.original_c = []
        self.original_A_master = []
        self.n_subproblems = dw_problem.n_subproblems

        # Loop thru to create subproblems
        for block_id in range(self.n_subproblems):

            row_id = dw_problem.row_indices[block_id]
            col_id = dw_problem.col_indices[block_id]

            # Create problem_tuple to build subproblems in parallel implementation
            problem_tuple = (
                dw_problem.obj_coeffs.iloc[col_id.start: col_id.end],
                dw_problem.A.iloc[row_id.start: row_id.end,
                                  col_id.start: col_id.end],
                dw_problem.rhs.iloc[row_id.start: row_id.end],
                dw_problem.inequalities.iloc[row_id.start: row_id.end],
                dw_problem.var_info.iloc[col_id.start: col_id.end],
                f'subproblem_{block_id}'
            )
            self.problem_tuples.append(problem_tuple)

            # Create original_c for parallel update_solve
            self.original_c.append(
                dw_problem.obj_coeffs.iloc[col_id.start: col_id.end]
            )

            # Create original_A_master for parallel update_solve
            self.original_A_master.append(
                dw_problem.A.iloc[:dw_problem.master_size,
                                  col_id.start: col_id.end]
            )

            # Create a subproblem for serial implementation
            subproblem = Subproblem.fit(
                model=Subproblem.get_gp_model_from_dataframes(
                    obj_coeffs=dw_problem.obj_coeffs.iloc[col_id.start: col_id.end],
                    A=dw_problem.A.iloc[row_id.start: row_id.end,
                                        col_id.start: col_id.end],
                    rhs=dw_problem.rhs.iloc[row_id.start: row_id.end],
                    inequalities=dw_problem.inequalities.iloc[row_id.start: row_id.end],
                    var_info=dw_problem.var_info.iloc[col_id.start: col_id.end],
                    model_name=f'subproblem_{block_id}'
                ),
                warmstart=warmstart,
                mipgap=mipgap,
                verbose=verbose,
                to_relax=self.to_relax,
                num_threads=self.num_threads
            )

            # Record block specific information
            subproblem.set_c_A(
                cost_coeffs=dw_problem.obj_coeffs.iloc[col_id.start:col_id.end],
                A_master=dw_problem.A.iloc[:dw_problem.master_size,
                                           col_id.start:col_id.end]
            )
            subproblem.set_id(block_id)
            self.all_subproblems.append(subproblem)

    def solve(self, dw_iter, record: DWRecord) -> None:
        for block_id, subproblem in enumerate(self.all_subproblems):
            if self.verbose:
                print(f'\n----- DW Solve: Subproblem {block_id}\n')

            solution = subproblem.optimize()
            if self.to_record:
                self._record_opt_stats(
                    block_id, subproblem.runtime, subproblem.itercount)

            record.update(
                Proposal.from_solution(
                    solution,
                    dw_iter,
                    block_id
                )
            )

    def update_solve(
            self,
            dw_phase: int,
            dw_iter: int,
            lambs: np.array,
            record: DWRecord
    ) -> None:
        ''' Call this method after the second DW iteration. We coupled
        the update and optimize steps together to save the overhead from using
        two for loops.
        '''
        for block_id, subproblem in enumerate(self.all_subproblems):
            subproblem.update_c(dw_phase, lambs)

            if self.verbose:
                print(f'\n----- DW Solve: Subproblem {block_id}\n')
            solution = subproblem.optimize()
            record.add_subproblem_objval(solution.objval)

            if self.to_record:
                self._record_opt_stats(
                    block_id, subproblem.runtime, subproblem.itercount)

            record.update(
                Proposal.from_solution(
                    solution,
                    dw_iter,
                    block_id
                )
            )

    def parallel_solve(self, dw_iter, record: DWRecord) -> None:
        ''' This method is used to create subproblems in parallel.
        '''

        # Optimize the models in parallel
        with mp.Pool() as pool:
            temp_list = pool.map(
                _build_and_solve, self.problem_tuples)

        # Update self.runtimes and self.itercounts
        for block_id, (runtime, itercount, solution) in enumerate(temp_list):
            self._record_opt_stats(
                block_id, runtime, itercount)
            # Proposal object is a solution object with additional attributes:
            # the DW iteration number and the subproblem ID
            record.update(
                Proposal.from_solution(
                    solution,
                    dw_iter,
                    block_id
                )
            )

    def parallel_update_solve(
            self,
            dw_phase: int,
            dw_iter: int,
            lambs: np.array,
            record: DWRecord
    ) -> None:
        # Optimize the models in parallel
        # Create a list of tuples to pass to starmap
        combined_list = list(
            zip(
                self.problem_tuples,
                self.original_A_master,
                [dw_phase]*self.n_subproblems,
                [lambs] * self.n_subproblems
            )
        )

        with mp.Pool() as pool:
            temp_list = pool.starmap(_update_and_solve, combined_list)

        # Update self.runtimes and self.itercounts
        for block_id, (runtime, itercount, solution) in enumerate(temp_list):

            # Assume that the objvals are returned in order by the subproblem ID
            record.add_subproblem_objval(solution.objval)

            self._record_opt_stats(
                block_id, runtime, itercount)
            # Proposal object is a solution object with additional attributes:
            # the DW iteration number and the subproblem ID
            record.update(
                Proposal.from_solution(
                    solution,
                    dw_iter,
                    block_id
                )
            )
