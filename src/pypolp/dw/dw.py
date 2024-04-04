import numpy as np
import pandas as pd

from pypolp.config import (
    get_dw_max_iter,
    get_dw_improve,
    get_dw_rmpgap,
    get_dw_recover_integer,
    get_dw_verbose,
    get_master_timelimit
)
from pypolp.dw.master import MasterProblem
from pypolp.dw.record import DWRecord
from pypolp.dw.subproblems import Subproblems
from pypolp.problem_class import DWProblem


class DantzigWolfe:
    ''' A manager of the Dantzig-Wolfe implementation.
    '''

    def __init__(
            self,
            max_iter: int = None,
            dw_improve: float = None,
            dw_rmpgap: float = None,
            recover_integer: bool = None,
            master_timelimit: int = None,
            relax_subproblems: bool = None,
            num_threads: int = None
            to_parallel: bool = False
    ):

        self.dw_verbose: bool = get_dw_verbose()
        self.relax_subproblems: bool = relax_subproblems
        self.num_threads: int = num_threads

        # Control whether to solve the subproblems in parallel
        self.to_parallel = to_parallel

        # Use parameters from user_conf.ini if not provided
        if max_iter is None:
            self.MAXITER: int = get_dw_max_iter()
        else:
            self.MAXITER: int = max_iter

        if dw_improve is None:
            self.DWIMPROVE: float = get_dw_improve()
        else:
            self.DWIMPROVE: float = dw_improve

        if dw_rmpgap is None:
            self.RMPGAP: float = get_dw_rmpgap()
        else:
            self.RMPGAP: float = dw_rmpgap

        # The default is not to recover an integer solution
        # We recover an integer solution by reoptimizing the master problem
        # with binary weights.
        if recover_integer is None:
            self.RECOVER_INTEGER: bool = get_dw_recover_integer()

        self.subproblems: Subproblems = None
        self.n_subproblems: int = None

        self.master_problem: MasterProblem = None
        self.master_size: int = None

        self.phase: int = 1
        self.dw_iter: int = None  # Record the final number of iterations

        self.solve_as_binary: bool = False  # Update to true when we solve as binary
        self.rmpgap: float = None  # The gap between RMP and its dual
        self.incre_improve: float = None  # incremental improvement of the DW algorithm

    def fit(self, dw_problem: DWProblem, record: DWRecord) -> None:
        # If there are master-only variables, then the master will contain those.
        # Otherwise, the master is an empty model.
        self.master_size = dw_problem.master_size
        self.master_problem = MasterProblem.fit(
            dw_problem, num_threads=self.num_threads
        )

        # We construct the master problem using proposals from the subproblems.
        # The master problem will collect proposals from subproblem as the first
        # step inside the following for loop.
        # Note that here is the zero-th iteration.
        self.subproblems = Subproblems(
            to_relax=self.relax_subproblems, num_threads=self.num_threads
        )
        self.subproblems.fit(dw_problem)
        self.n_subproblems = self.subproblems.n_subproblems

        # In the beginning, solve the subproblems using their original
        # cost coefficients
        if self.dw_verbose:
            print(f'\n\n==== DW ITER 0 Phase {self.phase} ====')

        if self.to_parallel:
            self.subproblems.parallel_solve(dw_iter=0, record=record)
        else:
            self.subproblems.solve(dw_iter=0, record=record)

    def solve(self, record: DWRecord) -> None:
        objval_old = np.inf
        total_reduced_cost = -np.inf
        dual_bound = -np.inf

        for dw_iter in range(1, self.MAXITER+1):
            # An iteration is counted when the master problem is optimize.
            if self.dw_verbose:
                print(f'\n\n==== DW ITER {dw_iter} Phase {self.phase} ====')
            # Populate the master problem with new columns
            self.master_problem.add_cols_from(record.current_PQs)
            _ = self.master_problem.solve()

            # Update the parameters
            self.phase = self.master_problem.phase
            duals = self.master_problem.get_duals()
            lambs = duals[:self.master_size].reshape(-1, 1)
            alphas = duals[self.master_size:]

            # Decide whether to terminate only when we are in Phase 2
            if self.phase == 2:
                record.add_primal_objval(self.master_problem.objval)

                # Terminate if a condition is met
                # 1) If the change in objval is below a threshold
                objval_new = abs(self.master_problem.objval)
                # Prevent division by zero
                if objval_new == 0:
                    objval_new += 1e-6
                # Add 0.0001 to the denominator to prevent division by zero
                percent_incre_improve = abs(
                    (objval_new - objval_old)) / (0.0001 + objval_new) * 100
                objval_old = objval_new
                if percent_incre_improve <= self.DWIMPROVE:
                    print(
                        f'\nTerminate DW: Improvement is less than tolerance: {round(percent_incre_improve, 4)} %')
                    break

                # 2) If the lower bound improvement is less than threshold
                reduced_costs = [ck - alpha_k for ck,
                                 alpha_k in zip(record.subproblem_objvals, alphas)]
                # Only consider negative reduced costs when picking a variable to enter
                # reduced_costs = [rc for rc in reduced_costs if rc < 0]
                new_total_reduced_cost = sum(reduced_costs)

                # If we do not get new extreme points/rays, then break
                if total_reduced_cost != new_total_reduced_cost:
                    total_reduced_cost = new_total_reduced_cost
                else:
                    print('\nTerminate DW: No new proposal from subproblems')
                    break

                # dual_bound is zero at the first iteration
                if dw_iter > 1:
                    dual_bound = objval_new + total_reduced_cost
                record.add_dual_bound(dual_bound)

                # rmpgap is in percent. Add 0.0001 to the denominator to prevent division by zero
                rmpgap = abs(dual_bound - objval_new) / \
                    (0.0001 + abs(dual_bound))*100
                # total_reduced_cost is zero at the first iteration
                if rmpgap <= self.RMPGAP:
                    print(
                        f'\nTerminate DW: RMPGap is less than tolerance: {round(rmpgap, 4)} %')
                    break
                # Remove all current objective values from record
                record.reset_subproblem_objvals()

                if self.dw_verbose:
                    print(
                        f'{"DW Solve: Incre. improve:":<25} {round(percent_incre_improve, 4)} %')
                    print(f'{"DW Solve: RMPGap:":<25} {round(rmpgap, 4)} %')

            if dw_iter == self.MAXITER:
                print(f'\nTerminate DW: Reached max iteration: {self.MAXITER}')

            if self.to_parallel:
                self.subproblems.parallel_update_solve(
                    dw_phase=self.phase,
                    dw_iter=dw_iter,
                    lambs=lambs,
                    record=record
                )
            else:
                self.subproblems.update_solve(
                    self.phase, dw_iter, lambs, record)

        # Produce an error if we have not reached Phase 2 after reaching the max iteration.
        if not self.phase == 2:
            raise ValueError('DantzigWolfe has not entered phase II.')

        # Record the statistics
        self.dw_iter = dw_iter
        if not self.solve_as_binary:
            self.rmpgap = rmpgap
            self.incre_improve = percent_incre_improve

    def _get_master_vars(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        ''' Return the master variables, master only variables, and the index of master only variables.'''
        # Get X from the master problem
        master_vars = self.master_problem.get_X()

        master_only_var_idx = ~master_vars['variable'].str.contains(
            'B\(', regex=True)
        if master_only_var_idx.sum() == 0:
            master_only_vars = None
        else:
            master_only_vars = master_vars[master_only_var_idx]
            master_only_vars = master_only_vars.set_index('variable')
        return master_vars, master_only_vars, master_only_var_idx

    def _get_betas(self, master_vars: pd.DataFrame, master_only_var_idx: pd.Series) -> pd.DataFrame:
        ''' Return the betas from the master problem.'''
        betas = master_vars[~master_only_var_idx].copy()
        # Label each row of beta with its subproblem_id and iteration_id
        p = r'B\((?P<j>\d+),(?P<i>\d+)\)'
        betas[['j', 'i']] = betas['variable'].str.extract(p)
        betas = betas.astype({'j': int, 'i': int})
        return betas

    def get_solution(
            self, record: DWRecord,
    ) -> tuple[float, pd.DataFrame]:
        ''' Return the final solution and its objective value.'''
        master_vars, master_only_vars, master_only_var_idx = self._get_master_vars()
        betas = self._get_betas(master_vars, master_only_var_idx)

        # For each beta, extract the corresponding solution X from the record
        betas['X'] = betas.apply(
            lambda row: record.get_proposal(row['i'], row['j']).X,
            axis=1
        )

        # Weight each solution by its beta and then do group sum
        betas['weighted_X'] = betas['X'].multiply(betas['value'])
        temp_x = betas.groupby('j').agg({'weighted_X': 'sum'}).values.flatten()

        final_solution = []
        for _, x in enumerate(temp_x):
            final_solution.extend(x)

        master_size = len(final_solution)
        final_solution = pd.DataFrame(
            final_solution,
            index=record.varnames[:master_size])

        final_solution.index = final_solution.index.rename('name')
        final_solution.columns = ['value']
        final_solution = pd.concat([final_solution, master_only_vars], axis=0)

        objval = self.master_problem.objval
        return objval, final_solution

    def reoptimize_with_binary_weights(self) -> None:
        ''' After we have generated extreme points/rays, 
         we might reoptimize the weights as binary variables.
         If the subproblems produce integer solutions, then
         the master problem will produce an integer solution.
         Of course, the solution is likely suboptimal.
        '''
        self.solve_as_binary = True
        self.master_problem.convert_betas_to_binary()
        _ = self.master_problem.solve()

    def reoptimize_with_rounded_weights(self) -> None:
        ''' Round the largest betas from each subproblem to one.
        This can quickly produce an integer solution.
        '''
        # Find the largest beta from each subproblem
        # and round it to one
        master_vars, _, master_only_var_idx = self._get_master_vars()
        betas = self._get_betas(master_vars, master_only_var_idx)
        max_beta_idx = betas.groupby('j')['value'].idxmax()
        # Set all the variables to one
        max_beta_names = betas.loc[max_beta_idx.values, 'variable'].to_list()
        self.master_problem.set_betas_to_one(max_beta_names)
        _ = self.master_problem.solve()

    def get_runtimes_dict(self) -> dict[int | str: list[float]]:
        runtimes = self.subproblems.runtimes
        runtimes['master'] = self.master_problem.runtimes
        return runtimes

    def get_itercounts_dict(self) -> dict[int | str: list[float]]:
        itercounts = self.subproblems.itercounts
        itercounts['master'] = self.master_problem.itercounts
        return itercounts

    def get_stats(self, mode) -> tuple[float | int, float | int]:
        ''' Return the total runtime/itercounts for (master, subproblem)
        '''
        stats: list | dict[int, list] = None
        if mode == 'runtime':
            stats = self.get_runtimes_dict()
        elif mode == 'itercount':
            stats = self.get_itercounts_dict()

        master_stats = None
        subproblem_stats = 0
        for k, v in stats.items():
            if k == 'master':
                master_stats = sum(v)
            else:
                subproblem_stats += sum(v)
        return master_stats, subproblem_stats

    def get_objvals(self) -> list:
        return self.primal_values
