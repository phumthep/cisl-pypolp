from __future__ import annotations
from collections import defaultdict
import math
import multiprocessing as mp
from multiprocessing import shared_memory
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


def _parallel_omp_solve(
    block_ids, problem_tuples, A_masters, shared_input_tuple, comm_queue, barrier
):
    """
    initiates a single worker process that sets up the models for the given problem parameters and
    iteratively optimizes the model until the main process specifies termination through the shared input.

    The worker process initializes its own shared memories for each subproblem to efficiently send solutions
    over to the main process. An alternative could be through the provided communication queue but repeatedly
    pickling large or complex numpy data is unefficient. The queue is used to share the names of the created shared
    memories so the main process can find the appropriate solutions for each subproblem.
    """
    with gp.Env() as env:
        # Extract the shared inputs (data coming from parent process)
        shared_input_name: str = shared_input_tuple[0]
        lambs_shape: tuple = shared_input_tuple[1]
        # the first 4 bytes (in shared input) is a boolean (whether to terminate iterating)
        # the next 4 bytes belong to the integer dw_phase
        # the remaining bytes belong to lambs
        shared_input = shared_memory.SharedMemory(name=shared_input_name, create=False)
        flags = np.ndarray(shape=(2,), dtype=np.int32, buffer=shared_input.buf[0:8])
        lambs = np.ndarray(lambs_shape, dtype=np.float64, buffer=shared_input.buf[8:])

        # the list of shared memory segments created by this process, the parent process is responsible for unlinking/disposing.
        shared_memory_objects: list[shared_memory.SharedMemory] = []

        # the local data necessary for each update iteration
        models = []
        model_obj_coeffs = []
        Xbuffers: list[np.array] = []  # one array contains the model soluion (model.x)
        metrics: list[np.array] = (
            []
        )  # one array contains: solution_size, objval, is_ray, runtime, itercount

        def update_solutions(
            index: int,
            X: np.array,
            objval: float,
            is_ray: bool,
            runtime: float,
            itercount: float,
        ):
            """
            instead of creating a solution object and returning the tuple(runtime, itercount, solution),
            this loads all of the above data into the shared memory segment (which will be read by the parent process)
            """
            metric = metrics[index]
            metric[1] = objval
            metric[2] = is_ray
            metric[3] = runtime
            metric[4] = itercount
            Xbuffers[index][:] = X
            return

        for i, problem_tuple in enumerate(problem_tuples):
            model = gp.Model(env=env)
            model.setParam("outputflag", get_subp_verbose())
            # Extract the problem_tuple
            obj_coeffs, A, rhs, inequalities, var_info, model_name = problem_tuple
            model_obj_coeffs.append(obj_coeffs)
            # Add variables to the model
            x = model.addMVar(shape=var_info.shape[0], name=var_info.index)
            x.setAttr("lb", var_info.lower.values)
            x.setAttr("ub", var_info.upper.values)
            x.setAttr("vtype", var_info.type.values)

            # Define the objective value
            model.setObjective(expr=obj_coeffs.values.T @ x, sense=GRB.MINIMIZE)

            # Add constraints
            model_constrs = model.addMConstr(
                A.values, x, inequalities["value"].values, rhs.values.reshape(-1)
            )
            model_constrs.setAttr("constrname", A.index)

            # After optimize, check if the optimization is bounded or unbounded
            model.optimize()

            if model.Status == 2:
                is_ray = False
            elif model.Status == 5:
                is_ray = True
            else:
                raise ValueError(f"Unsupported status: {model.Status}")

            X = np.array(model.x)
            solution_size = np.prod(X.shape)  # 8 bytes per float
            objval = model.objval
            runtime = model.Runtime
            itercount = model.IterCount
            # 5 additional values (solution size, objval, is_ray, runtime, itercount)
            shared_mem = shared_memory.SharedMemory(
                create=True, size=8 * solution_size + 40
            )
            shared_memory_objects.append(shared_mem)
            metric = np.ndarray(
                shape=(5,), dtype=np.float64, buffer=shared_mem.buf[:40]
            )  # first 40 bytes
            metric[0] = (
                solution_size  # this remains the same per subproblem throughout all iterations
            )
            metrics.append(metric)
            Xbuffer = np.ndarray(
                shape=X.shape, dtype=np.float64, buffer=shared_mem.buf[40:]
            )  # remaining bytes
            Xbuffers.append(Xbuffer)
            update_solutions(i, X, objval, is_ray, runtime, itercount)
            models.append(model)

        # communicate the shared memory segments (blocking communication)
        # this could be avoided if we can guarantee uniqueness of names like "subproblem0", "subproblem1" for shared memory blocks
        for index, shared_mem in enumerate(shared_memory_objects):
            block_id = block_ids[index]
            comm_queue.put((block_id, shared_mem.name))

        while True:
            # iteration 1 and onwards
            barrier.wait()  # all processes wait for main process to decide whether to terminate
            terminate = flags[0]
            if terminate:
                break
            dw_phase = flags[1]

            for index, model in enumerate(models):
                x = np.array(model.getVars())
                if dw_phase == 1:
                    new_c = np.matmul(lambs.T, A_masters[index].values)
                    new_c = new_c.flatten()
                    model.setObjective(expr=new_c @ x, sense=GRB.MINIMIZE)
                else:
                    obj_coeffs = model_obj_coeffs[index]
                    new_c = obj_coeffs.values.T - np.matmul(
                        lambs.T, A_masters[index].values
                    )
                    new_c = new_c.flatten()
                    model.setObjective(expr=new_c @ x, sense=GRB.MINIMIZE)
                model.update()
                model.optimize()
                if model.Status == 2:
                    is_ray = False
                elif model.Status == 5:
                    is_ray = True
                else:
                    raise ValueError(f"Unsupported status: {model.Status}")
                X = np.array(model.x)
                objval = model.objval
                runtime = model.Runtime
                itercount = model.IterCount
                update_solutions(index, X, objval, is_ray, runtime, itercount)
            barrier.wait()  # all processes come to this point, then main process reads solutions


def _build_and_solve(problem_tuple):
    with gp.Env() as env, gp.Model(env=env) as model:
        # model.setParam('Threads', 1)
        model.setParam("outputflag", get_subp_verbose())

        # Extract the problem_tuple
        obj_coeffs = problem_tuple[0]
        A = problem_tuple[1]
        rhs = problem_tuple[2]
        inequalities = problem_tuple[3]
        var_info = problem_tuple[4]
        model_name = problem_tuple[5]

        # Add variables to the model
        x = model.addMVar(shape=var_info.shape[0], name=var_info.index)
        x.setAttr("lb", var_info.lower.values)
        x.setAttr("ub", var_info.upper.values)
        x.setAttr("vtype", var_info.type.values)

        # Define the objective value
        model.setObjective(expr=obj_coeffs.values.T @ x, sense=GRB.MINIMIZE)

        # Add constraints
        model_constrs = model.addMConstr(
            A.values, x, inequalities["value"].values, rhs.values.reshape(-1)
        )
        model_constrs.setAttr("constrname", A.index)

        # After optimize, check if the optimization is bounded or unbounded
        model.optimize()
        if model.Status == 2:
            is_ray = False
        elif model.Status == 5:
            is_ray = True
        else:
            raise ValueError(f"Unsupported status: {model.Status}")
        # Workflow requires the Solution object
        solution = Solution(X=np.array(model.x), objval=model.objval, is_ray=is_ray)
        # outputs
        runtime = model.Runtime
        itercount = model.IterCount
        return runtime, itercount, solution


def _update_and_solve(problem_tuple, A_master, dw_phase, lambs):
    with gp.Env() as env, gp.Model(env=env) as model:
        # Change this to subproblem.fit
        """subproblem = Subproblem.get_gp_model_from_dataframes(
            obj_coeffs=problem_tuple[0],
            A=problem_tuple[1],
            rhs=problem_tuple[2],
            inequalities=problem_tuple[3],
            var_info=problem_tuple[4],
            model_name=problem_tuple[5]
        )"""

        # model.setParam('Threads', 1)
        model.setParam("outputflag", get_subp_verbose())

        # Extract the problem_tuple
        obj_coeffs = problem_tuple[0]
        A = problem_tuple[1]
        rhs = problem_tuple[2]
        inequalities = problem_tuple[3]
        var_info = problem_tuple[4]
        model_name = problem_tuple[5]

        # Add variables to the model
        x = model.addMVar(shape=var_info.shape[0], name=var_info.index)
        x.setAttr("lb", var_info.lower.values)
        x.setAttr("ub", var_info.upper.values)
        x.setAttr("vtype", var_info.type.values)

        # Define the objective value based on the Phase of Dantzig-Wolfe
        if dw_phase == 1:
            new_c = np.matmul(lambs.T, A_master.values)
            new_c = new_c.flatten()
            model.setObjective(expr=new_c @ x, sense=GRB.MINIMIZE)
        else:
            new_c = obj_coeffs.values.T - np.matmul(lambs.T, A_master.values)
            new_c = new_c.flatten()
            model.setObjective(expr=new_c @ x, sense=GRB.MINIMIZE)

        # Add constraints
        model_constrs = model.addMConstr(
            A.values, x, inequalities["value"].values, rhs.values.reshape(-1)
        )
        model_constrs.setAttr("constrname", A.index)

        # After optimize, check if the optimization is bounded or unbounded
        model.optimize()
        if model.Status == 2:
            is_ray = False
        elif model.Status == 5:
            is_ray = True
        else:
            raise ValueError(f"Unsupported status: {model.Status}")
        # Workflow requires the Solution object
        solution = Solution(X=np.array(model.x), objval=model.objval, is_ray=is_ray)
        # outputs
        runtime = model.Runtime
        itercount = model.IterCount
        return runtime, itercount, solution


class Subproblem(GurobipyOptimizer):
    """This class extends GurobipyOptimizer to keep block-specific information
    and provide functionalities to update the objective coefficients.
    """

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
            num_threads=num_threads,
        )
        # Add-on attribute
        self.block_id: int = None
        self.cost_coeffs = None  # original cost coefficients
        self.A_master = None  # constraint coeffs in the master problem

        # Set number of threads to 1
        self.model.setParam("Threads", 1)

    def set_id(self, block_id: int) -> None:
        """Label Subproblem class with an ID and change the gp.Model's name
        using the format 'subproblem_{block_id}.'
        """
        self.block_id = block_id
        self.model.setAttr("ModelName", f"subproblem_{block_id}")

    def set_c_A(self, cost_coeffs: pd.DataFrame, A_master: pd.DataFrame) -> None:
        """These are coefficients in the master problem section."""
        self.cost_coeffs = cost_coeffs
        self.A_master = A_master

    def _update_farkas(self, lambs: np.array) -> None:
        x = np.array(self.model.getVars())
        new_c = np.matmul(lambs.T, self.A_master.values)
        new_c = new_c.flatten()
        self.model.setObjective(expr=new_c @ x, sense=GRB.MINIMIZE)

    def _update_reduced_cost(self, lambs: np.array) -> None:
        x = np.array(self.model.getVars())
        new_c = self.cost_coeffs.values.T - np.matmul(lambs.T, self.A_master.values)
        new_c = new_c.flatten()
        self.model.setObjective(expr=new_c @ x, sense=GRB.MINIMIZE)

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
            num_threads=num_threads,
        )


# -------------- Section break


class Subproblems:
    """This class is a collection of subproblems."""

    def __init__(
        self,
        verbose: bool = None,
        to_record: bool = None,
        to_relax: bool = None,
        num_threads: int = None,
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
        self.problem_tuples: list[
            tuple[
                pd.DataFrame,
                pd.DataFrame,
                pd.DataFrame,
                pd.DataFrame,
                pd.DataFrame,
                str,
            ]
        ] = None

        # Store original parameters to update_solve subproblems
        self.original_c: list[pd.DataFrame, ...] = None
        self.original_A_master: list[pd.DataFrame, ...] = None

        # Store the subprocesses and synchronization structures used in parallel implementations (parallel_solve, parallel_update_solve)
        self.worker_processes: list[mp.Process] = None
        self.worker_comm_queue: mp.Queue = None
        self.worker_shared_input: shared_memory.SharedMemory = None
        self.worker_shared_memories: list[shared_memory.SharedMemory] = None
        self.worker_barrier: mp.Barrier = None

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
        """Chop the constraint matrix and create subproblems"""
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
                dw_problem.obj_coeffs.iloc[col_id.start : col_id.end],
                dw_problem.A.iloc[row_id.start : row_id.end, col_id.start : col_id.end],
                dw_problem.rhs.iloc[row_id.start : row_id.end],
                dw_problem.inequalities.iloc[row_id.start : row_id.end],
                dw_problem.var_info.iloc[col_id.start : col_id.end],
                f"subproblem_{block_id}",
            )
            self.problem_tuples.append(problem_tuple)

            # TODO: remove this, already included in problem_tuple
            # Create original_c for parallel update_solve
            self.original_c.append(
                dw_problem.obj_coeffs.iloc[col_id.start : col_id.end]
            )

            # Create original_A_master for parallel update_solve
            self.original_A_master.append(
                dw_problem.A.iloc[: dw_problem.master_size, col_id.start : col_id.end]
            )

            # Create a subproblem for serial implementation
            subproblem = Subproblem.fit(
                model=Subproblem.get_gp_model_from_dataframes(
                    obj_coeffs=dw_problem.obj_coeffs.iloc[col_id.start : col_id.end],
                    A=dw_problem.A.iloc[
                        row_id.start : row_id.end, col_id.start : col_id.end
                    ],
                    rhs=dw_problem.rhs.iloc[row_id.start : row_id.end],
                    inequalities=dw_problem.inequalities.iloc[
                        row_id.start : row_id.end
                    ],
                    var_info=dw_problem.var_info.iloc[col_id.start : col_id.end],
                    model_name=f"subproblem_{block_id}",
                ),
                warmstart=warmstart,
                mipgap=mipgap,
                verbose=verbose,
                to_relax=self.to_relax,
                num_threads=self.num_threads,
            )

            # Record block specific information
            subproblem.set_c_A(
                cost_coeffs=dw_problem.obj_coeffs.iloc[col_id.start : col_id.end],
                A_master=dw_problem.A.iloc[
                    : dw_problem.master_size, col_id.start : col_id.end
                ],
            )
            subproblem.set_id(block_id)
            self.all_subproblems.append(subproblem)

    def solve(self, dw_iter, record: DWRecord) -> None:
        for block_id, subproblem in enumerate(self.all_subproblems):
            if self.verbose:
                print(f"\n----- DW Solve: Subproblem {block_id}\n")

            solution = subproblem.optimize()
            if self.to_record:
                self._record_opt_stats(
                    block_id, subproblem.runtime, subproblem.itercount
                )

            record.update(Proposal.from_solution(solution, dw_iter, block_id))

    def update_solve(
        self, dw_phase: int, dw_iter: int, lambs: np.array, record: DWRecord
    ) -> None:
        """Call this method after the second DW iteration. We coupled
        the update and optimize steps together to save the overhead from using
        two for loops.
        """
        for block_id, subproblem in enumerate(self.all_subproblems):
            subproblem.update_c(dw_phase, lambs)

            if self.verbose:
                print(f"\n----- DW Solve: Subproblem {block_id}\n")
            solution = subproblem.optimize()
            record.add_subproblem_objval(solution.objval)

            if self.to_record:
                self._record_opt_stats(
                    block_id, subproblem.runtime, subproblem.itercount
                )
            record.update(Proposal.from_solution(solution, dw_iter, block_id))

    def parallel_solve(self, dw_iter, record: DWRecord, lambs_shape, procs) -> None:
        """This method is used to create subproblems in parallel."""
        problems_per_proc = math.ceil(
            self.n_subproblems / procs
        )  # could distribute a bit more evenly
        self.worker_comm_queue = mp.Queue()
        self.worker_barrier = mp.Barrier(
            parties=procs + 1
        )  # includes worker and main process
        self.worker_processes = []
        self.worker_shared_memories = [None] * self.n_subproblems
        lambs_arr_bytes = (
            np.prod(lambs_shape, dtype=np.int32) * np.dtype(np.float64).itemsize
        )
        self.worker_shared_input = shared_memory.SharedMemory(
            create=True, size=8 + lambs_arr_bytes
        )
        shared_input_data = (self.worker_shared_input.name, lambs_shape)
        for i in range(procs):
            low = i * problems_per_proc
            high = (i + 1) * problems_per_proc  # exclusive
            block_ids = [id for id in range(low, high)]
            problem_tuples = self.problem_tuples[low:high]
            A_masters = self.original_A_master[low:high]
            process = mp.Process(
                target=_parallel_omp_solve,
                args=(
                    block_ids,
                    problem_tuples,
                    A_masters,
                    shared_input_data,
                    self.worker_comm_queue,
                    self.worker_barrier,
                ),
            )
            self.worker_processes.append(process)
            process.start()  # start the processes

        # get the shared memories allocated by each subprocess (a single process may send multiple)
        # read the solution and update records/stats
        for _ in range(self.n_subproblems):
            (block_id, shared_memory_name) = self.worker_comm_queue.get()
            shm = shared_memory.SharedMemory(name=shared_memory_name)
            self.worker_shared_memories[block_id] = shm

        for block_id in range(self.n_subproblems):
            shm = self.worker_shared_memories[block_id]
            metric = np.ndarray(shape=(5,), dtype=np.float64, buffer=shm.buf[:40])
            X_size = int(metric[0])
            X = np.ndarray(shape=(X_size,), dtype=np.float64, buffer=shm.buf[40:])
            solution = Solution(X=np.copy(X), objval=metric[1], is_ray=bool(metric[2]))
            # saving metrics
            if self.to_record:
                self._record_opt_stats(block_id, runtime=metric[3], itercount=metric[4])
            record.update(Proposal.from_solution(solution, dw_iter, block_id))

    def parallel_update_worker_status(
        self, terminate: bool, dw_phase: int, lambs: np.array
    ) -> None:
        """
        update shared_input and then let all processes read the updated information.
        Based on the updated parameters, the processes adjust their models or quit iterating.
        """
        flags = np.ndarray(
            shape=(2, 1), dtype=np.int32, buffer=self.worker_shared_input.buf[:8]
        )
        flags[0] = terminate
        if not terminate:
            flags[1] = dw_phase
            shared_lambs = np.ndarray(
                shape=lambs.shape,
                dtype=np.float64,
                buffer=self.worker_shared_input.buf[8:],
            )
            shared_lambs[:] = lambs
        # all workers now proceeds to the start of an iteration (they will quit if terminate is true)
        self.worker_barrier.wait()
        return

    def parallel_process_clean_up(self) -> None:
        """
        all processes come to an end and we clean up allocated shared memories
        """
        for process in self.worker_processes:
            process.join()
        for shm in self.worker_shared_memories:
            shm.unlink()
        self.worker_shared_input.unlink()
        return

    def parallel_update_solve(
        self, dw_phase: int, dw_iter: int, lambs: np.array, record: DWRecord
    ) -> None:

        self.worker_barrier.wait()  # wait for all workers to finish an iteration
        # check solutions
        for block_id in range(self.n_subproblems):
            shm = self.worker_shared_memories[
                block_id
            ]  # get the shared memory for i-th subproblem to read solution
            metric = np.ndarray(shape=(5,), dtype=np.float64, buffer=shm.buf[:40])
            X_size = int(metric[0])
            X = np.ndarray(shape=(X_size,), dtype=np.float64, buffer=shm.buf[40:])
            solution = Solution(X=np.copy(X), objval=metric[1], is_ray=bool(metric[2]))
            # saving metrics
            if self.verbose:
                print(f"\n----- DW Solve: Subproblem {block_id}\n")
            record.add_subproblem_objval(solution.objval)
            if self.to_record:
                self._record_opt_stats(block_id, runtime=metric[3], itercount=metric[4])
            record.update(Proposal.from_solution(solution, dw_iter, block_id))
        return
