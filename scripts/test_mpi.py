"""
Solves a list of Gurobipy models in parallel using the multiprocessing library.
"""

from pypolp.functions import get_temp_dir
from gurobipy import GRB
import gurobipy as gp
import os
import datetime as dt


class GpModel:
    def __init__(self):
        self.objvals: list = None

    def solve_model(self, filename):
        with gp.Env() as env, gp.Model(env=env) as model:
            model = gp.read(filename)
            model.setParam("OutputFlag", 0)
            model.optimize()
            return model.objVal


if __name__ == "__main__":
    instance_path = os.path.join(get_temp_dir(), "laos_100re_24_instances")
    filenames = [os.path.join(instance_path, f"laos_100re_{i}.mps") for i in range(2)]

    parallel_timer = dt.datetime.now()
    ##### CODE HERE
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Rank 0 sends the filenames to the other ranks
    if rank == 0:
        for i in range(1, len(filenames) + 1):
            req = comm.isend(filenames[i - 1], dest=i, tag=11)
            req.wait()
            objval = None
    if rank > 0:
        req = comm.irecv(source=0, tag=11)
        filename = req.wait()
        model = GpModel()
        objval = model.solve_model(filename)
        # Add MPI barrier
    comm.Barrier()

    # Gather objval by Rank 0
    if rank == 0:
        results = [None] * size
    else:
        results = None
    results = comm.gather(objval, root=0)

    MPI.Finalize()
    ################
    parallel_timer = dt.datetime.now() - parallel_timer

    if rank == 0:
        # Print the results
        print("\n===== Results =====")
        for i, result in enumerate(results):
            print(f"Model {i}: {result}")
