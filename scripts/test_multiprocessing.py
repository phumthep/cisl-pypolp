"""
Solves a list of Gurobipy models in parallel using the multiprocessing library.
"""

from pypolp.functions import get_temp_dir
from gurobipy import GRB
import gurobipy as gp
import os
import datetime as dt
import multiprocessing as mp


class GpMODEL:
    def __init__(self, file):
        self.objvals: list = None
        self.filenames: list = None

    def solve_model(self, filename):
        with gp.Env() as env, gp.Model(env=env) as model:
            model = gp.read(filename)
            model.setParam("Threads", 1)
            model.optimize()
            return model.objVal


if __name__ == "__main__":
    instance_path = os.path.join(get_temp_dir(), "laos_100re_24_instances")
    filenames = [os.path.join(instance_path, f"laos_100re_{i}.mps") for i in range(2)]

    # Solve the models in parallel
    dw_model = GpMODEL(filenames)

    parallel_timer = dt.datetime.now()
    with mp.Pool() as pool:
        results = pool.map(dw_model.solve_model, filenames)
    parallel_timer = dt.datetime.now() - parallel_timer

    # Solve the models in serial
    serial_timer = dt.datetime.now()
    for filename in filenames:
        dw_model.solve_model(filename)
    serial_timer = dt.datetime.now() - serial_timer

    print("===== Serial vs Parallel Comparison =====")
    print(f"\nSerial solve time: {round(serial_timer.total_seconds(), 2)}")
    print(f"Parallel solve time: {round(parallel_timer.total_seconds(), 2)}")

    # Print the results
    print("\n===== Results =====")
    for i, result in enumerate(results):
        print(f"Model {i}: {result}")
