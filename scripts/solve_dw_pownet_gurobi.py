"""
Run experments with Dantzig-Wolfe decomposition for a power system model
The script will collect statistics of each instance and save them in a csv file.
"""

import csv
from datetime import datetime
import os

import gurobipy as gp
import pandas as pd

from pypolp.dw.dw import DantzigWolfe, DWRecord
from pypolp.parser import parse_mps_with_orders, parse_mps, get_dataframe_orders
from pypolp.functions import get_non_binary_from_df, get_temp_dir, count_mps_files


def calculate_set_mipgap(rounding_objval: float, true_objval: float) -> float:
    """The value to set the Gurobi solver is calculated as
    the fractional difference between the objective value from rounding
    and the actual:
    set_mipgap = |rounding_objval - mip_objval| / |mip_objval|
    """
    # If the model is infeasible, then set MIPGap of Gurobi to 1.0
    if rounding_objval is None:
        return 1.0
    else:
        return abs(rounding_objval - true_objval) / abs(true_objval)


def run_dw_experiment(
    model_name: str,
    T_simulate: int,
) -> None:
    # Start time
    timer_script = datetime.now()
    ctime = timer_script.strftime("%Y%m%d_%H%M")
    num_threads = 1  # 1 thread for Gurobi in parallel mode

    true_objvals = pd.read_csv(
        os.path.join(
            get_temp_dir(),
            "true_values",
            f"{model_name}_{T_simulate}.csv",
        ),
        usecols=["mip_objval"],
        header=0,
    )

    print(f"\nDW-PowNet: ==== Begin collecting statistics for {model_name} ====")

    # Define the session name and create a folder to save the outputs
    output_dir = os.path.join(get_temp_dir(), "stats")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    session_name = f"{ctime}_{model_name}_{T_simulate}"

    # Extract row/column orders to parse the DW structure
    instance_folder = os.path.join(
        get_temp_dir(), f"{model_name}_{T_simulate}_instances"
    )
    path_dec = os.path.join(instance_folder, f"{model_name}.dec")
    path_mps = os.path.join(instance_folder, f"{model_name}_0.mps")
    (_, A_df, _, _, col_df) = parse_mps(path_mps)
    row_order, col_order = get_dataframe_orders(path_dec, A_df, col_df)
    del A_df
    del col_df

    # Collect statistics to compare computational performance
    FIELDS = [
        "model_name",
        "T_simulate",
        "lp_gurobi_time",
        "lp_objval",
        "wall_clock_lp_gurobi",
    ]

    # Create a csv file with only headers. We will append to this csv later.
    csv_name = os.path.join(output_dir, f"{session_name}_gurobi.csv")
    with open(csv_name, "w", newline="", encoding="utf-8") as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the fields
        csvwriter.writerow(FIELDS)

    # Count the number of files ending with .mps
    # Days are labeled from k = 0 to k = 364 (max)
    num_instances = count_mps_files(instance_folder)
    for k in range(num_instances):
        print(f"\n\nDW-PowNet: === Solving step {k} ===")
        path_mps = os.path.join(instance_folder, f"{model_name}_{k}.mps")

        # ----- Solve as LP with Gurobi

        timer_lp = datetime.now()

        lp_model = gp.read(path_mps)
        lp_model.relax()
        lp_model.setParam("Threads", num_threads)

        lp_model.optimize()
        lp_objval = lp_model.objval
        lp_gurobi_time = lp_model.runtime

        timer_lp = (datetime.now() - timer_lp).total_seconds()

        # ----- Saving intermediate results
        with open(csv_name, "a", newline="", encoding="utf-8") as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            # writing the data rows
            csvwriter.writerow(
                [
                    model_name,
                    T_simulate,
                    lp_gurobi_time,
                    lp_objval,
                    timer_lp,
                ]
            )

    # Save solutions for future reference. Place them in a folder
    print(f"\n\nDW-PowNet: ==== Completed {session_name} ====")
    print(f'{"Total time to complete:":<20} {datetime.now()- timer_script}')


if __name__ == "__main__":
    model_name = "thailand"
    T_simulate = 24  # The length of each simulation horizon

    run_dw_experiment(
        model_name=model_name,
        T_simulate=T_simulate,
    )
