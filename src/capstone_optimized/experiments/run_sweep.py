"""
run_sweep.py — Sweep runner for test_i.py.

Loads a CSV of parameters, computes the cartesian product of all multi-valued
columns, and calls test_i.main(cfg) once per combination.

Usage
-----
    python run_sweep.py --config params.csv

CSV format
----------
Column headers must match SimulationConfig field names exactly.
Use '|' inside a cell to provide multiple values:

    intervention_class,num_injections,num_connections
    Switch_Network_Injection_Intervention,16384|32768|65536,,
    Switch_Network_Node_Growth_Intervention,,4|8,

The above produces 5 runs total (3 injection + 2 node-growth).
"""

import argparse
import time

import test_i
from test_i import SimulationConfig, configs_from_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep runner for test_i.py")
    parser.add_argument(
        "--config",
        metavar="CSV",
        required=True,
        help="Path to the CSV parameter file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = configs_from_csv(args.config)

    print(f"\nStarting sweep: {len(configs)} run(s) total.\n")
    t_sweep_start = time.time()

    for run_idx, cfg in enumerate(configs):
        # Assign a unique run label so each run's output file is distinct
        cfg.output_dir = cfg.output_dir  # already set from CSV; no change needed
        cfg_with_label = cfg  # label is derived inside run_simulation via run_label()

        print(f"\n[Sweep {run_idx+1}/{len(configs)}]")
        test_i.main(cfg, run_idx=run_idx)

    print(f"\nSweep complete. Total time: {time.time()-t_sweep_start:.3f}s")


if __name__ == "__main__":
    main()
