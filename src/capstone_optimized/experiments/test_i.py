"""
test_i.py — Polya / Barabasi-Albert simulation runner with intervention sweep.

Usage
-----
Default hard-coded config:
    python test_i.py

Run all cartesian-product combinations from a CSV:
    python test_i.py --config params.csv

CSV format (column headers must match SimulationConfig field names exactly):
    intervention_class,num_injections,num_connections,...
    Switch_Network_Injection_Intervention,16384|32768,,
    Switch_Network_Node_Growth_Intervention,,4|8,

Use '|' inside a cell to provide multiple values for that column.
The cartesian product across ALL columns is computed per row,
then all rows are pooled — giving one run per combination.

Growth phase lists use ';' as a separator inside a single cell:
    num_batches  →  "256;256;256"
    batch_sizes  →  "8;256;8192"
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ---------------------------------------------------------------------------
# GPU / CuPy bootstrap
# ---------------------------------------------------------------------------

try:
    from capstone_optimized import cupy_fallback as cf

    cp = getattr(cf, "cp", None)
    CUPY_AVAILABLE = getattr(cf, "CUPY_AVAILABLE", False)
    print(
        "cupy_fallback: cp module =",
        getattr(cp, "__name__", None),
        "CUPY_AVAILABLE =",
        CUPY_AVAILABLE,
    )
except Exception as _e:
    print("Could not import capstone_optimized.cupy_fallback:", _e)
    cp = None
    CUPY_AVAILABLE = False

if not CUPY_AVAILABLE:
    print(
        "\nGPU requested but CuPy/CUDA not available in this Python environment.\n"
        "Activate your virtualenv where CuPy is installed and re-run:\n"
        "  . .venv\\Scripts\\Activate\n"
        "  $env:PYTHONPATH = 'src'\n"
        "  python src\\capstone_optimized\\experiments\\test_i.py\n"
    )

from capstone_optimized.core import (
    Barabasi_Albert_Growth,
    Graph,
    Polya_Process,
    Switch_Network_Injection_Intervention,
    Switch_Network_Node_Growth_Intervention,
)

# Maps string names (used in CSV) to the actual classes
INTERVENTION_CLASS_MAP = {
    "Switch_Network_Injection_Intervention": Switch_Network_Injection_Intervention,
    "Switch_Network_Node_Growth_Intervention": Switch_Network_Node_Growth_Intervention,
}

# Valid method names per class — used for validation and dispatch
INTERVENTION_METHOD_MAP = {
    Switch_Network_Injection_Intervention: {
        "degree_centrality_optimized_intervention_step",
        "eigenvector_centrality_optimized_intervention_step",
    },
    Switch_Network_Node_Growth_Intervention: {
        "degree_centrality_optimized_intervention_step",
        "random_intervention_step",
    },
}


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """All tuneable parameters for one simulation run."""

    # --- Graph / growth ---
    num_colors: int = 2
    initial_nodes: int = 64
    barabasi_num_connections: int = 3
    # Semicolon-separated lists for growth phases
    num_batches: str = "256;256;256"
    batch_sizes: str = "8;256;8192"
    init_urn_red: int = 1
    init_urn_black: int = 1

    # --- Intervention ---
    # String names so they survive CSV round-trips; resolved to class/method at runtime
    intervention_class: str = "Switch_Network_Injection_Intervention"
    intervention_method: str = "degree_centrality_optimized_intervention_step"
    num_interventions: int = 1
    intervention_steps: int = 144
    once_per_node_iv: bool = False
    # Switch_Network_Injection_Intervention params
    num_injections: float = 131072 / 4
    intervener_urn_red: int = 1000
    intervener_urn_black: int = 0
    # Switch_Network_Node_Growth_Intervention params
    num_connections: int = 4

    # --- Polya process ---
    polya_steps: int = 720 + 144
    memory_enabled: bool = True
    memory_decay_time: int = 72
    delta_gain: int = 1
    mem_decay_loss: int = 1

    # --- Output ---
    num_bins: int = 50
    output_dir: str = "."

    # ------------------------------------------------------------------ helpers

    def num_batches_list(self) -> list[int]:
        return [int(x) for x in str(self.num_batches).split(";")]

    def batch_sizes_list(self) -> list[int]:
        return [int(x) for x in str(self.batch_sizes).split(";")]

    def resolved_intervention_class(self):
        cls = INTERVENTION_CLASS_MAP.get(self.intervention_class)
        if cls is None:
            raise ValueError(
                f"Unknown intervention_class '{self.intervention_class}'. "
                f"Valid options: {list(INTERVENTION_CLASS_MAP.keys())}"
            )
        return cls

    def resolved_intervention_method(self):
        cls = self.resolved_intervention_class()
        valid_methods = INTERVENTION_METHOD_MAP[cls]
        if self.intervention_method not in valid_methods:
            raise ValueError(
                f"Unknown intervention_method '{self.intervention_method}' "
                f"for class '{self.intervention_class}'. "
                f"Valid options: {sorted(valid_methods)}"
            )
        return getattr(cls, self.intervention_method)

    def run_label(self, run_index: int) -> str:
        return f"run_{run_index:04d}"


# ---------------------------------------------------------------------------
# CSV → list[SimulationConfig]
# ---------------------------------------------------------------------------

def _coerce(value: str, target_type: type) -> Any:
    """Cast a string to the field's declared type."""
    if target_type == bool:
        return value.strip().lower() in ("1", "true", "yes")
    return target_type(value.strip())


def configs_from_csv(csv_path: str) -> list[SimulationConfig]:
    """
    Read a CSV and return one SimulationConfig per cartesian-product combination.

    Each column corresponds to a SimulationConfig field name.
    Use '|' inside a cell for multiple values; the cartesian product of all
    multi-valued columns in a row is computed, then all rows are pooled.

    Empty cells are skipped (the dataclass default is kept for that field).
    """
    field_map = {f.name: f for f in fields(SimulationConfig)}

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        raw_rows = list(reader)

    if not raw_rows:
        raise ValueError(f"CSV file '{csv_path}' is empty.")

    unknown = [k.strip() for k in raw_rows[0] if k.strip() not in field_map]
    if unknown:
        raise ValueError(f"CSV contains unknown parameter column(s): {unknown}")

    all_configs: list[SimulationConfig] = []

    for row in raw_rows:
        per_field_values: dict[str, list[Any]] = {}
        for col_name, raw_cell in row.items():
            col_name = col_name.strip()
            if col_name not in field_map or not raw_cell.strip():
                continue  # skip unknown cols and empty cells
            target_type = field_map[col_name].type
            if isinstance(target_type, str):
                target_type = eval(target_type)  # safe: only our own dataclass types
            alternatives = [v.strip() for v in raw_cell.split("|") if v.strip()]
            per_field_values[col_name] = [_coerce(v, target_type) for v in alternatives]

        keys = list(per_field_values.keys())
        for combo in itertools.product(*[per_field_values[k] for k in keys]):
            cfg = SimulationConfig()
            for k, v in zip(keys, combo):
                setattr(cfg, k, v)
            all_configs.append(cfg)

    print(
        f"Loaded {len(raw_rows)} CSV row(s) → "
        f"{len(all_configs)} cartesian-product combination(s)."
    )
    return all_configs


# ---------------------------------------------------------------------------
# Graph / urn helpers
# ---------------------------------------------------------------------------

def build_initial_urns(num_nodes: int, cfg: SimulationConfig):
    urns = cp.zeros((num_nodes, cfg.num_colors), dtype=cp.int32)
    urns[:, 0] = cfg.init_urn_red
    urns[:, 1] = cfg.init_urn_black
    return urns


def make_urn_callback(cfg: SimulationConfig):
    """Return a grow_batch-compatible callback that initialises urns for new nodes."""
    def _callback(b_a, batch_size: int) -> None:
        new_urns = cp.zeros((batch_size, b_a.graph.num_colors), dtype=cp.int32)
        new_urns[:, 0] = cfg.init_urn_red
        new_urns[:, 1] = cfg.init_urn_black
        b_a.graph.node_urns = cp.vstack([b_a.graph.node_urns, new_urns])
        b_a.graph.num_nodes += batch_size
    return _callback


# ---------------------------------------------------------------------------
# Growth
# ---------------------------------------------------------------------------

def _is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        isinstance(exc, MemoryError)
        or "out of memory" in msg
        or "no memory" in msg
        or ("cuda" in msg and "memory" in msg)
    )


def _gpu_mem_str(graph: Graph) -> str:
    if not (graph.use_gpu and CUPY_AVAILABLE and cp is not None):
        return ""
    try:
        free, total = cp.cuda.runtime.memGetInfo()
        return f"  free={free/1024**2:.1f}MB/{total/1024**2:.1f}MB"
    except Exception:
        return ""


def run_growth_phase(
    graph: Graph,
    growth: Barabasi_Albert_Growth,
    num_batches: int,
    batch_size: int,
    callback=None,
) -> None:
    """Run one growth phase with automatic OOM back-off."""
    for b in range(num_batches):
        current_bs = batch_size
        attempt = 0
        while True:
            attempt += 1
            t0 = time.time()
            try:
                growth.grow_batch(batch_size=current_bs, urn_addition_callback=callback)
            except Exception as exc:
                if _is_oom(exc):
                    if current_bs <= 1:
                        raise RuntimeError("Batch size 1 still causes OOM.") from exc
                    current_bs = max(1, current_bs // 2)
                    print(
                        f"  OOM on batch {b+1} — reducing to {current_bs} "
                        f"(attempt {attempt})"
                    )
                    continue
                raise
            else:
                if b % 100 == 0:
                    mem = _gpu_mem_str(graph)
                    print(
                        f"  Batch {b}/{num_batches}  size={current_bs}"
                        f"  time={time.time()-t0:.3f}s  edges={graph.num_edges}{mem}"
                    )
                break


# ---------------------------------------------------------------------------
# Intervention dispatch
# ---------------------------------------------------------------------------

def fire_intervention(
    intervention,
    intervention_class,
    cfg: SimulationConfig,
    polya: Polya_Process,
) -> None:
    """
    Call the configured intervention method with the correct arguments.
    Dispatches on intervention_class to determine which kwargs to pass.
    """
    intervener_urn = cp.array(
        [cfg.intervener_urn_red, cfg.intervener_urn_black], dtype=cp.int32
    )
    method = getattr(intervention, cfg.intervention_method)

    if intervention_class is Switch_Network_Injection_Intervention:
        method(
            num_injections=cfg.num_injections,
            intervener_urn=intervener_urn,
        )

    elif intervention_class is Switch_Network_Node_Growth_Intervention:
        method(
            num_connections=cfg.num_connections,
            intervener_urn=intervener_urn,
            mid_polya_intervention=True,
            polya_process=polya,
            num_interventions=cfg.num_interventions,
        )

    else:
        raise ValueError(f"Unhandled intervention class: {intervention_class}")


# ---------------------------------------------------------------------------
# Polya process
# ---------------------------------------------------------------------------

def run_polya(
    graph: Graph,
    intervention,
    intervention_class,
    cfg: SimulationConfig,
) -> np.ndarray:
    """
    Run the Polya process, collect per-step histogram data, and fire the
    intervention at the configured step.

    Returns
    -------
    hist_data : np.ndarray, shape (polya_steps, num_bins)
    """
    deltas = cp.full((graph.num_nodes, graph.num_colors), cfg.delta_gain, dtype=cp.int32)
    mem_decay = cp.full(
        (graph.num_nodes, graph.num_colors), cfg.mem_decay_loss, dtype=cp.int32
    )
    polya = Polya_Process(
        graph,
        memory_enabled=cfg.memory_enabled,
        memory_decay_time=cfg.memory_decay_time,
        delta=deltas,
        mem_decay=mem_decay,
    )

    n_steps = int(os.environ.get("CAPSTONE_POLYA_STEPS", str(cfg.polya_steps)))
    hist_bins = cp.linspace(0, 1, cfg.num_bins + 1)
    hist_data = np.zeros((n_steps, cfg.num_bins), dtype=np.float64)

    t_start = time.time()
    for step in range(n_steps):
        t0 = time.time()
        polya.step()

        # Collect histogram
        probs = graph.node_urns[:, 0] / cp.sum(graph.node_urns, axis=1)
        hist_i, _ = cp.histogram(probs, hist_bins)
        hist_data[step] = cp.asnumpy(hist_i)

        # Fire intervention at the configured step
        if step == cfg.intervention_steps:
            fire_intervention(intervention, intervention_class, cfg, polya)
            print(
                f"  [step {step}] Intervention fired "
                f"(class={cfg.intervention_class}  "
                f"total={intervention.current_num_interventions})"
            )

        if step % 100 == 0:
            print(f"  Polya step {step}/{n_steps}  time={time.time()-t0:.3f}s")

    print(f"  Polya complete. Total time: {time.time()-t_start:.3f}s")
    return hist_data


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_simulation(cfg: SimulationConfig, run_label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Run: {run_label}")
    print(f"{'='*60}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intervention_class = cfg.resolved_intervention_class()

    # --- Build graph ---
    initial_urns = build_initial_urns(cfg.initial_nodes, cfg)
    graph = Graph(
        num_nodes=cfg.initial_nodes,
        num_colors=cfg.num_colors,
        use_gpu=True,
        node_urns=initial_urns,
    )
    growth = Barabasi_Albert_Growth(
        graph, m=cfg.barabasi_num_connections, initial_nodes=cfg.initial_nodes
    )

    print(
        f"\nGraph init: initial_nodes={cfg.initial_nodes}  "
        f"num_batches={cfg.num_batches}  "
        f"batch_sizes={cfg.batch_sizes}  "
        f"barabasi_m={cfg.barabasi_num_connections}"
    )

    # --- Growth phases ---
    t_growth = time.time()
    urn_callback = make_urn_callback(cfg)
    for nb, bs in zip(cfg.num_batches_list(), cfg.batch_sizes_list()):
        print(f"\n  Growth phase: {nb} batches × size {bs}")
        run_growth_phase(graph, growth, nb, bs, callback=urn_callback)
    growth.finalize_growth()
    print(f"\nAll growth phases done.  Total: {time.time()-t_growth:.3f}s")

    # --- Intervention ---
    print(
        f"\nIntervention init: class={cfg.intervention_class}  "
        f"num_interventions={cfg.num_interventions}  "
        f"num_injections={cfg.num_injections}  "
        f"num_connections={cfg.num_connections}  "
        f"intervener_urn=[{cfg.intervener_urn_red},{cfg.intervener_urn_black}]  "
        f"once_per_node_iv={cfg.once_per_node_iv}  "
        f"intervention_steps={cfg.intervention_steps}"
    )
    intervention = intervention_class(graph=graph, once_per_node_iv=cfg.once_per_node_iv)

    # --- Polya ---
    print(
        f"\nPolya init: polya_steps={cfg.polya_steps}  "
        f"memory_enabled={cfg.memory_enabled}  "
        f"memory_decay_time={cfg.memory_decay_time}  "
        f"delta_gain={cfg.delta_gain}  "
        f"mem_decay_loss={cfg.mem_decay_loss}"
    )
    hist_data = run_polya(graph, intervention, intervention_class, cfg)

    # --- Save output ---
    csv_path = out_dir / f"hist_data_{run_label}.csv"
    np.savetxt(csv_path, hist_data, delimiter=",", fmt="%d")
    print(f"\n  Saved → {csv_path}")

    # Save matching config so parameters can always be traced back to a run
    cfg_path = out_dir / f"cfg_{run_label}.json"
    with open(cfg_path, "w") as fh:
        json.dump(asdict(cfg), fh, indent=2)
    print(f"  Saved → {cfg_path}")

    print(
        f"  Run '{run_label}' complete.  "
        f"nodes={graph.num_nodes}  edges={graph.num_edges}"
    )

    # --- Explicit cleanup to free GPU memory before the next run ---
    # CuPy arrays are not released until both Python's GC and the CuPy memory
    # pool are flushed, so we do both explicitly here.
    del graph, growth, intervention, hist_data
    gc.collect()
    if CUPY_AVAILABLE and cp is not None:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            free, total = cp.cuda.runtime.memGetInfo()
            print(
                f"  GPU memory after cleanup: "
                f"{free/1024**2:.1f}MB free / {total/1024**2:.1f}MB total"
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polya / BA simulation runner")
    parser.add_argument(
        "--config",
        metavar="CSV",
        default=None,
        help=(
            "Path to a CSV of parameters. Column headers must match "
            "SimulationConfig field names. Use '|' in a cell for multiple "
            "values; the cartesian product is run across all columns."
        ),
    )
    return parser.parse_args()


def main(cfg: SimulationConfig | None = None, run_idx: int = 0) -> None:
    # If called directly (e.g. from run_sweep.py), use the provided config.
    # When run as a standalone script, fall back to CLI / default config.
    if cfg is not None:
        run_simulation(cfg, cfg.run_label(run_idx))
        return

    args = parse_args()
    configs = configs_from_csv(args.config) if args.config else [SimulationConfig()]

    for run_idx, cfg in enumerate(configs):
        run_simulation(cfg, cfg.run_label(run_idx))

    print("\nAll runs complete.")


if __name__ == "__main__":
    main()
