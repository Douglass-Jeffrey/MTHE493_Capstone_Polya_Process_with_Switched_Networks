# Polya process simulator: optimization branch (`scale-experiments`)

Experimental branch for making the networked Polya simulation faster and larger on a GTX 1660 Ti (6 GB).
It builds on `optimization_extensions` (fused step kernel) and `polya-step-experiments` (GPU CSR build,
growth without spilling, GPU renumbering), and reaches 10^9 nodes at about 4 s per step. Thesis results,
plots and the website's profiling reports are on the other branches; here they are ignored by git.

## Running a simulation

    python run_sim.py --batches 128                      # 71.6M nodes, 100 steps, every optimization
    python run_sim.py --batches 128 --preset original    # the code as it was before the optimizations
    python run_sim.py --help                             # every switch, backend and preset

`run_sim.py` grows the benchmark's Barabasi-Albert graph (n = 64 + 559,240 x batches), runs the steps, prints
the time of each phase and appends a summary, with an md5 of the final urns, to `work/runs.jsonl`. Each
optimization can be switched on or off (`--kernel`, `--memory-format`, `--csr-build`, `--csr-placement`,
`--renumber`, `--reserve`, `--free-cache`, `--keep-edges`), and there are three backends: `core` (the
simulator itself), `replicas` (several runs side by side on one graph) and `engine` (graphs larger than VRAM).
Switches that should not change the result do not: the presets give identical urns for the same seed.

## Layout

| path | what |
|---|---|
| `src/capstone_optimized/core/` | the simulator: graph, BA and switch-network growth, Polya process, interventions, CUDA kernels (`csr_kernels.py`) |
| `src/capstone_optimized/scale/` | large-graph tools: out-of-core generator (`gen.py`), engine (`engine.py`, native feeder and CPU step), replicas, Philox |
| `src/capstone_optimized/experiments/` | the thesis experiment scripts, plus tests for the GPU CSR build and renumbering |
| `performance_analysis/profiling/` | how the original code's slowdown was profiled, and the runs behind the website's Nsight screenshots |
| `performance_analysis/fused_kernel/` | the fused kernel's prototype, the node-ordering and streaming benchmarks, bit-exactness checks |
| `performance_analysis/experiments/` | growth, CSR build and renumbering results |
| `performance_analysis/scale/` | the 10^9-node work: results (`README.md`), benchmarks, verification scripts, Nsight reports |
| `work/` (not in git) | generated graphs, CSR caches, run logs; set `POLYA_WORK` to put it elsewhere |

Run scripts from the repository root with `PYTHONPATH=src` (run_sim.py sets it itself).
