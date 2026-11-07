# Polya process simulator: optimization branch (`scale-experiments`)

Experimental branch for making the networked Polya simulation faster and larger on a GTX 1660 Ti (6 GB).
It builds on `optimization_extensions` (fused step kernel) and `polya-step-experiments` (GPU CSR build,
growth without spilling, GPU renumbering), and reaches 10^9 nodes at about 4 s per step. Thesis results,
plots and the website's profiling reports are on the other branches; here they are ignored by git.

## Running a simulation

    python run_sim.py --batches 128                      # 71.6M nodes, 100 steps, every optimization
    python run_sim.py --batches 128 --preset original    # the code as it was before the optimizations
    python run_sim.py --help                             # every switch, backend and preset

    # 10^9 nodes (needs ~40 GB free on the drive and ~20 GB of free RAM; size --ram-gb to what is free):
    python run_sim.py --backend engine --batches 1788 --placement mixed --block-mb 96 --block-rows 8388608 \
        --pinned-gb 13 --ram-gb 5 --memory 72 --ring-file work/data/ring_1b_m72.bin --steps 864 \
        --hist work/reports/1b/hist864.csv --progress 24

`run_sim.py` grows the benchmark's Barabasi-Albert graph (n = 64 + 559,240 x batches), runs the steps, prints
the time of each phase and appends a summary, with an md5 of the final urns, to `work/runs.jsonl`. Each
optimization can be switched on or off (`--kernel`, `--memory-format`, `--csr-build`, `--csr-placement`,
`--renumber`, `--reserve`, `--free-cache`, `--keep-edges`), and there are three backends: `core` (the
simulator itself), `replicas` (several runs side by side on one graph) and `engine` (graphs larger than VRAM).
Switches that should not change the result do not: the presets give identical urns for the same seed.
`--hist` writes the red-fraction histogram after every step as it goes, and `--progress N` prints the step time
every N steps, for long runs.

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
