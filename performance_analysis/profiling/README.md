# Profiling harnesses

How the Nsight numbers on the website were produced. The reports themselves (`.nsys-rep`, `.ncu-rep`, 2-3 GB with
their `.sqlite` exports) are not in this branch: the ones used on the website are in
`performance_analysis/nsight/` on `optimization_extensions`. Rerunning a suite writes new reports to
`work/reports/` (see `src/capstone_optimized/scale/paths.py`). The suites call `python` from the active environment
and Nsight from its default install folder; run them elevated for the GPU-metrics and WDDM traces.

## spill/: why the 1660 Ti fell to CPU speed

| file | what |
|---|---|
| `profile_polya.py` | the original code (`work/orig_src`) on one graph size, with optional run-time patches (`--fix`: float32 U, freed growth buffers) and memory/bandwidth probes |
| `run.ps1`, `suite.ps1`, `admin_suite.ps1` | Nsight Systems / Compute runs of it: the original and float32-fixed code at 17.9M to 71.6M nodes, the PCIe calibration (`pcie_calib.py`), the cuSPARSE kernel |
| `cpu_bench.py`, `gather_bench.py` | the CPU (NumPy/SciPy) step, and random vs sequential gathers from VRAM and system RAM |
| `analyze.py`, `analyze_admin.py`, `figures.py` | summaries of the `.sqlite` exports and the website figures (`fig/`) |
| `*.json`, `*.log`, `admin/` | the results of those runs |

## website/: the reports behind the website screenshots

| file | what |
|---|---|
| `profile_step.py` | growth, CSR build and steps with the original, current or night-2 code, with VRAM/RAM probes |
| `suite.sh`, `suite2.sh`, `suite3.sh` | the Nsight runs: original vs fused kernel at 8.9M to 143.2M nodes, the step loop resident / zero-copy / streamed, growth vs BFS order |
| `analyze2.py`, `cap.ps1` | report summaries; a screen capture helper for the GUI screenshots |
| `results/`, `smoke_*.json`, `suite*.log` | the results of those runs |
