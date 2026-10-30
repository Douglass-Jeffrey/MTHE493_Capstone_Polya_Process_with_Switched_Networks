Nsight reports for the GTX 1660 Ti Polya step
=============================================

Taken on an idle machine with a Ryzen 3 3100, 32 GB RAM and a GTX 1660 Ti (6 GB), on Windows 10.
Nsight Systems 2026.1.2 (.nsys-rep) and Nsight Compute 2025.3.1 (.ncu-rep). Open them with these versions or newer.
Each run grows an m=3 Barabasi-Albert network, builds the CSR matrix, then runs a few Polya steps.
The .json beside each report holds that run's settings and step times.
The "admin" runs were run as administrator, so they also record PCIe traffic and VRAM vs system RAM usage.

original/  - code before this branch (cuSPARSE A @ U, upcast to float64 every step)
  n18M.nsys-rep       17.9M nodes, 5 steps. Last size that fits in VRAM.
  n36M.nsys-rep       35.8M nodes, 5 steps. Spills into system RAM.
  n72M.nsys-rep       71.6M nodes, 3 steps. Spills badly, about CPU speed.
  a36.nsys-rep        35.8M nodes, 5 steps, admin.
  a72.nsys-rep        71.6M nodes, 3 steps, admin.
  c36.ncu-rep         35.8M nodes, the cuSPARSE kernel in step 2: time, PCIe bytes, DRAM bytes.
  c72.ncu-rep         71.6M nodes, same as c36.
  ncu_calib.ncu-rep   Calibration: sequential vs random reads from VRAM and from system RAM.

float32_fix/  - original code with two patches applied at run time (not in the repo)
  1. Cast U to float32 before the product, so no float64 copies are made.
  2. Free the edge and stub lists after the CSR build.
  n36M_fix.nsys-rep   35.8M nodes, 5 steps.
  n72M_fix.nsys-rep   71.6M nodes, 3 steps.
  a36f.nsys-rep       35.8M nodes, 5 steps, admin.
  a72f.nsys-rep       71.6M nodes, 4 steps, admin.
  c36f.ncu-rep        35.8M nodes, the cuSPARSE kernel in step 3.
  c72f.ncu-rep        71.6M nodes, same as c36f.

fused_kernel/  - this branch (src/capstone_optimized/core/csr_kernels.py)
  One custom CUDA kernel replaces the sparse product and the draw. It reads the CSR indices and urns
  directly as int32 and sums them as int64, so it makes no float64 copies and gives bit-identical results.
  The CSR keeps no values array, since every weight is 1. The growth buffers are freed after the CSR build.
  If VRAM is short, the CSR indices go to pinned system RAM, which is fine because they are read sequentially.
  e2e72_new.nsys-rep  71.6M nodes, 4 steps, admin, full pipeline. Compare with original/a72.
  res72.nsys-rep      71.6M nodes, 3 steps, step loop only on a pre-built CSR.
  ncu_res72.ncu-rep   71.6M nodes, the fused kernel: time, DRAM throughput, L1/L2 hit rates.

Notes
  The .sqlite exports are not kept. To regenerate one, run: nsys export -t sqlite <report>.nsys-rep
  The .ncu-rep files hold only the listed metrics, so the GUI's Memory Workload Analysis chart is empty for them.
