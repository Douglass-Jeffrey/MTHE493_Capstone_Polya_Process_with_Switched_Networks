# Overnight plan, 2026-10-31: toward 10^9 nodes on the GTX 1660 Ti

Branch `scale-experiments` (worktree `C:\Users\dougl\polya-overnight`), on top of `polya-step-experiments`.
The user's own working tree (`OneDrive\...\MTHE493_...`) is not touched. Nothing is committed by workers; the
coordinator commits at the end.

## Machine
GTX 1660 Ti, 6 GB (5.0 GB free to CUDA), 1.5 MB L2, 24 SMs, PCIe 3.0 x16 (~13 GB/s copies, ~11-12 GB/s
zero-copy reads). Ryzen 3 3100 (4c/8t), 32 GB RAM (~21 GB free with the user's apps open), C: = 500 GB NVMe
(WD SN750, ~3.4 GB/s read, **51 GB free**), D: = 1 TB HDD. Python 3.12, CuPy 13.6 (CUDA 13), NumPy 2.3, no Numba.
MSVC 2022 (`cl.exe`) and nvcc 13.0 are installed, so native C++/CUDA DLLs (ctypes) are possible.
Nsight Systems 2026.5.1 at `C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.5.1` (shell is elevated,
so GPU metrics work). Nsight Compute is NOT installed.

## Rules for every worker
- Run Python with `PYTHONDONTWRITEBYTECODE=1` and `PYTHONPATH=src` from the worktree root.
- **Every run that touches the GPU, or uses more than ~2 GB of host RAM, goes through the lock**:
  `python performance_analysis/scale/gpulock.py <tag> -- python <script> ...`
  Hold it only for the run itself. Timings are only valid under the lock.
- Data (graphs) go in `<scratchpad>/data/<name>/`, scratchpad =
  `C:\Users\dougl\AppData\Local\Temp\claude\c--Users-dougl-OneDrive-Documents-dev-projects-MTHE493-Capstone-Polya-Process-with-Switched-Networks\7bf0ea5c-bb18-4a6c-8e60-f701292fffbe\scratchpad`.
  C: has 51 GB free: keep the data folder under ~38 GB total, check free space before writing big files, and
  delete your intermediate datasets when done. Only the generator workstream writes the 10^9-node graph
  (`data/ba_1b/`, ~32 GB).
- Do not modify files owned by another workstream. Do not `git commit`. Do not touch the user's repo folder.
- Record every measurement (command, size, result) in your workstream's `results.md`.
- Never leave background processes running when you finish.

## Shared pieces (coordinator, done)
- `src/capstone_optimized/scale/philox.py`: `PHILOX_SRC` (Philox4x32-10, checked against Random123's known
  answers) with `philox_uniform(seed, step, node, stream)` -> double in [0,1); `uniform_ref` NumPy reference.
- `src/capstone_optimized/scale/split_format.py`: the on-disk graph format (out part with fixed stride m, in part
  as CSR), `from_csr` converter, `open_arrays` (memmaps), `validate`. Test graphs already converted:
  `data/n1118544_split`, `data/n17895744_split` (from the repo's growth, seed 493, batches 32).
- Cached repo CSRs (indptr.npy, indices.npy, meta.json) from earlier nights: `<scratchpad>/opt/cache/`
  n1118544, n17895744, n35791424, n71582784 (growth order, seed 493, batch schedule [8,128,2048,32768,524288]
  x batches, initial clique 64, m 3), plus _bfs / _rcm renumbered variants.
- Draw semantics (must match `csr_kernels.draw_two_colour`): super urn = own urn + neighbours' urns;
  `p = red / (red + black)` in float64 (sum 0 -> 1); draw = 1 (black) if `rand > p` else 0 (red);
  then `urns[v, draw] += delta`.
- Benchmark schedule: initial 64, sizes [8,128,2048,32768,524288], `batches` of each.
  n = 64 + batches * 559240. batches=128 -> 71.6M, 256 -> 143.2M, 512 -> 286.3M, 1024 -> 572.7M,
  1788 -> 999,921,184 (the "1B" target).

## Workstreams
- **A (coordinator)**: memory-access experiments on resident graphs at 71.6M: L2 fetch granularity 32 vs 64 B,
  16-bit urns, in-kernel Philox vs a random array, bit-packed finite-memory ring, several runs per graph (R
  replicas sharing one gather). Files: `performance_analysis/scale/quick/`, `src/capstone_optimized/scale/quick_kernels.py`.
- **B (generator fork)**: out-of-core batched BA generator writing the split format directly, bit-identical to
  the repo's growth for the same seed and schedule, scaling to 1788 batches (~10^9 nodes). Files:
  `src/capstone_optimized/scale/gen.py`, `performance_analysis/scale/gen/`.
- **C (engine fork)**: out-of-core Polya step on the split format with 16-bit urns in VRAM, Philox, bit-packed
  draws, and the adjacency read over PCIe from pinned RAM and/or streamed from NVMe by native CPU threads into
  pinned buffers; then CPU+GPU concurrency. Files: `src/capstone_optimized/scale/engine.py`,
  `src/capstone_optimized/scale/native/`, `performance_analysis/scale/engine/`.
