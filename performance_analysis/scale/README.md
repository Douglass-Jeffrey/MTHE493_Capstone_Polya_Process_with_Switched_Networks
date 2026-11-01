# Toward 10^9 nodes on the GTX 1660 Ti (experiments, 2026-10-31)

Branch `scale-experiments`, on top of `polya-step-experiments` (2026-10-30). Not in the website write-up.
Work was split three ways (see `PLAN.md`): memory-access experiments on resident graphs (A, `quick/`), an
out-of-core graph generator (B, `gen/`), and an out-of-core step engine with CPU feeding and CPU+GPU
concurrency (C, `engine/`). Every timing was taken under a GPU lock (`gpulock.py`) with nothing else running
on the GPU.

## Summary

- **10^9 nodes runs on the 1660 Ti: 3.93 s per Polya step** on a 999,921,184-node, 3.0 x 10^9-edge graph,
  generated in 115 s. 100 steps including generation take about 9 minutes, an 864-step run about an hour.
  The step is bound by the GPU's random urn reads (SMs 99% active), not by the drive.
- The graph comes from an out-of-core generator that makes **exactly the repo's graph** (same seed, same
  schedule, checked against the repo's growth) without storing stubs or edges: 6.2 s at 71.6M nodes (repo:
  7.7 s with last night's changes, 61 s before them).
- The step engine keeps 16-bit urns in VRAM and reads the adjacency over PCIe from pinned RAM, pageable RAM
  (copied into pinned slots by native CPU threads) or the NVMe drive (unbuffered reads by the same threads),
  so the CPU feeds blocks while the GPU computes. Its urns equal the repo's after every step.
- **Several runs per gather** is the biggest win for throughput: 8 runs on one graph take 43 ms per run per step
  at 71.6M (growth order) instead of 269 ms, 31 ms on a breadth-first numbered graph; 16 runs at 35.8M take
  13.5 ms per run. One cache miss serves every run.
- **Finite memory as bits** is now in `Polya_Process` (identical urns): with M = 72 the step at 17.9M goes from
  294 to 64 ms, and 71.6M, which could not allocate its 20.6 GB memory array, runs at full speed.
- **CPU**: the native 8-thread step takes 1.49 s at 71.6M against SciPy's 15.0 s (10x on the Ryzen alone);
  CPU and GPU together (CPU takes ~11% of the rows) cut the 71.6M step by 9%, 0.281 -> 0.256 s.
- What did not help: the L2 fetch-granularity hint (ignored by this GPU), more than 4 threads per row, pushing
  hub contributions with atomics at 10^9 (slower than reading the in-lists).
- Machine facts that shaped this: Windows pins at most 15 GB in total (11 GiB in one allocation); the NVMe drive
  reads at only 1.6 GB/s because it sits in a PCIe 3.0 x2 M.2 slot (the CPU-attached x4 slot would give ~3.4 GB/s).
- Fixed on the way: since c99bdcf, `finalize_growth` freed the edge buffer, which broke mid-run node-growth
  interventions; `finalize_growth(release_edges=False)` keeps it.

## B. Generator (details: gen/results.md)

`src/capstone_optimized/scale/gen.py`. Makes the same CuPy random calls as `grow_batch` (so the samples are
bit-identical), resolves stub indices arithmetically or from the stored targets (VRAM cache + pinned 1 GiB
chunks), and builds the in part in passes sized to VRAM. Checked identical to the repo at 1.1M and 17.9M (seed
493, cached CSRs) and against the repo's growth run in the same process (seeds 7 and 21, including runs that
force every lookup through pinned memory and many fill passes).

| nodes | total | peak RSS | disk |
|---|---|---|---|
| 71.6M | 6.2 s | 2.1 GB | 2.1 GB |
| 143.2M | 7.3 s | 3.9 GB | 4.3 GB |
| 286.3M | 21 s | 5.6 GB | 8.5 GB |
| 572.7M | 79 s | 8.3 GB | 17.1 GB |
| 999.9M | 115 s | 15.0 GB | 29.8 GB |

## C. Engine (details: engine/results.md)

`src/capstone_optimized/scale/engine.py`, native `scale/native/` (`feeder.dll`: unbuffered NVMe reads and
memcpy into pinned slots; `cpu_step.dll`: the CPU draw with prefetching). Placements: VRAM, pinned (zero copy),
pageable RAM, streamed from the drive, or a mix sized from free RAM; optional finite memory (bit ring in RAM or a
file) and push mode for hub rows.

| 71.6M nodes | step |
|---|---|
| repo fused step | 0.290 s |
| engine, adjacency in VRAM | 0.280 s |
| engine, adjacency in pinned RAM | 0.274 s |
| engine, all adjacency from the drive each step | 1.06 s (drive-bound, 1.6 GB/s) |
| engine, VRAM + CPU draws ~11% of the rows | 0.256 s |
| engine, CPU only (8 threads) | 1.49 s (SciPy: 15.0 s) |
| **999.9M nodes, mixed pinned / RAM / drive** | **3.93 s** |

## Data and how to regenerate it
The 10^9-node graph (29.8 GB) was deleted afterwards to give C: its space back. To rebuild it (about 2 minutes):
from the worktree root with `PYTHONPATH=src`,

    python performance_analysis/scale/gpulock.py gen -- python performance_analysis/scale/gen/run_gen.py 1788 ba_1b --validate
    python performance_analysis/scale/gpulock.py run -- python performance_analysis/scale/engine/bench_engine.py --dir ba_1b --placement mixed --block-mb 96 --block-rows 8388608 --auto-budget

It needs ~30 GB free on C: and ~20 GB of free RAM (close other programs). The data folder is in the session's
scratchpad, `...\scratchpad\data`; edit `SCRATCH`/`DATA` in the scripts to put it elsewhere.

## A. Memory access (details: quick/results.md)

| change | effect at 71.6M nodes | status |
|---|---|---|
| L2 fetch granularity 64 -> 32 B (`cudaLimitMaxL2FetchGranularity`) | none: same time, same 29.8 GB of DRAM reads per call; the hint is ignored on this GPU | dropped |
| 16-bit urns (one 32-bit word per node) | kernel -2% (a miss costs the same however few of its bytes are used); half the urn memory | used by the engine |
| Philox random numbers in the kernel | -7 ms per step and no 573 MB random array | used by the engine |
| R runs per gather (urns of R runs side by side) | per-run step 269 -> 43 ms at R = 8 (growth order), 186 -> 31 ms (BFS order); 13.5 ms per run at R = 16 (35.8M) | `scale/replicas.py` |
| finite memory as a ring of bits instead of int32 | 17.9M, M = 72: 294 -> 64 ms per step; 71.6M: from out of memory to 269 ms (same as without memory) | in `Polya_Process` (identical urns) |
