# Toward 10^9 nodes on the GTX 1660 Ti (experiments, 2026-10-31)

Branch scale-experiments, on top of polya-step-experiments (2026-10-30). The 10^9-node run is in the website
write-up ("Scaling to one billion nodes"); the other experiments are not. Work was split three ways (see PLAN.md):
memory-access experiments on resident graphs (A, quick/), an out-of-core graph generator (B, gen/), and an
out-of-core step engine with CPU feeding and CPU+GPU concurrency (C, engine/). Every timing was taken under a GPU
lock (gpulock.py) with nothing else running on the GPU.

## Summary

10^9 nodes runs on the 1660 Ti at 3.93 s per Polya step, on a 999,921,184-node, 3.0 x 10^9-edge graph generated
in 115 s (71 s on 2026-11-01). A full 864-step run with finite memory (M = 72) took 60 min of steps (median 4.18 s,
the same before and after step 72 since the WDDM fix of 2026-11-02) and 67 min in all, with the GPU clock capped at
1,200 MHz (about 4.0 s per step expected at the usual 1,960 MHz; see engine/results.md, engine/run864/ and the
Nsight traces in engine/nsys/). The website now describes it. The first 864-step run (2026-11-01) went off once at
step 32 and is superseded. The step is bound by the GPU's random urn reads (SMs 96-99% active), not by the drive.

The graph comes from an out-of-core generator that makes exactly the repo's graph (same seed, same schedule,
checked against the repo's growth) without storing stubs or edges: 6.2 s at 71.6M nodes (repo: 7.7 s with last
night's changes, 61 s before them).

The step engine keeps 16-bit urns in VRAM and reads the adjacency over PCIe from pinned RAM, pageable RAM (copied
into pinned slots by native CPU threads) or the NVMe drive (unbuffered reads by the same threads), so the CPU feeds
blocks while the GPU computes. Its urns equal the repo's after every step.

Several runs per gather is the biggest win for throughput: 8 runs on one graph take 43 ms per run per step at 71.6M
(growth order) instead of 269 ms, 31 ms on a breadth-first numbered graph; 16 runs at 35.8M take 13.5 ms per run.
One cache miss serves every run.

Finite memory as bits is now in Polya_Process (identical urns): with M = 72 the step at 17.9M goes from 294 to
64 ms, and 71.6M, which could not allocate its 20.6 GB memory array, runs at full speed.

On the CPU, the native 8-thread step takes 1.49 s at 71.6M against SciPy's 15.0 s (10x on the Ryzen alone). CPU and
GPU together (the CPU takes about 11% of the rows) cut the 71.6M step by 9%, from 0.281 to 0.256 s.

What did not help: the L2 fetch-granularity hint (ignored by this GPU), more than 4 threads per row, and pushing hub
contributions with atomics at 10^9 (slower than reading the in-lists).

Machine facts that shaped this: Windows pins at most 15 GB in total (11 GiB in one allocation), and the NVMe drive
reads at only 1.6 GB/s because it sits in a PCIe 3.0 x2 M.2 slot (the CPU-attached x4 slot would give about
3.4 GB/s).

Fixed on the way: since c99bdcf, finalize_growth freed the edge buffer, which broke mid-run node-growth
interventions; finalize_growth(release_edges=False) keeps it.

## B. Generator (details: gen/results.md)

src/capstone_optimized/scale/gen.py. Makes the same CuPy random calls as grow_batch (so the samples are
bit-identical), resolves stub indices arithmetically or from the stored targets (VRAM cache + pinned 1 GiB chunks),
and builds the in part in passes sized to VRAM. Checked identical to the repo at 1.1M and 17.9M (seed 493, cached
CSRs) and against the repo's growth run in the same process (seeds 7 and 21, including runs that force every lookup
through pinned memory and many fill passes).

It takes 6.2 s at 71.6M nodes (2.1 GB peak RSS, 2.1 GB on disk), 7.3 s at 143.2M (3.9 GB, 4.3 GB), 21 s at 286.3M
(5.6 GB, 8.5 GB), 79 s at 572.7M (8.3 GB, 17.1 GB) and 115 s at 999.9M (15.0 GB, 29.8 GB).

## C. Engine (details: engine/results.md)

src/capstone_optimized/scale/engine.py, with the native code in scale/native/ (feeder.dll: unbuffered NVMe reads
and memcpy into pinned slots; cpu_step.dll: the CPU draw with prefetching). Placements: VRAM, pinned (zero copy),
pageable RAM, streamed from the drive, or a mix sized from free RAM; optional finite memory (bit ring in RAM or a
file) and push mode for hub rows.

At 71.6M nodes the repo's fused step takes 0.290 s. The engine takes 0.280 s with the adjacency in VRAM, 0.274 s
with it in pinned RAM, 1.06 s with all of it read from the drive each step (drive-bound, 1.6 GB/s), 0.256 s from
VRAM with the CPU drawing about 11% of the rows, and 1.49 s on the CPU alone with 8 threads (SciPy: 15.0 s). At
999.9M nodes, with the adjacency mixed across pinned RAM, pageable RAM and the drive, a step takes 3.93 s.

## Data and how to regenerate it

The 10^9-node graph (29.8 GB) was deleted afterwards to give C: its space back. To rebuild it and time the step
(about 2 minutes to generate, then about 4 s per step), from the repository root:

    python run_sim.py --backend engine --batches 1788 --placement mixed --block-mb 96 --block-rows 8388608 --auto-budget --steps 5

It needs about 30 GB free on the drive and about 20 GB of free RAM (close other programs). Graphs, CSR caches and
the reference copy of the code for the e2e checks live in work/ at the repository root (ignored by git), or in the
folder named by the POLYA_WORK environment variable; see src/capstone_optimized/scale/paths.py. The scripts here
read from the same place, e.g. bench_engine.py --dir n17895744_split uses work/data/n17895744_split.

## A. Memory access (details: quick/results.md)

Effects are at 71.6M nodes unless noted.

L2 fetch granularity 64 -> 32 B (cudaLimitMaxL2FetchGranularity): no effect, the same time and the same 29.8 GB of
DRAM reads per call, because the hint is ignored on this GPU. Dropped.

16-bit urns (one 32-bit word per node): the kernel is 2% faster (a miss costs the same however few of its bytes are
used) and the urns take half the memory. Used by the engine.

Philox random numbers in the kernel: 7 ms less per step and no 573 MB random array. Used by the engine.

R runs per gather (urns of R runs side by side): the per-run step goes from 269 to 43 ms at R = 8 (growth order)
and from 186 to 31 ms (BFS order), and takes 13.5 ms per run at R = 16 (35.8M). In scale/replicas.py.

Finite memory as a ring of bits instead of int32: at 17.9M with M = 72 the step goes from 294 to 64 ms, and 71.6M
goes from out of memory to 269 ms (the same as without memory). In Polya_Process (identical urns).
