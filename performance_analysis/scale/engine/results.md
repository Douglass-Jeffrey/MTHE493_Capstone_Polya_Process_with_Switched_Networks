# Workstream C: out-of-core step engine, results (2026-10-31)

## Machine limits (measure_io.py, under the GPU lock)

| what | result |
|---|---|
| cudaHostAlloc, 1 GB at a time | 15 GB pinned, the 16th GB fails (out of memory) |
| NVMe unbuffered read (feeder.dll), 1-8 threads, 1-16 MB chunks | 1.34 GB/s (1 thread, 1 MB) to 1.63 GB/s; flat from 2 threads up |
| why the NVMe is slow | the WD SN750 links at PCIe 3.0 **x2** (max x4) on this B450 AORUS ELITE V2 slot: ~1.97 GB/s raw. The drive in the CPU-attached M.2 slot would read ~3.4 GB/s |
| zero-copy read of pinned RAM by a kernel | 13.2 GB/s (any grid size) |

Consequence for 10^9 nodes (24 GB of adjacency, 1 GB of in-degrees): at most ~13 GB can stay resident in pinned
RAM, so ~11 GB must come from the drive every step, ~7 s at 1.6 GB/s, unless more RAM is used (pageable tier).

## Verification (test_engine.py)

17.9M nodes (data/n17895744_split, 18 blocks of 2^20 rows), 5 steps, seed 7:
- (a) with the repo's random numbers (`cp.random.seed(7)`, `cp.random.rand(n)` per step), the engine's 16-bit
  urns equal the repo's fused step (`csr_kernels.draw_two_colour` + `apply_draws`, int32 urns) **after every
  step**, for placements vram, pinned, stream and mixed.
- (b) Philox: identical urns (md5 1889f83d...) for vram, pinned, stream, mixed.
- (c) CPU+GPU (3 of 18 blocks drawn by the CPU threads): identical urns too.
- `cpu_step.dll`'s draws equal a Python reference (Philox reference + exact sums) on 370 sampled rows (1.1M graph).

Step times at 17.9M (Philox): vram 0.064 s, pinned 0.066 s, stream 0.265 s (all 430 MB from the drive each
step: 1.6 GB/s), vram + 3 CPU blocks 0.061-0.064 s (no real gain; CPU draw 33-74 ms for 17% of the rows).
For reference the repo's fused step at 17.9M took ~71 ms (2026-10-29 measurements).

71.6M nodes (data/n71582784_split, 18 blocks of 2^22 rows), 3 steps, seed 7: (a) urns equal to the repo after
every step for vram, pinned, mixed; (b) Philox md5 270205ad... identical for vram, pinned, mixed and vram+CPU.

## Step time at 71.6M (bench_engine.py, Philox, 16-bit urns, LANES 4, 10 timed steps after 1 warm-up)

| configuration | step (s) | notes |
|---|---|---|
| repo fused step (int32 urns, cuRAND array), for reference | 0.29 | kernel 0.275 s (2026-10-29) |
| vram (adjacency in device memory) | 0.280 | 3.1 GiB VRAM free |
| vram, cudaLimitMaxL2FetchGranularity = 32 | 0.280 | no effect |
| vram, LANES 2 | 0.284 | |
| vram, 5 blocks of 2^24 rows | 0.279 | block count does not matter |
| pinned (all adjacency in pinned RAM, zero copy) | **0.274** | slightly faster than VRAM: the index stream moves to PCIe, leaving VRAM bandwidth to the random urn reads; 4.7 GiB VRAM free |
| stream (all 1.72 GB from the drive each step) | 1.06 | drive-bound, 1.6 GB/s |
| mixed: 10 blocks pinned, 8 streamed (0.67 GB/step from the drive) | 0.415 | |
| mixed: 4 pinned + 14 pageable-RAM blocks (memcpy into pinned slots by the feeder threads) | 0.283 | the RAM tier costs ~3% |
| all 18 blocks in the pageable-RAM tier | 0.284 | memcpy keeps up easily (6 GB/s needed) |
| vram + CPU draws the last block (4.2M rows) | 0.281 | CPU: apply 29 ms + draw 8 ms |
| vram + CPU draws the last 2 blocks (8.4M rows) | 0.275 | CPU: apply 33 ms + draw 99 ms; -2% |

16-bit urns do not make the kernel faster (0.280 vs 0.275 s for the int32 fused kernel): the kernel is bound by
the number of random urn reads (each a DRAM transaction), not by their size; the 32 B L2 fetch granularity
result agrees.

### Byte-sized blocks, CPU+GPU, Nsight (71.6M)

| configuration | step (s) | notes |
|---|---|---|
| pinned, blocks of ~96 MB (`block_mb=96`, 18 blocks) | 0.276 | same as row blocks here; byte blocks matter at 10^9, where the first 2^22 rows would hold ~780 MB of hub in-lists |
| mixed 96 MB blocks: 6 pinned, 5 RAM, 7 streamed | 0.433 | drive-bound |
| vram + CPU draws last 3 blocks (12.6M rows) | **0.263** | -6% vs 0.280; CPU apply 29 ms + draw 205 ms, hidden behind the GPU's 0.26 s |
| vram + CPU draws last 4 blocks | 0.377 | the CPU becomes the bottleneck (draw 317 ms) |
| vram + CPU draws last 5 blocks | 0.494 | |
| pinned + CPU draws last 4 blocks | 0.360 | |

Byte-sized blocks pass the 17.9M Philox check (52 blocks of 8 MB: vram, pinned, stream, mixed and vram + 4 CPU
blocks identical).

CPU+GPU at 71.6M: the Ryzen's 8 threads gather ~170-340 M urn reads/s against the GPU's ~1.85 G/s, so the best
split (CPU takes ~15% of the rows) saves ~6%. It does not apply at 10^9 nodes: there the CPU is busy feeding
blocks, the drive is the bottleneck, and a host copy of the urns would take 4 GB of the RAM the resident tiers need.

Nsight Systems (`nsys/e71_mixed.nsys-rep`, mixed: 0.6 GB pinned + 0.4 GB RAM tier + rest streamed, 3 steps):
per ~490 ms step the host waits ~280 ms for streamed block data (NVTX "wait for block data"); GPU kernel time is
262 ms/step for draw_block plus ~10 ms/step for the per-block offsets (deg8 -> uint32 + scan); PCIe RX ~36%,
SMs active ~55%. The drive is the limit as expected, and the transfers overlap the kernels.

## Finite memory (test_memory.py)

`Engine.enable_memory(M, decay, ring_path=None)`: a ring of M rows of n bits (host RAM, or a file for 10^9 nodes:
9 GB at M = 72), one row to the GPU and one back per step; `apply_bits_mem` adds delta to the drawn colour and
takes decay off the colour drawn M steps earlier, as `Polya_Process.memory_step`.
17.9M, M = 3, 9 steps, the repo's random numbers: engine urns equal to `Polya_Process(memory_enabled=True)` after
**every** step (9/9). Philox + memory: vram and mixed (ring in a file) identical.

## Push rows (test_push.py, sweep71d)

`Engine(push_k=K)`: rows [init, K) (the earliest nodes, i.e. the hubs) never read their in-lists. Every out entry
(v -> u) with u in [init, K) adds urns[v] to a VRAM accumulator S[u] (one 64-bit atomicAdd, red in the low and
black in the high 32 bits; safe while (in-degree + m) * 65535 < 2^32, checked at load), and those rows are drawn
after all blocks from S + their out-part sum + own urn (`draw_early`). In a BA graph the first K nodes hold
sqrt(K/n) of all edge ends, so a small K removes a large part of the in part from what is read every step.
`push_k=-1` takes as many rows as fit in the VRAM left (all rows at 17.9M and 71.6M).

Verification (17.9M): with the repo's random numbers, push (K = 2M) urns equal the repo after every step; with
Philox, pull/vram, push/vram (K = 2M), push/stream (K = 2M) and push/mixed (K = all rows) give identical urns,
with and without finite memory (M = 3).

| 71.6M, 96 MB blocks | step (s) | read per step |
|---|---|---|
| vram, pull (for reference) | 0.280 | - |
| vram, push all rows | 0.478 | - |
| pinned, push all rows | 0.479 | - |
| pinned, push K = 16.8M | 0.407 | - |
| stream, pull (for reference) | 1.06 | 1.72 GB |
| stream, push K = 4.2M | 0.836 | 1.36 GB |
| stream, push K = 16.8M | 0.672 | 1.09 GB |
| stream, push all rows | **0.535** | 0.87 GB (the out part only) |

The atomics cost ~0.9 ns per pushed entry (0.2 s for the 215M in-entries at 71.6M), so push loses when the data
is resident and wins when the drive is the bottleneck: at 10^9 nodes K has to balance the two.

## CPU+GPU with a prefetching CPU step (sweep71e)

`cpu_step.dll` now walks runs of 4096 rows per thread and prefetches the neighbours' urns 6 rows ahead
(`_mm_prefetch`): 524 M urn reads/s at 17.9M (was ~170-340 M), draws still equal to the Python reference.

| 71.6M, vram, 35 blocks of 48 MB | step (s) | CPU apply / draw (s) |
|---|---|---|
| GPU only | 0.281 | - |
| CPU draws the last 4 blocks | **0.256** (-9%) | 0.033 / 0.158 |
| CPU draws the last 6 blocks | 0.321 | 0.029 / 0.252 |
| CPU draws the last 8 blocks | 0.426 | 0.036 / 0.369 |

The best split hands the CPU ~11% of the adjacency; beyond that the CPU (plus the ~20-40 ms of bit exchange and
host update that are not overlapped) becomes the bottleneck.

## 10^9 nodes (data/ba_1b, written up by the coordinator after this workstream stopped at the usage limit)

Graph: 999,921,184 nodes, 2,999,765,376 edges (6.0 x 10^9 stored entries), from `scale.gen` (seed 493,
1788 batches of the benchmark schedule) in 115 s; 29.8 GB on disk. Checked by `../check_symmetry.py`: 3,270
rows (all clique rows, the largest hubs, 3,000 random rows), 6.9M edge ends, each edge present from both ends.

Step (`bench_engine.py --dir ba_1b --placement mixed --block-mb 96 --block-rows 8388608 --auto-budget`,
Philox, 16-bit urns, 3 timed steps after 1 warm-up; log `run1b.log`, commands `sweep1b.txt`):

| configuration | step (s) | adjacency per step | notes |
|---|---|---|---|
| pull, 139 blocks pinned (13.5 GiB) + 62 in pageable RAM (5.8 GiB) + 38 streamed from the drive | **3.93** (3.92-3.93) | ~10 GB through the feeder, ~3.6 GB of it from the drive | load 27 s, 1.1 GiB VRAM free, RSS 20.8 GiB |
| push K = 8.4M (hub in-lists not read) | 4.58 | 8.0 GB | atomics cost more than the reads they save |
| push K = 25.2M | 4.83 | 6.6 GB | |
| push, as many rows as fit (K = 107.9M) | 5.34 | 3.5 GB | |

Nsight Systems (`nsys/e1b_pull.nsys-rep`, 4 steps, `nsys_summary.py`): `draw_block` runs 3.8 s of each 3.9 s
step, SMs active 99%, DRAM read 36%, PCIe receive 63%; the host waits only 3-55 ms per step for block data.
The step is bound by the GPU's random urn reads (4 GB of 16-bit urns, L2 hit rate ~2% by the sqrt rule), not by
the drive: the pinned and RAM tiers plus 6 feeder slots keep the kernels fed. 14x the nodes of the 71.6M graph
take 14x the time (0.28 s -> 3.9 s), and one step is ~3.4x faster than the original code's step at 71.6M nodes.

So on this machine a 10^9-node benchmark (generate 115 s + load 27 s + 100 steps) takes about 9 minutes, and an
864-step run about an hour. Finite memory at 10^9 (a 9 GB bit ring, one row read and written per step) was not
run: C: had too little free space for the ring file. It is supported (`--memory M --ring-file`), e.g. on D:.
