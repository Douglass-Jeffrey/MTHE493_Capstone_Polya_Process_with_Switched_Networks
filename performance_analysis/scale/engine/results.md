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

## 864 steps at 10^9 nodes with finite memory (2026-11-01, `run864/`)

`run_sim.py --backend engine --batches 1788 --placement mixed --block-mb 96 --block-rows 8388608 --pinned-gb 13
--ram-gb 5 --memory 72 --ring-file work/data/ring_1b_m72.bin --steps 864 --hist ... --progress 24` (seed 493,
Philox, delta 1, decay 1, no interventions). VS Code left ~20 GiB available, so the RAM tier was set by hand
(auto-budget would have kept only ~1.3 GiB): 139 blocks pinned (12.98 GiB of blocks + 0.56 GiB of slots), 53 in
pageable RAM (4.97 GiB), 47 from the drive (4.40 GiB, ~2.9 s per step at 1.6 GB/s). 3.5 GiB of RAM left after load.

| phase | time |
|---|---|
| generate (smoke run, same graph, `smoke.log`) | 70.5 s |
| load | 24.8 s |
| 864 steps | 3,672 s (61.2 min): median 4.221 s, steps 1-72 4.012 s, 73-864 ~4.23 s, p95 4.64 s, max 8.78 s (one) |
| whole run incl. load, per-step histograms (~0.24 s each) and md5 | 3,900.7 s (65.0 min) |

No 16-bit overflow; final urns md5 4896f1007c00d4a37082811d06508c3d. **This run was hit by a one-off fault at step 32
and its urns are not the deterministic result; the rerun of 2026-11-02 (next section) replaces it.** The ring adds
~0.2 s per step after step 72 (125 MB read back and 125 MB written, not overlapped with the kernels; the cause was
WDDM batching, fixed in the next section). Nsight GPU metrics need an elevated session, so this run has no trace;
`nsys/e1b_pull.nsys-rep` is the same graph without memory.

Red fraction (`hist864.csv`, 50 bins per step; heatmap `hist864_heatmap.png` made with the main repo's
`tools/plot_hist_heatmap.py`): symmetric about 1/2 throughout; the spread shrinks during warm-up and is constant
from ~step 200 (bin sd 0.0725, 84.9% of nodes in [0.4, 0.6), 4,422 nodes >= 0.8 and 4,421 <= 0.2 at step 864;
the histogram changes by 4e-5 of the nodes per step after step 500). Without an intervention neither colour takes
over.

`run_sim.py` changes for this run (urns and histograms unchanged on 1.1M, engine md5 4af761d1..., core 11386f4d...):
histogram by a single kernel with cupy.histogram's exact binning (0.25 s at 10^9 instead of ~2.3 s; equal to
cupy.histogram on 14 test arrays including totals 1, 2, 50, 100), md5 in chunks (no 8 GB int32 copy), the engine's
urns kept 16-bit, histogram rows written as they are made, `--progress N`, per-step times in the log.

## Nsight capture, WDDM fix and rerun at 10^9 nodes (2026-11-02, `run864/`, `nsys/`)

### Nsight Systems
Nsight Systems 2026.5.1 in an elevated session (GPU metrics need it), `--gpu-metrics-devices=all
--capture-range=cudaProfilerApi --capture-range-end=stop` around `run_sim.py ... --steps 104 --profile-steps 101-104`
with the settings above. nsys drops empty-string arguments (`--log ""` became an argparse error and nsys exited 2
with "No reports were generated"; give a real path), and the app's stdout is not shown, so read the run's summary
from the log. The reports open only in the 2026.5.1 GUI/export.

`nsys/e1b_mem72_before_fix.nsys-rep`: after step 72 the GPU was idle for 0.25 s or more before the last
`draw_block` of every step. Under WDDM the driver batches launches and submits them at the next CUDA call; after
launching the last blocks the host read the old ring row (125 MB from the memmap) before making one, so the last
blocks waited for the read.

Fix (`engine.py`): the old row is read into a pinned row on a thread started at the beginning of the step,
`cp.cuda.get_current_stream().done` (a stream query) after the last launch submits the queue, and the new row is
written to the ring on a thread while the next step draws (a second pinned row). Unchanged results: M = 3 with the
ring in RAM, a file or VRAM (b51f19cb...), M = 5 with decay 2 (deb56427...), test_memory M = 1, 3, 7, and 104 steps
at 10^9 (a81824ec... before and after).

`nsys/e1b_mem72.nsys-rep` (fixed), step 102: 4,185.6 ms, GPU busy 4,178 ms; `draw_block` 239 launches, 3.955 s;
offset kernels (deg8_to_u32, scan, copy) 0.157 s; `apply_bits_mem` 48 ms; the 125 MB row in and out, 9.5 ms each;
the host waits 2 ms for block data; SMs Active 95.7%, GR Active 99.4%, DRAM Read 34.1%, PCIe RX 59.1%, TX 12.4%.
Steps under nsys: 1-72 4.180 s, 73-100 4.192 s (before the fix: 4.231 s and 4.445 s).

### Rerun of the 864 steps (fixed engine, no profiler)

| phase | time |
|---|---|
| load | 29.8 s |
| 864 steps | 3,607 s (60.1 min): median 4.176 s, steps 1-72 4.162 s, 73-864 4.176 s, p95 4.180 s, max 4.487 s (step 1) |
| whole run incl. load, per-step histograms and md5 | 4,006.9 s (66.8 min) |

No overflow, final urns md5 a0d9f3f13c46e4620bd11c108fc8ca64, 8.4 GiB of RAM available after load. The steps after
step 72 now take as long as those before. The time outside the steps is ~0.43 s per step (0.24 s on 11-01): the
histogram kernel is slower at the lower GPU clock (below).

### Which 864-step run is right
`hist864b.csv` equals `hist864.csv` up to step 31 and differs from step 32 on: by 4,876 bin counts in total at
step 32 (~2,400 nodes in a different bin), and by a few hundred to a few thousand counts at every later step.
Two 104-step checks with the same settings, `chkA_new` (current engine) and `chkB_old` (the committed engine,
loaded in place of the working copy by a wrapper), both end at a81824ec..., like both Nsight runs, and their
histograms equal `hist864b.csv` row for row. So the engine is deterministic, the fix does not change results, and the
11-01 run went off once at step 32. The System event log has no GPU (nvlddmkm), WHEA or disk events during either
run, and the feeder (per-slot pending counts under a mutex, read errors raised) and the slot reuse (an event after
each launch) look correct. The cause is unknown; neither the GPU nor the RAM has ECC, and the graph is read from host
RAM on every step. The difference is too small to show in the statistics of the previous section, which are the
same for the rerun (sd 0.0725 from step 200, 84.9% in [0.4, 0.6), 4,422 >= 0.8 and 4,421 <= 0.2 at step 864), and
the heatmap drawn from `hist864b.csv` is pixel-identical to `hist864_heatmap.png`.

### GPU clock
The same committed engine took 4.012 s per step (steps 1-72) on 11-01 and 4.218 s on 11-02 (`chkB_old`), 4.224 s
and 4.338 s after step 72. On 11-02 the fix saves 0.05 s per step before step 72 and 0.16 s after it (`chkA_new`:
4.165 s and 4.178 s). The difference between the days is the GPU clock: the GPC clock is 1,963 MHz (median) in
`nsys/e1b_pull.nsys-rep` (10-31) and a flat 1,200 MHz in both 11-02 traces and in `clock_check_smi.csv` (nvidia-smi
during a 12-step run, `clock_check_run.log`: SM 1,200 MHz, memory 5,750 MHz, PCIe gen 3 x16, 55 W of a 150 W limit,
51 C, no power, thermal or HW slowdown, no application clocks). It was capped between 11-01 13:18 and 11-02 02:16,
with no reboot since 10-31 16:30 and no overclocking tool running; the source was not found. The step is bound by
memory, so the 39% lower clock costs ~5%: `draw_block` 3.75 s per step on 10-31 against 3.96 s, the offset kernels
0.14 s against 0.16 s. At the 10-31 clock the fixed engine should take ~4.0 s per step throughout.

A reboot (11-02 11:10) removed the cap: the same 12-step run (`work/reports/1b/smi_after_reboot*.{csv,log}`) ran
at 2,115-2,130 MHz (88 W, 52-55 C) and took 4.080 s per step against 4.152 s at 1,200 MHz, only 1.7% less, with the
user at the machine (5.9 GiB of RAM available after load against 8.4 GiB); a 4096^2 float32 matmul loop held
2,040 MHz at the 150 W power limit. So the cap explains only part of the 11-01/11-02 difference; a quiet-machine
run is needed to say more.
