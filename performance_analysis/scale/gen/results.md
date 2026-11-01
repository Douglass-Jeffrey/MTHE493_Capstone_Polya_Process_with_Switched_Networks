# Workstream B: out-of-core batched BA generator (scale.gen)

`src/capstone_optimized/scale/gen.py` grows the repo's batched BA graph with the same CuPy random calls as
`Barabasi_Albert_Growth.grow_batch` and writes the split format directly. Stubs are resolved arithmetically
(clique head, source stubs) or from the stored targets (the out part, pinned host RAM + a VRAM prefix cache).
The in part is built afterwards: in-degrees counted on the GPU (in_ptr written in chunks), then row ranges filled
with 64-bit (row, node) keys, sorted, and written.

## Validation (`performance_analysis/scale/gen/test_gen.py`, all PASS)

| case | nodes | result |
|---|---|---|
| seed 493, batches 2, vs `data/n1118544_split` (repo CSR cached 2026-10-29) | 1,118,544 | identical (in_ptr, in_idx, every out row) |
| seed 493, batches 32, vs `data/n17895744_split` | 17,895,744 | identical |
| same two, pinned chunks of 2^20 entries and no VRAM cache (every target lookup through the chunk table) | | identical |
| seed 7, batches 4, vs the repo's growth + finalize_growth run in the same process, in part built in 7 passes (key cap 10^6) | 2,237,024 | identical |

17.9M nodes: generator 1.9 s in total (growth 0.8 s, in part 1.1 s), against 1.1 s growth + 5.7 s CSR for the
committed code when the cache was built.

## Pinned memory on this machine (`probe_pinned.py`, `probe_pinned2.py`)

- A single `cudaHostAlloc` works up to 11 GiB; 12 GiB fails (`cudaErrorMemoryAllocation`).
- With 1 GiB allocations, 15 GiB in total could be pinned (available RAM went from 21.7 GB to 6.5 GB); the 16th failed.
- So the 10^9-node out part (11.2 GiB) is held as 1 GiB chunks (`gen.PinnedChunks`), indexed by kernels through a
  device table of chunk addresses. This limit matters for workstream C as well.

## Generation runs

Times in seconds. RSS = peak resident set of the process (pinned memory included); avail = lowest system
available RAM seen; VRAM free = lowest free VRAM seen.

| name | nodes | growth | in-degree + in_ptr | in fill | out write wait | total | peak RSS GB | min avail RAM GB | min VRAM free GB | disk GB | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ba_b128 | 71,582,784 | 2.86 | 1.29 | 1.76 (1 passes) | 3.04 | 6.16 | 2.09 | 19.59 | 0.93 | 2.13 | valid |
| ba_b256 | 143,165,504 | 4.9 | 2.61 | 3.63 (2 passes) | 6.24 | 11.57 | 3.14 | 18.33 | 0.83 | 4.27 | valid |
| ba_b1024 | 572,661,824 | 19.35 | 24.05 | 34.11 (8 passes) | 58.16 | 79.22 | 8.3 | 13.26 | 0.0 | 17.07 | valid |
| ba_1b | 999,921,184 | 40.44 | 25.57 | 46.89 (14 passes) | 72.46 | 115.34 | 15.0 | 5.92 | 0.0 | 29.8 | INVALID: in row 0 lists a node that did not choose 0; in row 1 lists a node that did not choose 1; in row 2 lists a node that did not choose 2 |
| ba_1b | re-validation | | | | | | | | | | valid, READY |
| ba_b512 | 286,330,944 | 8.7 | 4.42 | 13.27 (4 passes) | 17.69 | 27.23 | 6.84 | 17.66 | 0.0 | 8.53 |  |
| ba_b512 | 286,330,944 | 9.03 | 4.44 | 6.68 (6 passes) | 11.12 | 21.02 | 5.64 | 18.45 | 1.91 | 8.53 |  |
| ba_b512 | 286,330,944 | 8.97 | 4.4 | 8.45 (5 passes) | 12.85 | 22.66 | 6.15 | 18.17 | 1.91 | 8.53 |  |
| ba_b256 | 143,165,504 | 1.93 | 2.23 | 2.61 (3 passes) | 4.84 | 7.25 | 3.92 | 20.4 | 2.55 | 4.27 |  |
