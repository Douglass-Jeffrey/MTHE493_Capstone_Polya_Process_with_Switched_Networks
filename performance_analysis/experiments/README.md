# Growth, CSR build and renumbering experiments (2026-10-30)

Local branch `polya-step-experiments`, on top of `optimization_extensions` (c99bdcf). Not in the website write-up.
With the fused kernel the Pólya step was no longer the slowest part: at 71.6M nodes the benchmark spent 61 s
growing the network and building the CSR matrix, and 29.6 s on 100 steps. These changes go after that 61 s.

## Changes

1. **CSR build on the GPU** (`csr_kernels.csr_from_edges`, used by `Graph.build_csr` above 10^6 edges).
   The edge list is staged in pinned host memory, so the 3.4 GB device edge buffer can be freed first. Then
   the degrees are counted, the entries placed with atomics (reading the staged list in order over PCIe), and
   every row sorted by column in chunks of 64-bit (row, column) keys. The result is the same `indptr` and
   `indices` scipy builds. Duplicate edges (weights > 1) fall back to the scipy path, as before.
2. **Growth no longer spills to system RAM.**
   - CuPy cached every buffer growth had outgrown, about 14 GB at 71.6M. These were the urn arrays replaced
     by the callbacks' `vstack` and the old stub and edge buffers. The driver pushed live arrays out of VRAM
     to make room. `grow_batch` now frees the cache, but only when VRAM runs low (each `cudaFree` costs
     ~20 ms under WDDM).
   - The stub list is no longer stored. After the initial clique it is exactly the first column of the edge
     buffer, so `_stub_values` reads `edges[r - shift, 0]`, which saves up to 2.6 GB. The samples are
     identical.
   - `Barabasi_Albert_Growth.reserve(final_nodes)` sizes the edge buffer once, so it is never resized. A
     resize holds the old and the new buffer at once. It is skipped when the buffer would not fit in VRAM:
     at 143M a buffer living in system RAM from the start made growth 4x slower.
     `test_benchmark.py` now calls it.
3. **Breadth-first renumbering on the GPU** (`Graph.bfs_order`, `Graph.renumber`). This is opt-in and not
   called anywhere, because it changes node ids (`graph.node_order` maps them back). It uses one kernel
   launch per BFS level. Each node's key is (rank of its first-visited parent, node id), so the order is
   exactly scipy's `breadth_first_order`. Rows are relabelled into a VRAM chunk buffer and copied out in
   bulk.

## Results (GTX 1660 Ti, same harness and seed)

| 71.6M nodes | committed (fused kernel) | this branch |
|---|---|---|
| growth | 28.7 s | 5.1 s (8.9 s without `reserve`) |
| CSR build | 38.4 s | 2.6 s |
| 100 steps | 29.5 s | 29.5 s |
| BFS renumbering | 98 s (scipy, CPU) | about 4 s, then 100 steps take 21.2 s instead of 29.5 s |

`test_benchmark.py`, total time (growth + CSR + 100 steps):

| nodes | Ryzen 3 3100 | original | fused kernel | this branch | A100, original |
|---|---|---|---|---|---|
| 17.9M | 348 s | 25.3 s | 14.2 s | 9.2 s | 9.3 s |
| 35.8M | 720 s | 388 s | 36.7 s | 18.1 s | |
| 71.6M | 1,565 s | 1,557 s | 90.7 s | 36.9 s | 37.9 s |

At 143.2M nodes, where the CSR indices live in pinned system RAM: CSR build 82 s -> 16 s, growth 63 s -> 51 s
(`reserve` skips itself here). Renumbering takes 17.6 s at that size, because every BFS level reads the 3.4 GB
of indices over PCIe, and cuts a step from 0.58 s to 0.44 s. It pays off after about 125 steps.
`node_order` is kept in host memory: on the GPU its 573 MB pushed the step's arrays into system RAM, and a
step took 0.97 s.

## Checks

- MD5 of the final urns, `indptr` and `indices` equal to the committed code at 17.9M, 71.6M and 143.2M
  nodes, and equal to the original code (fd79e4d) at 17.9M.
- The end-to-end suite (memory, random delta, Fortran-ordered delta, host placement, small graphs on the
  cupyx path) gives identical urns. With 3 colours, both versions fail the same way: an existing bug in
  `Polya_Process._draws`, where `rand_vals > cumulative` needs `rand_vals[:, None]`.
- `experiments/test_gpu_csr_build.py`: identical to scipy with the indices in VRAM and in pinned memory; the
  duplicate-edge fallback works.
- `experiments/test_bfs_renumber.py`: order identical to scipy's, renumbered CSR identical to `A[p][:, p]`,
  draws identical after permuting.

## Notes

- The initial clique's edges are entered in the stub list twice (`src` and `dst`, while both directions are
  already in `src`), so the clique nodes start with double weight. This is kept as it was, so the results
  stay identical.
- The urn callbacks in the experiment scripts still `vstack` the whole urn array every batch (quadratic
  copying). It is cheap now that it stays in VRAM, but a capacity-managed urn buffer would remove it.
