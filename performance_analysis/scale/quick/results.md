# Workstream A: memory-access experiments on resident graphs (2026-10-31)

GTX 1660 Ti, graphs from the repo's growth (seed 493, batch schedule [8,128,2048,32768,524288] x 128 = 71.6M
nodes, 429.5M stored entries), CSR in VRAM. Times are medians of 6-8 steps after warm-up, CUDA events, under
the GPU lock, machine otherwise idle. Raw lines in `raw.jsonl`. Kernels in
`src/capstone_optimized/scale/quick_kernels.py`, harness `bench_quick.py`.

## Correctness (bench_quick.py n1118544 --verify --steps 20)
- int32 urns + array random numbers: urns identical to the repo's `csr_kernels.draw_two_colour` + `apply_draws`.
- 16-bit packed urns + array random numbers: identical to the repo.
- Philox: 16-bit and int32 kernels identical to each other.
- R runs per gather: run 0 of R = 2, 4, 8 identical to the single run; the other runs differ.
- Bit-packed finite memory (bench_mem.py --verify): identical to the repo's `Polya_Process` with memory at
  M = 10 (60 steps) and M = 72 (100 steps), 1.1M nodes.

## 1. L2 fetch granularity (cudaLimitMaxL2FetchGranularity 64 -> 32 B)
The device reports 64 B by default and accepts 32. Step and kernel times are unchanged to 0.1% for every
variant (e.g. u16_phx draw kernel 265.8 ms at both settings), and so are the bytes: Nsight Systems' DRAM
read throughput integrated over the kernel (`nsys_dram.py`) gives 29.8 GB per call at both settings. The hint
is ignored on this GPU; every miss moves 64 bytes.

## 2. Urn width and random numbers, 71.6M growth order, one run
| variant | step | draw kernel | random numbers | apply |
|---|---|---|---|---|
| repo (int32, cp.random.rand array) | 290.5 ms | 275.4 ms | 7.2 ms | 7.9 ms |
| int32, array | 284.8 | 272.7 | 7.2 | 4.9 |
| int32, Philox in kernel | 275.6 | 270.7 | 0 | 4.9 |
| 16-bit, array | 277.5 | 267.8 | 7.2 | 2.6 |
| 16-bit, Philox | **268.4** | 265.8 | 0 | 2.6 |

16-bit urns save only ~2% of the kernel: in growth order almost every neighbour read is a separate cache
miss, and a miss costs the same whether 4 or 8 of its bytes are used. Philox removes the 573 MB random array
and its 7 ms fill. Together with a cheaper apply: 290 -> 268 ms (-8%), and 8 + 4 fewer bytes per node.

## 3. Several runs per gather (16-bit, Philox stream = run), per-run time
| runs R | growth order: step | per run | BFS order: step | per run |
|---|---|---|---|---|
| 1 | 268.7 ms | 268.7 | 186.5 ms | 186.5 |
| 2 | 277.3 | **138.7** | 195.8 | **97.9** |
| 4 | 328.0 | **82.0** | 252.8 | **63.2** |
| 8 | 540.5 | 67.6 | 497.6 | 62.2 |

A neighbour's R urns are R x 4 bytes in one place, so one miss brings in all of them. Memory: 4R bytes per
node of urns (2.3 GB at R = 8, 71.6M).

Threads per row matter once a row carries R runs (more work per load, and fewer threads per row put more
rows' misses in flight). 71.6M, per-run step time:

| runs R | growth, 4 lanes | growth, 2 lanes | growth, 1 lane | BFS, 4 lanes | BFS, 2 lanes | BFS, 1 lane |
|---|---|---|---|---|---|---|
| 1 | **268.7** | 272.2 | | **186.5** | | |
| 2 | 138.7 | **139.6** | 142.4 | **97.9** | 98.5 | 102.1 |
| 4 | 82.0 | **71.9** | 73.7 | 63.2 | **52.0** | 54.4 |
| 8 | 67.6 | 43.6 | **43.2** | 62.2 | 33.9 | **30.7** |

8 lanes is worse everywhere (R = 4: 132 ms per run). At 35.8M nodes (n35791424, where 16 runs fit):

| runs R | growth, 1 lane: step | per run | BFS, 1 lane: step | per run |
|---|---|---|---|---|
| 1 | 136.5 ms (133.5 with 2 lanes) | 133.5 | 96.7 (93.7 with 2 lanes) | 93.7 |
| 8 | 172.5 | 21.6 | 123.1 | 15.4 |
| 16 | 215.7 | **13.5** | 189.4 | **11.8** |

16 runs fill a 64-byte fetch exactly (16 x 4 B). Against one run per step this is 10x the throughput in growth
order; against the repo's step it is about 10-12x. Packaged as `scale/replicas.py` (`ReplicaPolya`: R runs with
their own Philox streams, deltas, decays, bit-ring memory and `inject`), tested by `test_replicas.py`.

## 4. Finite memory as a ring of bits
The repo stores the last M draws as an (M, n) int32 array: 288 B per node at M = 72, i.e. 5.2 GB at 17.9M
nodes and 20.6 GB at 71.6M. The bit ring stores 9 B per node and does expiry + record in one kernel.

| M = 72, steps past expiry | repo `Polya_Process` | bit ring (16-bit urns, Philox) |
|---|---|---|
| 17.9M nodes | 293.7 ms/step (ring spilled to system RAM) | **63.6 ms/step** |
| 71.6M nodes | out of memory (cannot allocate the 20.6 GB ring) | **268.8 ms/step** (268.4 without memory) |

### In the simulator: `Polya_Process` keeps finite memory as bits
`Polya_Process(memory_enabled=True)` with two colours on the GPU now stores the ring as (M, ceil(n/32)) uint32
and runs `csr_kernels.memory_bits_step` (expire the draw from M steps ago, record this one). `memory_arr` still
reads and assigns as an (M, n) array (unpacked/packed on access), so the node-growth intervention's `hstack`
works unchanged. Other cases (3+ colours, urns not int32 on the GPU) keep the int32 array.

`e2e_mem_suite.sh` (this branch vs polya-step-experiments, full pipeline: growth, CSR, steps): urns identical
for memory 5 with random per-node delta + random decay + remove_initial_urns, memory 10 (40 steps), memory 72
(100 steps), memory 7 with Fortran-ordered delta, memory 3 with the indices in pinned host RAM, memory 3 on small
graphs (GPU CSR path), and memory off. `e2e_nodegrowth.py`: two mid-run node-growth interventions with memory,
identical.

### Found while testing: mid-run node-growth interventions were broken
Since c99bdcf (fused kernel), `finalize_growth` frees the edge buffer, so
`Switch_Network_Node_Growth_Intervention(..., mid_polya_intervention=True)` after it has no edges to append to
(`test_node_growth_intervention_polya.py` does this). Fixed on this branch: `finalize_growth(release_edges=False)`
keeps the buffer, and that experiment passes it. Not changed, but worth knowing: when the intervention adds nodes
mid-run it extends the memory with zeros, i.e. "drew red", so for the next M steps the new nodes lose `mem_decay`
red balls for draws they never made (count can reach 0; seen in `e2e_nodegrowth.py`, min count 0, in both versions).

## 5. CPU only: the engine's native step vs SciPy
`bench_engine.py --dir n71582784_split --placement vram --cpu-blocks 18` (all rows drawn by `cpu_step.dll`, 8
threads with prefetching, 16-bit urns): **1.49 s per step** at 71.6M nodes (apply 0.05 s + draw 1.43 s), against
15.0 s for SciPy's single-threaded sparse product on the same Ryzen 3 3100: 10x on the CPU alone. The GPU's 0.28 s
is then 5x a good CPU step, not 51x.
