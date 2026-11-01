# Fused kernel, locality and streaming experiments (first optimization night)

The harness behind the website's "A fused gather-and-draw kernel", "Locality" and "Graphs larger than VRAM"
sections. The fused kernel itself now lives in `src/capstone_optimized/core/csr_kernels.py`; these scripts are
the stand-alone prototype (`kern.py`, the kernel the Nsight screenshots call `draw2`) and the tools used to time
and check it. Run them from this folder; graphs and outputs go to `work/` (see `src/capstone_optimized/scale/paths.py`),
and the original code they compare against is expected at `work/orig_src` (`git archive fd79e4d src/capstone_optimized | tar -x --strip-components=1 -C work/orig_src`).

| file | what |
|---|---|
| `build_graph.py` | grows the benchmark's BA graph with the original code and caches its CSR in `work/cache/n<nodes>` (`python run_sim.py --save-csr` writes the same format) |
| `reorder.py` | renumbers a cached CSR: `random`, `deg`, `bfs`, `rcm`; the orderings table on the website |
| `bench_step.py` | times one step on a cached CSR: `resident`, `zerocopy` (indices in pinned RAM), `stream` (chunks copied over PCIe), `orig` (the original cuSPARSE step); appends to `results.jsonl` |
| `kern.py`, `validate.py` | the prototype kernels, and a bit-exactness check of them against the original step |
| `e2e.py`, `e2e_suite.sh`, `e2e_suite_exp.sh` | full pipeline (growth, CSR, steps) with the original, the current or the night-2 code (`work/base_src`), urns compared bit for bit |
| `fig_opt.py`, `fig/` | the step-time figure; `bench_new/` holds the benchmark re-run with the fused kernel |
| `results.jsonl` | every timing taken, one JSON line per run |
