"""
Checks for scale.replicas.ReplicaPolya on a cached graph (default the 1.1M one):
- run 0 of R = 2, 4, 8 equals R = 1 (same delta, Philox stream 0), with and without memory
- R = 1 with memory equals the separately verified bit-ring path (draw_u16 + apply_u16_mem), which equals
  the repo's Polya_Process with memory (bench_mem.py --verify)
- per-run deltas: total balls per run = 2n + steps * n * delta_r (no memory), 2n + M * n * delta_r (memory, after M)
- inject adds balls to one run only
"""
import os
import sys

import numpy as np
import cupy as cp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from capstone_optimized.scale import quick_kernels as qk           # noqa: E402
from capstone_optimized.scale.replicas import ReplicaPolya         # noqa: E402

from capstone_optimized.scale.paths import CACHE
name = sys.argv[1] if len(sys.argv) > 1 else "n1118544"
indptr = cp.asarray(np.load(os.path.join(CACHE, name, "indptr.npy")))
indices = cp.asarray(np.load(os.path.join(CACHE, name, "indices.npy")))
n = indptr.size - 1
ok = True


def check(label, cond):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label, flush=True)


for M in (0, 10):
    steps = 30
    base = ReplicaPolya(indptr, indices, 1, deltas=[2], memory=M, seed=5)
    for _ in range(steps):
        base.step()
    for R in (2, 4, 8):
        rp = ReplicaPolya(indptr, indices, R, deltas=[2] + [1] * (R - 1), memory=M, seed=5)
        for _ in range(steps):
            rp.step()
        check(f"M={M} R={R}: run 0 equals the single run", cp.array_equal(rp.urns(0), base.urns(0)))
        tot = rp.urns().sum(axis=(0, 2)).get()
        exp = [2 * n + min(steps, M if M else steps) * n * d for d in [2] + [1] * (R - 1)]
        check(f"M={M} R={R}: balls per run {list(tot)} == {exp}", list(tot) == exp)

# R = 1 with memory vs the bit-ring path verified against the repo
M, steps = 10, 30
rp = ReplicaPolya(indptr, indices, 1, memory=M, seed=9)
words = cp.full(n, 1 | (1 << 16), cp.uint32)
ring = cp.zeros((M, (n + 31) // 32), cp.uint32)
draws = cp.empty(n, cp.uint8)
for s in range(steps):
    rp.step()
    qk.draw_u16(indptr, indices, words, 1, draws, None, seed=9, step=s)
    qk.apply_u16_mem(words, draws, ring, (s + 1) % M, s + 1 > M)
check("R=1 memory equals apply_u16_mem path", cp.array_equal(rp.urns(0), qk.unpack_u16(words)))

rp = ReplicaPolya(indptr, indices, 4, seed=1)
before = rp.urns().copy()
rp.inject(2, [0, 5, 7], red=10)
diff = (rp.urns() - before).get()
check("inject touches one run only", diff[:, 2, 0][[0, 5, 7]].tolist() == [10] * 3 and np.abs(diff).sum() == 30)

try:
    ReplicaPolya(indptr, indices, 1, urns=cp.full((n, 1, 2), 65530, cp.int32), deltas=[10]).step()
    check("overflow detected", False)
except OverflowError:
    check("overflow detected", True)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
