"""
Calibration for ncu's pcie__read_bytes: a sequential read and a random 8-byte gather of a 1 GiB
float64 buffer, once in VRAM and once pushed into system RAM by a VRAM-filling ballast.
Each measured kernel sits in its own NVTX range so ncu can select it with --nvtx-include.
"""
import time
import cupy as cp
from cupy.cuda import nvtx

MB = 1024 ** 2
N = 1024 * MB // 8
N_IDX = 50_000_000


def warm(buf):
    t = time.time()
    while time.time() - t < 1.0:
        buf[: 16 * MB // 8].sum()
        cp.cuda.Device().synchronize()


def run(tag, buf):
    idx = cp.random.randint(0, N, size=N_IDX, dtype=cp.int32)
    out = cp.empty(N_IDX, dtype=cp.float64)
    warm(idx.view(cp.float64))
    for name, fn in (("seq", lambda: buf.view(cp.int64).sum()), ("gather", lambda: cp.take(buf, idx, out=out))):
        nvtx.RangePush(f"calib_{tag}_{name}")
        fn()
        cp.cuda.Device().synchronize()
        nvtx.RangePop()
    del idx, out


vram = cp.ones(N, dtype=cp.float64)
run("vram", vram)
del vram
cp.get_default_memory_pool().free_all_blocks()

free, _ = cp.cuda.runtime.memGetInfo()
ballast = cp.empty(free + 256 * MB, dtype=cp.uint8)
ballast[::4096] = 1
sysbuf = cp.ones(N, dtype=cp.float64)
run("sys", sysbuf)
print("done")
