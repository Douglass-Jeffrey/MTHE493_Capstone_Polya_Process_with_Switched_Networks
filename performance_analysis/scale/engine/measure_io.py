"""Measures the limits the out-of-core engine is designed around:
  1. how much host memory cudaHostAlloc will pin (1 GB at a time, up to --max-gb)
  2. unbuffered NVMe read speed through native/feeder.dll, by threads and chunk size
  3. zero-copy read throughput: a kernel summing a pinned buffer over PCIe
"""
import argparse
import ctypes
import json
import os
import sys
import time

import cupy as cp
import numpy as np

from capstone_optimized.scale.feeder import Feeder

ap = argparse.ArgumentParser()
ap.add_argument("--max-gb", type=int, default=18)
ap.add_argument("--file", required=True)
ap.add_argument("--skip-pin", action="store_true")
args = ap.parse_args()
res = {}

# 1. pinning limit
if not args.skip_pin:
    bufs = []
    t0 = time.time()
    try:
        for i in range(args.max_gb):
            bufs.append(cp.cuda.runtime.hostAlloc(1 << 30, 0))
    except Exception as e:
        res["pin_error"] = str(e)
    res["pinned_gb_ok"] = len(bufs)
    res["pin_s"] = round(time.time() - t0, 2)
    for p in bufs:
        cp.cuda.runtime.freeHost(p)
    print("pinned", res, flush=True)

# 2. unbuffered read speed
size = os.path.getsize(args.file) // 4096 * 4096
buf = cp.cuda.alloc_pinned_memory(size + 4096)
ptr = (buf.ptr + 4095) // 4096 * 4096
reads = []
for threads in (1, 2, 4, 8):
    for chunk_mb in (1, 4, 16):
        f = Feeder(threads=threads, slots=1, chunk=chunk_mb << 20)
        fi = f.open(args.file, unbuffered=True)
        t0 = time.time()
        f.submit(0, fi, 0, size, ptr)
        f.wait(0)
        dt = time.time() - t0
        f.close()
        reads.append({"threads": threads, "chunk_mb": chunk_mb, "GBps": round(size / dt / 1e9, 2)})
        print(reads[-1], flush=True)
res["nvme_unbuffered"] = reads
# check the data against a plain read
ref = np.fromfile(args.file, dtype=np.uint8, count=1 << 20, offset=size - (1 << 20))
got = np.ctypeslib.as_array((ctypes.c_uint8 * (1 << 20)).from_address(ptr + size - (1 << 20)))
res["read_check"] = bool(np.array_equal(ref, got))

# 3. zero-copy sequential read throughput
src = r"""
extern "C" __global__ void sum_u4(const uint4* __restrict__ a, long long n, unsigned long long* out) {
    unsigned long long s = 0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        uint4 v = a[i]; s += v.x + v.y + v.z + v.w;
    }
    atomicAdd(out, s);
}"""
k = cp.RawModule(code=src).get_function("sum_u4")
out = cp.zeros(1, cp.uint64)
n16 = size // 16
for grid in (96, 192, 768):
    k((grid,), (256,), (np.uint64(ptr), np.int64(n16), out))
    cp.cuda.Device().synchronize()
    ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
    ev0.record()
    for _ in range(3):
        k((grid,), (256,), (np.uint64(ptr), np.int64(n16), out))
    ev1.record()
    ev1.synchronize()
    ms = cp.cuda.get_elapsed_time(ev0, ev1) / 3
    res.setdefault("zerocopy", []).append({"grid": grid, "GBps": round(size / ms / 1e6, 2)})
    print(res["zerocopy"][-1], flush=True)
print(json.dumps(res))
