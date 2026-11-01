"""Pinned memory limits: largest single cudaHostAlloc, and the total reachable with 1 GB allocations."""
import numpy as np, cupy as cp, psutil
GB = 2**30
def try_alloc(nbytes):
    try:
        return cp.cuda.runtime.hostAlloc(nbytes, 0)
    except Exception as e:
        return None
print("avail RAM at start %.1f GB" % (psutil.virtual_memory().available / GB), flush=True)
for gb in [6, 7, 8, 9, 10, 11]:
    p = try_alloc(gb * GB)
    print(f"single {gb} GB: {'ok' if p else 'FAILED'}", flush=True)
    if p: cp.cuda.runtime.freeHost(p)
ptrs = []
while len(ptrs) < 24:
    p = try_alloc(GB)
    if not p: break
    ptrs.append(p)
print(f"total with 1 GB allocations: {len(ptrs)} GB, avail RAM now {psutil.virtual_memory().available / GB:.1f} GB", flush=True)
# can the GPU read the last chunk (mapped)?
k = cp.RawKernel(r'extern "C" __global__ void rd(const int* p, int* o){ o[0] = p[0] + p[1]; }', 'rd')
o = cp.zeros(1, cp.int32)
if ptrs:
    a = np.frombuffer((np.ctypeslib.as_array if False else None) or b"", np.int8) if False else None
    k((1,), (1,), (np.uint64(ptrs[-1]), o)); cp.cuda.Device().synchronize(); print("GPU read of last chunk ok", flush=True)
for p in ptrs: cp.cuda.runtime.freeHost(p)
