"""How much pinned host memory can be allocated, and how fast the GPU reads it (zero-copy, in order and at random)."""
import time, sys
import numpy as np, cupy as cp, psutil
GB = 2**30
for size_gb in [float(x) for x in sys.argv[1:]]:
    t = time.time()
    try:
        mem = cp.cuda.alloc_pinned_memory(int(size_gb * GB))
    except Exception as e:
        print(f"{size_gb} GB: FAILED {type(e).__name__}: {e}"); continue
    a = np.frombuffer(mem, np.int32, int(size_gb * GB) // 4)
    a[::1024] = 1
    print(f"{size_gb} GB pinned in {time.time()-t:.1f} s, avail RAM now {psutil.virtual_memory().available/GB:.1f} GB", flush=True)
    del a, mem
    cp.get_default_pinned_memory_pool().free_all_blocks()
