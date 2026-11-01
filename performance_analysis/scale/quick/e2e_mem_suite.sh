#!/bin/bash
# Bit-exactness of this branch's Polya_Process (bit-ring memory) against polya-step-experiments (int32 memory).
# Usage: bash e2e_mem_suite.sh   (runs under the GPU lock)
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
OUT=e2e_out
mkdir -p $OUT
run() {  # name, args...
    local name=$1; shift
    python ../gpulock.py A-e2e -- python e2e_mem.py --src orig --out $OUT/${name}_orig "$@" 2>&1 | grep -v "^using\|gpulock" | tail -1
    python ../gpulock.py A-e2e -- python e2e_mem.py --src new --out $OUT/${name}_new "$@" 2>&1 | grep -v "^using\|gpulock" | tail -1
    python -c "
import numpy as np
a, b = np.load('$OUT/${name}_orig_urns.npy'), np.load('$OUT/${name}_new_urns.npy')
print(f'== ${name}: urns identical: {np.array_equal(a, b)}  (balls {int(a.sum()):,})')"
}
run c2_mem5_delta   --batches 2 --steps 25 --delta random --memory 5 --random-urns
run c2_mem10        --batches 2 --steps 40 --memory 10
run c2_mem72        --batches 2 --steps 100 --memory 72
run c2_mem7_fdelta  --batches 2 --steps 30 --memory 7 --delta random_f
run c2_mem3_host    --batches 2 --steps 20 --memory 3 --placement host --random-urns
# 3 colours: both versions fail in Polya_Process._draws (pre-existing: rand_vals > cumulative needs
# rand_vals[:, None]), so there is nothing to compare; the bit ring only applies to 2 colours anyway
run c2_mem3_small  --batches 10 --sizes 8,128,2048 --steps 20 --random-urns --memory 3 --delta random
run c2_nomem        --batches 2 --steps 20
