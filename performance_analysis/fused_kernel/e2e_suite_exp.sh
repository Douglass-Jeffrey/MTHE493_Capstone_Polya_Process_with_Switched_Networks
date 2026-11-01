#!/bin/bash
# Bit-exactness of the working tree against HEAD across configurations. Usage: bash e2e_suite.sh
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
PY="${PY:-python}"
WORK="${POLYA_WORK:-$(cd ../.. && pwd -W)/work}"   # as src/capstone_optimized/scale/paths.py
E="$WORK/e2e"
mkdir -p "$E"
run() {  # name, args...
    local name=$1; shift
    "$PY" e2e.py --src orig --out $E/${name}_orig "$@" 2>&1 | tail -1
    "$PY" e2e.py --src exp --out $E/${name}_exp "$@" 2>&1 | tail -1
    "$PY" -c "
import numpy as np, os
a, b = np.load('$E/${name}_orig_urns.npy'), np.load('$E/${name}_exp_urns.npy')
m = os.path.exists('$E/${name}_orig_mega.npy') and np.array_equal(np.load('$E/${name}_orig_mega.npy'), np.load('$E/${name}_exp_mega.npy'))
print(f'== ${name}: urns identical: {np.array_equal(a, b)}  mega identical: {m}  (balls {int(a.sum()):,})')"
}
run c2_default      --batches 2 --steps 30
run c2_mem_delta    --batches 2 --steps 25 --delta random --memory 5 --random-urns
run c3_delta        --batches 1 --steps 20 --colors 3 --delta random --random-urns
run c2_host         --batches 2 --steps 20 --placement host --random-urns
run c2_fdelta       --batches 1 --steps 15 --delta random_f
run c2_small_gpubuild --batches 10 --sizes 8,128,2048 --steps 20 --random-urns --memory 3 --delta random
run c3_small_gpubuild --batches 10 --sizes 8,128,2048 --steps 20 --colors 3 --random-urns
run c2_mid          --batches 32 --steps 10
