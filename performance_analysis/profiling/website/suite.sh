#!/bin/bash
# Overnight Nsight suite for the write-up (run elevated, one job at a time). Progress lines start with "##".
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
PY="${PY:-python}"
WORK="${POLYA_WORK:-$(cd ../../.. && pwd -W)/work}"   # as src/capstone_optimized/scale/paths.py
REP="$WORK/reports/website"
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2026.1.2/target-windows-x64/nsys.exe"
NCU="/c/Program Files/NVIDIA Corporation/Nsight Compute 2025.3.1/ncu.bat"
OPT="../../fused_kernel"
mkdir -p "$REP"
MID=4,64,1024,16384,262144
BIG=8,128,2048,32768,524288
SECTIONS="--section SpeedOfLight --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart
 --section MemoryWorkloadAnalysis_Tables --section WarpStateStats --section SchedulerStats --section Occupancy
 --section LaunchStats --metrics pcie__read_bytes.sum,pcie__write_bytes.sum"

state() {
    "$PY" -c '
import psutil, subprocess, sys, time
v = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
t = time.strftime("%H:%M:%S")
print(f"## {t} | {sys.argv[1]} | free RAM {psutil.virtual_memory().available / 1e9:.1f} GB | VRAM used {v} MiB | CPU {psutil.cpu_percent(1):.0f}%", flush=True)' "$1"
}

nsys_run() {  # name, python args...
    local name=$1; shift
    local t=$(date +%s)
    "$NSYS" profile -t cuda,nvtx,cusparse,wddm --gpu-metrics-devices=0 --gpu-metrics-set=tu11x --gpu-metrics-frequency=2000 \
        -s none --cpuctxsw=none --cuda-memory-usage=true --force-overwrite=true -o "$REP/$name" "$PY" "$@" > "$REP/$name.log" 2>&1
    "$NSYS" export -t sqlite --force-overwrite=true -o "$REP/$name.sqlite" "$REP/$name.nsys-rep" > /dev/null 2>&1
    state "$name done in $(( $(date +%s) - t ))s: $(grep -E '^step|median|RAM GUARD|Error' "$REP/$name.log" | tr '\n' ' ' | cut -c1-300)"
}

ncu_run() {  # name, kernel regex, launch-skip, extra ncu flags, python args...
    local name=$1 kern=$2 skip=$3 extra=$4; shift 4
    local t=$(date +%s)
    "$NCU" --target-processes all --clock-control none --kernel-name "regex:$kern" --launch-skip "$skip" --launch-count 1 \
        $extra $SECTIONS -o "$REP/$name" -f "$PY" "$@" > "$REP/$name.log" 2>&1
    state "$name done in $(( $(date +%s) - t ))s: report $( [ -f $REP/$name.ncu-rep ] && echo ok || echo MISSING ), $(grep -c 'pass' "$REP/$name.log") pass lines"
}

state "start"
# ---- part 1: original code baseline that fits in VRAM (8.9M) ----
nsys_run a9   profile_step.py --src orig --batches 32 --sizes $MID --steps 5 --probe --out $REP/a9.json
ncu_run  cf9  csrmm 1 "" profile_step.py --src orig --batches 32 --sizes $MID --steps 2

# ---- part 2: new code, same harness and flags as the original-code admin runs ----
nsys_run b9   profile_step.py --src new --batches 32  --sizes $MID --steps 5 --probe --out $REP/b9.json
nsys_run b18  profile_step.py --src new --batches 64  --sizes $MID --steps 5 --probe --out $REP/b18.json
nsys_run b36  profile_step.py --src new --batches 64  --sizes $BIG --steps 5 --probe --out $REP/b36.json
nsys_run b72  profile_step.py --src new --batches 128 --sizes $BIG --steps 5 --probe --out $REP/b72.json
ncu_run  nf72 draw_two 1 "" profile_step.py --src new --batches 128 --sizes $BIG --steps 2

# step loop alone on the cached 71.6M CSR: indices in VRAM, zero-copy from pinned RAM, explicit chunk streaming
nsys_run r72  $OPT/bench_step.py n71582784 --mode resident --lanes 4 --steps 5 --tag prof2
nsys_run z72  $OPT/bench_step.py n71582784 --mode zerocopy --lanes 4 --steps 5 --tag prof2
nsys_run s72  $OPT/bench_step.py n71582784 --mode stream --chunk-mb 128 --lanes 4 --steps 5 --tag prof2
# locality: same kernel on the growth-ordered and the BFS-ordered graph, plus zero-copy
ncu_run  kr72  draw2 2 "" $OPT/bench_step.py n71582784     --mode resident --lanes 4 --steps 1 --tag ncu2
ncu_run  kb72  draw2 2 "" $OPT/bench_step.py n71582784_bfs --mode resident --lanes 4 --steps 1 --tag ncu2
ncu_run  kz72  draw2 2 "" $OPT/bench_step.py n71582784     --mode zerocopy --lanes 4 --steps 1 --tag ncu2

# ---- long jobs last ----
# beyond VRAM: 143M nodes, CSR indices in pinned host memory
nsys_run b143 profile_step.py --src new --batches 256 --sizes $BIG --steps 5 --probe --free-pool --out $REP/b143.json
# original cuSPARSE kernel at 71.6M with full memory sections; application replay (kernel replay would back up ~19 GB)
ncu_run  cf72 csrmm 1 "--replay-mode application" profile_step.py --src orig --batches 128 --sizes $BIG --steps 2
state "SUITE COMPLETE"
