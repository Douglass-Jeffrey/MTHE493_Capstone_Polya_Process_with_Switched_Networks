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

SECTIONS="--section SpeedOfLight --section MemoryWorkloadAnalysis_Chart --section WarpStateStats --section LaunchStats
 --section Occupancy --metrics pcie__read_bytes.sum,pcie__write_bytes.sum,dram__bytes_read.sum,lts__t_sector_hit_rate.pct"
state "suite3 start"
ncu_run cf36 csrmm 1 "--replay-mode application" profile_step.py --src orig --batches 64 --sizes $BIG --steps 2 --ram-guard-gb 0.5
state "SUITE3 COMPLETE"
