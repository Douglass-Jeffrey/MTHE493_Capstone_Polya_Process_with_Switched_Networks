# Admin-only measurements, run one at a time on an idle machine. Progress lines start with "##".
$S = $PSScriptRoot
$R = if ($env:POLYA_WORK) { Join-Path $env:POLYA_WORK 'reports\spill' } else { Join-Path $S '..\..\..\work\reports\spill' }
New-Item -ItemType Directory -Force $R | Out-Null
$A = "$R\admin"
$py = "python"
$nsys = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.2\target-windows-x64\nsys.exe"
$ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.1\ncu.bat"
$big = '8,128,2048,32768,524288'
$m1 = "gpu__time_duration.sum,pcie__read_bytes.sum,pcie__write_bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,profiler__replayer_passes"
New-Item -ItemType Directory -Force $A | Out-Null

function State($tag) {
    $os = Get-CimInstance Win32_OperatingSystem
    $v = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits).Trim()
    $d = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
    "## {0} | {1:HH:mm:ss} | free RAM {2:N1} GB | VRAM used {3} MiB | CPU {4}% | disk free {5} GB" -f $tag, (Get-Date),
        ($os.FreePhysicalMemory / 1MB), $v, (Get-CimInstance Win32_Processor).LoadPercentage, $d
}

State "start"

# 1. ncu calibration of pcie__read_bytes
& $ncu --target-processes all --clock-control none --nvtx `
    --nvtx-include "calib_vram_seq/" --nvtx-include "calib_vram_gather/" `
    --nvtx-include "calib_sys_seq/" --nvtx-include "calib_sys_gather/" `
    --metrics $m1 -o "$A\ncu_calib" -f $py "$S\pcie_calib.py" 2>&1 | Out-File -Encoding utf8 "$A\ncu_calib.log"
State "ncu calibration done"

# 2. nsys with GPU Metrics (PCIe / DRAM throughput) and the WDDM trace (VRAM vs shared usage, paging)
foreach ($r in @(@('a36', '64', 'none', '5'), @('a72', '128', 'none', '3'), @('a36f', '64', 'both', '5'), @('a72f', '128', 'both', '4'))) {
    $t = Get-Date
    & $nsys profile -t cuda,nvtx,cusparse,wddm --gpu-metrics-devices=0 --gpu-metrics-set=tu11x --gpu-metrics-frequency=2000 `
        -s none --cpuctxsw=none --cuda-memory-usage=true --force-overwrite=true -o "$A\$($r[0])" `
        $py "$S\profile_polya.py" --batches $r[1] --sizes $big --steps $r[3] --probe --fix $r[2] --out "$A\$($r[0]).json" 2>&1 |
        Out-File -Encoding utf8 "$A\$($r[0]).nsys.log"
    $pt = ((Get-Date) - $t).TotalSeconds
    & $nsys export -t sqlite --force-overwrite=true -o "$A\$($r[0]).sqlite" "$A\$($r[0]).nsys-rep" 2>&1 | Out-Null
    if (Test-Path "$A\$($r[0]).json") { $j = Get-Content "$A\$($r[0]).json" -Raw | ConvertFrom-Json; $st = ($j.steps | ForEach-Object { "{0:N3}" -f $_ }) -join ', ' } else { $st = 'NO JSON (failed)' }
    State ("{0} done in {1:N0}s, steps: {2}" -f $r[0], $pt, $st)
}

# 3. ncu on the SpMM kernel itself (1-pass metric set: no replay, no memory backup)
foreach ($r in @(@('c36', '64', 'none', '3', '1'), @('c72', '128', 'none', '3', '1'), @('c36f', '64', 'both', '4', '2'), @('c72f', '128', 'both', '4', '2'))) {
    & $ncu --target-processes all --clock-control none --kernel-name regex:csrmm --launch-skip $r[4] --launch-count 1 `
        --metrics $m1 -o "$A\$($r[0])" -f $py "$S\profile_polya.py" --batches $r[1] --sizes $big --steps $r[3] --fix $r[2] `
        --out "$A\$($r[0]).json" 2>&1 | Out-File -Encoding utf8 "$A\$($r[0]).ncu.log"
    $ok = Test-Path "$A\$($r[0]).ncu-rep"
    State "$($r[0]) done, report: $ok"
}
"## ADMIN SUITE COMPLETE"
