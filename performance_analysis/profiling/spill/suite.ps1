# Clean re-run of every measurement, one at a time. Progress lines start with "##".
$S = $PSScriptRoot
$R = if ($env:POLYA_WORK) { Join-Path $env:POLYA_WORK 'reports\spill' } else { Join-Path $S '..\..\..\work\reports\spill' }
New-Item -ItemType Directory -Force $R | Out-Null
$py = "python"
$big = '8,128,2048,32768,524288'
$mid = '4,64,1024,16384,262144'
$small = '8,128,2048,32768'

function State($tag) {
    $os = Get-CimInstance Win32_OperatingSystem
    $v = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits).Trim()
    "## {0} | {1:HH:mm:ss} | free RAM {2:N1} GB | VRAM used by others {3} MiB | CPU {4}%" -f $tag, (Get-Date),
        ($os.FreePhysicalMemory / 1MB), $v, (Get-CimInstance Win32_Processor).LoadPercentage
}

function Quiet($block) { & $block 2>&1 | Where-Object { $_ -notmatch '^\[' -and $_ -notmatch 'NativeCommandError|RemoteException|^At |^\+|^nsys.exe :|^\s*$|CategoryInfo|FullyQualified' } }

State "start"
& $py -I "$S\gather_bench.py" "$R\gather.json"
State "gather done"

foreach ($r in @(@('n18M', '64', $mid, 'none', '5'), @('n36M', '64', $big, 'none', '5'), @('n72M', '128', $big, 'none', '3'),
                 @('n36M_fix', '64', $big, 'both', '5'), @('n72M_fix', '128', $big, 'both', '3'))) {
    Quiet { & "$S\run.ps1" -Name $r[0] -PyArgs @('--batches', $r[1], '--sizes', $r[2], '--steps', $r[4], '--probe', '--fix', $r[3]) } |
        Out-File -Encoding utf8 "$R\$($r[0]).log"
    $ok = Test-Path "$R\$($r[0]).json"
    if ($ok) { $j = Get-Content "$R\$($r[0]).json" -Raw | ConvertFrom-Json; $st = ($j.steps | ForEach-Object { "{0:N3}" -f $_ }) -join ', ' } else { $st = 'NO JSON (failed)' }
    State "$($r[0]) done, steps: $st"
}

foreach ($c in @(@('f1M', '32', $small), @('f2M', '64', $small), @('f4M', '128', $small), @('f9M', '32', $mid), @('f18M', '64', $mid))) {
    & $py "$S\profile_polya.py" --batches $c[1] --sizes $c[2] --steps 6 --fix both --out "$R\$($c[0]).json" | Out-Null
}
State "fixed small sizes done"

foreach ($c in @(@('cpu18M', '64', $mid), @('cpu36M', '64', $big), @('cpu72M', '128', $big))) {
    & $py "$S\cpu_bench.py" --batches $c[1] --sizes $c[2] --steps 5 --out "$R\$($c[0]).json" | Out-Null
    $ok = Test-Path "$R\$($c[0]).json"
    if ($ok) { $j = Get-Content "$R\$($c[0]).json" -Raw | ConvertFrom-Json; $st = ($j.steps | ForEach-Object { "{0:N2}" -f $_ }) -join ', ' } else { $st = 'NO JSON (failed)' }
    State "$($c[0]) done, steps: $st"
}
"## SUITE COMPLETE"
