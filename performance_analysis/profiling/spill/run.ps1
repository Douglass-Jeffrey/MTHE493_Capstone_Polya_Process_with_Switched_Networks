param([string]$Name, [string[]]$PyArgs)
$S = $PSScriptRoot
$R = if ($env:POLYA_WORK) { Join-Path $env:POLYA_WORK 'reports\spill' } else { Join-Path $S '..\..\..\work\reports\spill' }
New-Item -ItemType Directory -Force $R | Out-Null
$nsys = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.2\target-windows-x64\nsys.exe"
$py = "python"
$t = Get-Date
& $nsys profile -t cuda,nvtx,cusparse -s none --cpuctxsw=none --cuda-memory-usage=true --force-overwrite=true `
    -o "$R\$Name" $py "$S\profile_polya.py" @PyArgs --out "$R\$Name.json" | Out-Null
"profiled in {0:N0}s" -f ((Get-Date) - $t).TotalSeconds
& $nsys export -t sqlite --force-overwrite=true -o "$R\$Name.sqlite" "$R\$Name.nsys-rep" | Out-Null
& $py -I "$S\analyze.py" "$R\$Name.sqlite" 8
$j = Get-Content "$R\$Name.json" -Raw | ConvertFrom-Json
"nodes={0:N0} edges={1:N0} growth={2:N1}s csr={3:N1}s" -f $j.nodes, $j.edges, $j.growth_s, $j.csr_s
$j.snapshots.PSObject.Properties | ForEach-Object { "{0,-13} free={1,6:N0} MB  pool used={2,6:N0} MB  held={3,6:N0} MB" -f $_.Name, $_.Value.gpu_free_mb, $_.Value.pool_used_mb, $_.Value.pool_total_mb }
"steps: " + (($j.steps | ForEach-Object { "{0:N3}" -f $_ }) -join ", ")
foreach ($p in $j.probe.PSObject.Properties) {
    "probe $($p.Name)"
    $p.Value.PSObject.Properties | ForEach-Object {
        if ($_.Value.mb) { "  {0,-15} {1,8:N0} MB  {2}" -f $_.Name, $_.Value.mb, (($_.Value.gbps | ForEach-Object { "{0,6:N1}" -f $_ }) -join ' /') }
        else { "  {0,-15} {1}" -f $_.Name, (($_.Value | ForEach-Object { "{0:N3}s" -f $_ }) -join ', ') }
    }
}
