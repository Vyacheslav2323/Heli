$p = Get-Process QGroundControl -ErrorAction SilentlyContinue
if (-not $p) { Write-Host "QGC not running"; exit 0 }
Write-Host "pid=$($p.Id)"
Get-NetUDPEndpoint -OwningProcess $p.Id -ErrorAction SilentlyContinue | Format-Table LocalAddress,LocalPort -AutoSize
Get-NetTCPConnection -OwningProcess $p.Id -ErrorAction SilentlyContinue | Format-Table LocalAddress,LocalPort,State -AutoSize
