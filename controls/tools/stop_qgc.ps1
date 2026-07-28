$p = Get-Process QGroundControl -ErrorAction SilentlyContinue
if ($p) {
    Write-Host "Stopping QGC pid=$($p.Id)"
    Stop-Process -Id $p.Id -Force
    Start-Sleep -Seconds 2
}
Write-Host "QGC stopped"
