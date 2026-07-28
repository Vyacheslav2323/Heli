$p = Get-Process QGroundControl -ErrorAction SilentlyContinue
if ($p) {
    $path = $p.Path
    Write-Host "QGC path: $path"
    Stop-Process -Id $p.Id -Force
    Start-Sleep -Seconds 2
    Start-Process $path
    Write-Host "QGC restarted"
} else {
    Write-Host "QGC not running"
}
