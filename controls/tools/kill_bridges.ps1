Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -match 'mavlink_wsl_bridge|mavlink_wsl_side|fake_qgc'
} | ForEach-Object {
    Write-Host "Killing $($_.ProcessId): $($_.CommandLine)"
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}
# Also kill any python holding 15570
foreach ($line in (netstat -ano | Select-String '15570')) {
    if ($line -match '\s(\d+)\s*$') {
        $pid = [int]$Matches[1]
        Write-Host "Killing PID on 15570: $pid"
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Seconds 2
Write-Host "Remaining 15570:"
netstat -ano | findstr 15570
Write-Host "Done"
