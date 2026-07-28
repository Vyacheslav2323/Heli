$wslIp = (wsl -d Ubuntu-24.04 -- hostname -I).Trim().Split(' ')[0]
Write-Host "WSL IP: $wslIp"

$udp = New-Object System.Net.Sockets.UdpClient 14552
$udp.Client.ReceiveTimeout = 3000

$bytes = [Text.Encoding]::ASCII.GetBytes('punch')
$udp.Send($bytes, $bytes.Length, $wslIp, 18570) | Out-Null
Write-Host "punched to ${wslIp}:18570"

wsl -d Ubuntu-24.04 -- python3 -c "import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.sendto(b'REPLY_TEST', ('172.22.176.1',14552)); print('wsl sent')"

try {
  $ep = New-Object System.Net.IPEndPoint([Net.IPAddress]::Any, 0)
  $resp = $udp.Receive([ref]$ep)
  Write-Host "SUCCESS got reply from $ep len=$($resp.Length)"
} catch {
  Write-Host "FAIL no reply: $_"
} finally {
  $udp.Close()
}
