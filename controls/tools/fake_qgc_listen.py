#!/usr/bin/env python3
"""Minimal stand-in for QGC: listen UDP 14550, print MAVLink magic, reply to sender."""
import socket
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("0.0.0.0", 14550))
sock.settimeout(1.0)
print("listening 0.0.0.0:14550", flush=True)
count = 0
t0 = time.time()
while time.time() - t0 < 10:
    try:
        data, addr = sock.recvfrom(65535)
    except socket.timeout:
        continue
    count += 1
    if count <= 5 or count % 100 == 0:
        print(f"rx#{count} from {addr} len={len(data)} magic=0x{data[0]:02x}", flush=True)
    # echo a tiny payload back so bridge sees from_qgc
    sock.sendto(b"\xfd\x00\x00\x00\x00\xff\xbe\x00\x00\x00\x00", addr)
print(f"done, total_rx={count}", flush=True)
sock.close()
