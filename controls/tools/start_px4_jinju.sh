#!/usr/bin/env bash
# PX4 SITL home: Jinju (dongbu-ro 169beon-gil 12)
# WGS84: 35.165820, 128.126647
set -euo pipefail
export PX4_HOME_LAT=35.165820
export PX4_HOME_LON=128.126647
export PX4_HOME_ALT=40
export DISPLAY="${DISPLAY:-:0}"

# Stop a previous SITL instance if still running
pkill -x px4 2>/dev/null || true
pkill -f 'gz sim' 2>/dev/null || true
sleep 1
rm -f /tmp/px4_* 2>/dev/null || true

cd "${PX4_DIR:-$HOME/PX4-Autopilot}"
echo "Starting SITL at Jinju home: $PX4_HOME_LAT, $PX4_HOME_LON (alt ${PX4_HOME_ALT}m)"
exec make px4_sitl gz_x500
