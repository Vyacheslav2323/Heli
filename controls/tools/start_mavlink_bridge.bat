@echo off
REM Bridge PX4 SITL (WSL) <-> QGroundControl (Windows)
REM 1) Start SITL in WSL:  cd ~/PX4-Autopilot && make px4_sitl gz_x500
REM 2) Start QGroundControl on Windows
REM 3) Run this script and leave it open

cd /d "%~dp0"
echo Starting MAVLink WSL bridge...
python mavlink_wsl_bridge.py
pause
