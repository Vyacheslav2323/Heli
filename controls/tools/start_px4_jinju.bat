@echo off
REM Restart PX4 SITL in WSL with Jinju home (동부로169번길 12)
wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/yj.park/Repo/helicopter/tools/start_px4_jinju.sh
pause
