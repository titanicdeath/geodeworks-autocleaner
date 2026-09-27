@echo off
cd /d "%~dp0"
py -3 -m venv .venv
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Setup complete. Open launch.cmd next.
pause
exit /b 0
:failed
echo Setup failed. Install 64-bit Python 3.11 or 3.12 with the Python launcher, then retry.
pause
exit /b 1
