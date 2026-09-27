@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe goto install
echo Creating .venv ...
py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv 2>nul || py -3 -m venv .venv
if errorlevel 1 goto failed
:install
.venv\Scripts\python.exe -m pip install --upgrade -r requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m geoclean selftest
echo.
echo Setup complete. Run launch.cmd next.
pause
exit /b 0
:failed
echo Setup failed. Install 64-bit Python 3.11 or 3.12 with the py launcher, then retry.
pause
exit /b 1
