@echo off
setlocal

cd /d "%~dp0backend"
if errorlevel 1 exit /b 1

if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 exit /b %errorlevel%
)

".venv\Scripts\python.exe" -m pip install --no-input -r requirements.txt > pip-install.log 2>&1
if errorlevel 1 (
  echo Backend dependency installation failed. See backend\pip-install.log.
  type pip-install.log
  exit /b 1
)

echo Backend dependencies installed successfully.