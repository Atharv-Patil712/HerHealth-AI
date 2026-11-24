@echo off
start "HerHealth Backend" /D "%~dp0backend" cmd /k ".venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000"
start "HerHealth Frontend" /D "%~dp0" cmd /k "npm run dev"