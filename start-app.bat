@echo off
start "HerHealth Backend" cmd /k "cd backend && .\.venv\Scripts\activate && uvicorn app.main:app"
start "HerHealth Frontend" cmd /k "npm run dev"