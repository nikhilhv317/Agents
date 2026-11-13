@echo off
REM ===== DPMS Agentic AI - start the app =====
cd /d %~dp0
call conda activate agents || (echo Could not activate conda env "agents". & pause & exit /b 1)

REM open the browser shortly after the server starts
start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8000"

echo Starting DPMS Agentic AI at http://127.0.0.1:8000  (press Ctrl+C to stop)
uvicorn app:app
