@echo off
REM ===== DPMS Agentic AI - start the app =====
cd /d %~dp0
set "PY=D:\medrag_transfer\.venv\Scripts\python.exe"
if not exist "%PY%" (echo Python venv not found at %PY% & pause & exit /b 1)

REM open the browser shortly after the server starts
start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8000"

echo Starting DPMS Agentic AI at http://127.0.0.1:8000  (press Ctrl+C to stop)
"%PY%" -m uvicorn app:app
