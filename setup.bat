@echo off
REM ===== DPMS Agentic AI - one-time setup on a Windows machine =====
REM Run this once. It installs packages, generates data, and trains the model.
REM Assumes: the Python venv below exists (it already has CUDA torch), and Ollama is installed.
REM WARNING: generate_data.py rebuilds dpms.db from scratch.

cd /d %~dp0
set "PY=D:\medrag_transfer\.venv\Scripts\python.exe"
if not exist "%PY%" (echo Python venv not found at %PY% & pause & exit /b 1)

echo Installing Python packages (torch pinned to the installed CUDA build)...
"%PY%" -m pip install torchvision "torch==2.15.0+cu132" --index-url https://download.pytorch.org/whl/cu132
"%PY%" -m pip install -r requirements.txt "torch==2.15.0+cu132" --extra-index-url https://download.pytorch.org/whl/cu132

echo Generating synthetic appointment history...
"%PY%" ml\generate_data.py

echo Training the no-show model...
"%PY%" ml\train.py

echo.
echo ============================================================
echo Setup complete.
echo Make sure the local model is pulled (once):
echo     ollama pull qwen3:1.7b
echo Then start the app any time with:  run.bat
echo ============================================================
pause
