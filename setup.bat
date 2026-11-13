@echo off
REM ===== DPMS Agentic AI - one-time setup on a Windows machine =====
REM Run this once. It installs packages, generates data, and trains the model.
REM Assumes: conda env "agents" exists, and Ollama is installed.

cd /d %~dp0
echo Activating conda env "agents"...
call conda activate agents || (echo Could not activate conda env "agents". & pause & exit /b 1)

echo Installing Python packages...
pip install -r requirements.txt

echo Generating synthetic appointment history...
python ml\generate_data.py

echo Training the no-show model...
python ml\train.py

echo.
echo ============================================================
echo Setup complete.
echo Make sure the local model is pulled (once):
echo     ollama pull qwen3:4b
echo Then start the app any time with:  run.bat
echo ============================================================
pause
