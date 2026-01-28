@echo off
echo ============================================
echo    MATAN Analysis Tool - Setup
echo ============================================
echo.

cd /d "%~dp0"

echo [1/3] Creating virtual environment...
python -m venv .venv

echo [2/3] Installing dependencies...
.venv\Scripts\pip.exe install -r requirements.txt -q

echo [3/3] Setup complete!
echo.
echo ============================================
echo Now you can run the analysis with: run.bat
echo ============================================
pause
