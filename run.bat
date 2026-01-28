@echo off
echo ============================================
echo    MATAN Analysis Tool
echo    Analiza uspjesnosti MA1 i MA2
echo ============================================
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo Please run: setup.bat
    pause
    exit /b 1
)

echo Starting analysis...
echo.

.venv\Scripts\python.exe main.py

echo.
echo ============================================
echo Analysis complete! Check the output folder.
echo ============================================
pause
