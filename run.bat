@echo off
title LGAF Medical AI Suite
echo =====================================================================
echo           LGAF Medical AI - Local Deployment Launcher
echo =====================================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

:: Virtual environment check/setup
if not exist ".venv" (
    echo [INFO] Creating Python virtual environment in .venv...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
)

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [INFO] Installing/verifying dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo =====================================================================
echo Starting LGAF Streamlit Diagnostic Application...
echo Browser will automatically open at http://localhost:8501
echo Press Ctrl+C in this window to stop the server.
echo =====================================================================
echo.

streamlit run app.py

pause
