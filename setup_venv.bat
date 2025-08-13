@echo off
echo ========================================
echo EstAlan Development Environment Setup
echo ========================================
echo.

REM Check if uv is installed
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: uv is not installed or not in PATH
    echo Please install uv first: https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)

echo Checking uv installation...
uv --version
echo.

REM Create virtual environment
echo Creating virtual environment...
uv venv
if %errorlevel% neq 0 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install dependencies
echo Installing dependencies...
uv pip install -e src
if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To activate the virtual environment in the future:
echo   .venv\Scripts\activate.bat
echo.
echo To deactivate:
echo   deactivate
echo.
pause
