@echo off
echo ========================================
echo Cleaning EstAlan Virtual Environment
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo Virtual environment not found. Nothing to clean.
    pause
    exit /b 0
)

echo WARNING: This will remove the entire virtual environment!
echo All installed packages will be deleted.
echo.
set /p confirm="Are you sure you want to continue? (y/N): "

if /i not "%confirm%"=="y" (
    echo Operation cancelled.
    pause
    exit /b 0
)

echo.
echo Removing virtual environment...
rmdir /s /q .venv
if %errorlevel% neq 0 (
    echo Error: Failed to remove virtual environment
    pause
    exit /b 1
)

echo Virtual environment removed successfully!
echo.
echo To recreate the environment, run: setup_venv.bat
echo.
pause
