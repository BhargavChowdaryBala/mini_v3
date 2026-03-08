@echo off
TITLE Bus Monitoring System - Launcher
COLOR 0B

echo ======================================================
echo    BUS MONITORING SYSTEM - AUTO-START BOOTSTRAP
echo ======================================================
echo.

:: Get the directory of the batch file
SET ROOT_DIR=%~dp0
cd /d %ROOT_DIR%

:: 1. Start Flask Backend
echo [1/2] Launching AI Backend (Flask)...
start "Bus_Backend" cmd /c "python backend/main.py"

timeout /t 5 /nobreak > nul

:: 2. Start React Frontend
echo [2/2] Launching UI Dashboard (Vite)...
cd frontend
start "Bus_Frontend" cmd /c "npm run dev"

echo.
echo ======================================================
echo    SYSTEM INITIALIZED SUCCESSFULLY!
echo    Backend: http://localhost:5000
echo    Frontend: http://localhost:5173
echo ======================================================
echo.
echo Keep these windows open while the system is in use.
timeout /t 10
