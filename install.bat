@echo off
REM Claude Memory Palace - Windows Installer
REM This is a shim that calls the PowerShell installer

echo ============================================
echo  Claude Memory Palace - Installer
echo ============================================
echo.

REM Check if PowerShell is available
where powershell >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: PowerShell is not available on this system.
    echo Please install PowerShell or run install.ps1 manually.
    pause
    exit /b 1
)

REM Run the PowerShell installer with execution policy bypass
powershell -ExecutionPolicy Bypass -File "%~dp0install.ps1"

echo.
echo ============================================
echo  Installation complete. Press any key to exit.
echo ============================================
pause
