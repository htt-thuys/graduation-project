@echo off
REM Quick activation script for Command Prompt (cmd.exe)
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo.
echo ========================================
echo Virtual environment activated!
echo Run: python test_local.py
echo Or:  deactivate  (to exit venv)
echo ========================================
echo.
