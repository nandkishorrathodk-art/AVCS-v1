@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set SCRIPT_DIR=%~dp0
set VENV_PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment not found at %VENV_PYTHON%
    exit /b 1
)

"%VENV_PYTHON%" -m avcs.ace.run %*
exit /b %ERRORLEVEL%

