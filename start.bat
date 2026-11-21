@echo off
rem TextMind Windows launcher: locate the project root from this script and use the bundled .venv.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv\Scripts\python.exe not found.
  echo Please install Python 3.11 dependencies first:
  echo   py -3.11 scripts\install.py
  exit /b 1
)

".venv\Scripts\python.exe" scripts\start.py %*
if errorlevel 1 exit /b %errorlevel%
endlocal
