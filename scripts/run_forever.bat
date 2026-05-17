@echo off
REM Nailong Elite — auto-restart wrapper for Windows.
REM Restarts the agent on any crash with a 30-second cooldown.
REM Usage:
REM   scripts\run_forever.bat
REM   start /b scripts\run_forever.bat > trader.log 2>&1

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0\.."

if not defined NAILONG_SLUG set NAILONG_SLUG=eval_nailonguic
if not defined NAILONG_COOLDOWN set NAILONG_COOLDOWN=30
if not defined NAILONG_LOG set NAILONG_LOG=trader.log

echo [%date% %time%] Starting Nailong forever-loop slug=%NAILONG_SLUG%
echo [%date% %time%] Log: %NAILONG_LOG%  Cooldown: %NAILONG_COOLDOWN%s

set ATTEMPT=0
:loop
set /a ATTEMPT+=1
echo [%date% %time%] ==== Attempt %ATTEMPT%: starting agent ====
python -m agent.run --slug %NAILONG_SLUG% >> %NAILONG_LOG% 2>&1
echo [%date% %time%] Agent exited; cooldown %NAILONG_COOLDOWN%s before restart
timeout /t %NAILONG_COOLDOWN% /nobreak > nul
goto loop
