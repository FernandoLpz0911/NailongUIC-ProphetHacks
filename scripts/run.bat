@echo off
REM Prophet Hacks 2026 - Trading Track agent launcher (Windows).
REM
REM Usage:
REM   scripts\run.bat                  -- 14-day eval defaults
REM   scripts\run.bat --dry            -- smoke-test pipeline only
REM   scripts\run.bat --slug v02       -- override slug

setlocal
cd /d "%~dp0\.."

if not exist ".env" (
  echo ERROR: .env not found. Copy .env.template to .env and fill in keys. 1>&2
  exit /b 1
)

python -m agent.run %*
