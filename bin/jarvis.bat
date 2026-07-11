@echo off
REM ===================================================================
REM  Jarvis launcher - lets you start Jarvis by typing "jarvis" in any
REM  terminal, from any directory.
REM
REM  This file lives in <project>\bin. %~dp0 expands to that folder (with
REM  a trailing backslash), so "%~dp0..\run.py" always points at run.py in
REM  the project root - the launcher is fully portable: move the project
REM  and it still works, as long as this bin folder stays on your PATH.
REM
REM  Prefers the Windows "py" launcher (never resolves to the Microsoft
REM  Store stub); falls back to "python" if py is not installed.
REM ===================================================================
setlocal
where py >nul 2>nul
if %errorlevel%==0 (
  py "%~dp0..\run.py" %*
) else (
  python "%~dp0..\run.py" %*
)
