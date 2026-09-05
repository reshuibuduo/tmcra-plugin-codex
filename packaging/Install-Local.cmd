@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-TMCRA.ps1" -LocalMemory
if errorlevel 1 pause
