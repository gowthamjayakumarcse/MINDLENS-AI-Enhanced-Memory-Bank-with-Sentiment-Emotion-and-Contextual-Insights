@echo off
REM MindLens Application Launcher
REM This launches the PowerShell startup script

powershell.exe -ExecutionPolicy Bypass -File "%~dp0start_app_with_ffmpeg.ps1"
