@echo off
cd /d "%~dp0.."
title HAR Flask - NO CERRAR
echo Abre http://127.0.0.1:5000
.venv\Scripts\pip.exe install flask -q
.venv\Scripts\python.exe app\flask_app.py
pause
