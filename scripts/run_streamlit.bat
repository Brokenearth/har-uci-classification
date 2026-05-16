@echo off
cd /d "%~dp0.."
title HAR Streamlit
.venv\Scripts\python.exe scripts\run_streamlit.py
