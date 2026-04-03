@echo off
cd /d "%~dp0"
call conda activate rag
if errorlevel 1 (
    echo ERROR: Could not activate conda env "rag". Make sure you ran: conda create -n rag python=3.12
    pause
    exit /b 1
)
python start.py
pause
