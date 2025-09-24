@echo off
REM Batch file to run FarmFlow model training script automatically
REM Update the following paths as needed

set PYTHON_EXE=C:\Users\RUDRA\AppData\Local\Programs\Python\Python311\python.exe
set SCRIPT_PATH=e:\OneDrive\Desktop\Farm_flow\New_PP\train_farmflow_model.py
set WORK_DIR=e:\OneDrive\Desktop\Farm_flow\New_PP

cd /d %WORK_DIR%
"%PYTHON_EXE%" "%SCRIPT_PATH%"
