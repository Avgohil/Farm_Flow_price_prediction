@echo off
REM Portable pipeline runner for FarmFlow
REM Backed up original to run_farmflow_daily.bat.bak

REM Allow overriding python executable by setting PYTHON_EXE environment var
if "%PYTHON_EXE%"=="" (
	set PYTHON_EXE=python
)

REM Use batch file directory as working directory
set WORK_DIR=%~dp0
cd /d "%WORK_DIR%"

echo ===== FarmFlow daily pipeline starting =====
echo Working directory: %WORK_DIR%
echo Using Python: %PYTHON_EXE%

REM 1) Fetch (fixed fetcher)
if exist "%WORK_DIR%fetch_agmarknet_daily_fixed.py" (
	echo Running fetch_agmarknet_daily_fixed.py
	"%PYTHON_EXE%" "%WORK_DIR%fetch_agmarknet_daily_fixed.py"
) else (
	echo fetch_agmarknet_daily_fixed.py not found, falling back to fetch_agmarknet_daily.py if present
	if exist "%WORK_DIR%fetch_agmarknet_daily.py" (
		"%PYTHON_EXE%" "%WORK_DIR%fetch_agmarknet_daily.py"
	) else (
		echo No fetch script found; skipping fetch step
	)
)

REM 2) Combine daily files
if exist "%WORK_DIR%combine_filter_daily.py" (
	echo Running combine_filter_daily.py
	"%PYTHON_EXE%" "%WORK_DIR%combine_filter_daily.py"
) else (
	echo combine_filter_daily.py not found; skipping
)

REM 3) Update master
if exist "%WORK_DIR%update_master.py" (
	echo Running update_master.py
	"%PYTHON_EXE%" "%WORK_DIR%update_master.py"
) else (
	echo update_master.py not found; skipping
)

REM 4) Train model
if exist "%WORK_DIR%train_farmflow_model.py" (
	echo Running train_farmflow_model.py
	"%PYTHON_EXE%" "%WORK_DIR%train_farmflow_model.py"
) else (
	echo train_farmflow_model.py not found; skipping
)

echo ===== FarmFlow daily pipeline finished =====
