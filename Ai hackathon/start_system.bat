@echo off
setlocal
cd /d "%~dp0"

:MENU
cls
echo ===========================================
echo      ENGINE HEALTH MONITORING SYSTEM
echo ===========================================
echo.
echo 1. Start with Training
echo 2. Start without Training (Testing the test data using the existing model)
echo 3. View current finished CSV
echo 4. Open Dashboard (UI Only)
echo 5. Exit
echo.
set /p choice=Enter your choice (1-5): 

if "%choice%"=="1" goto TRAINING
if "%choice%"=="2" goto TESTING
if "%choice%"=="3" goto VIEW_CSV
if "%choice%"=="4" goto START_DASHBOARD
if "%choice%"=="5" goto EXIT
goto MENU

:TRAINING
echo.
echo [1/3] Configuring Environment...
echo [1/3] Configuring Environment...
echo Installing Python dependencies...
pip install -r requirements.txt
echo [2/3] Training Model...
python train.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Training failed. Please check the logs.
    pause
    goto MENU
)
goto START_DASHBOARD

:TESTING
echo.
echo [1/3] Configuring Environment...
echo [1/3] Configuring Environment...
echo Installing Python dependencies...
pip install -r requirements.txt
echo [2/3] Testing Existing Model...
python train.py --test_only
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Testing failed. Please check the logs.
    pause
    goto MENU
)
goto START_DASHBOARD

:START_DASHBOARD
echo.
echo [3/3] Starting Dashboard...
cd rul-dashboard

if not exist node_modules (
    echo Installing dependencies...
    call npm install
)

echo.
echo Opening Dashboard in browser...
start http://localhost:5173

echo Launching Server...
call npm run dev
pause
goto MENU

:VIEW_CSV
if exist final_submission.csv (
    echo Opening CSV...
    start final_submission.csv
) else (
    echo.
    echo [ERROR] final_submission.csv not found. Please run training/testing first.
    pause
)
goto MENU

:EXIT
exit /b
