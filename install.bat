@echo off
echo Setting up training environment...
python setup_environment.py
if %errorlevel% neq 0 (
    echo Environment setup failed
    pause
    exit /b %errorlevel%
)
echo Environment setup completed successfully
pause