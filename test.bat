@echo off
echo Starting training process...
call .\venv\Scripts\activate
python test_functioncall.py
if %errorlevel% neq 0 (
    echo test failed
    pause
    exit /b %errorlevel%
)
echo test completed successfully
pause