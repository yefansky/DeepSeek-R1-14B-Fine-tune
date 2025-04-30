@echo off
echo Starting GGUF conversion...
call .\venv\Scripts\activate
python convert_to_gguf.py
if %errorlevel% neq 0 (
    echo Conversion failed
    pause
    exit /b %errorlevel%
)
echo Conversion completed successfully
pause
