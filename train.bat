@echo off
echo Starting training process...
call .\venv\Scripts\activate
python train_model.py
if %errorlevel% neq 0 (
    echo Training failed
    pause
    exit /b %errorlevel%
)
echo Training completed successfully
pause