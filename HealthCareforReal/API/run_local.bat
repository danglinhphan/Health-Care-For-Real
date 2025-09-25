@echo off
REM Windows batch script to run Qwen LoRA API locally

echo Installing dependencies...
pip install -r requirements.txt

echo Starting Qwen LoRA API server...
python run_api.py --host 127.0.0.1 --port 8000

pause