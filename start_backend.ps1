# Khởi động Backend API (port 8000)
# Chạy từ thư mục gốc dự án: .\start_backend.ps1

Set-Location $PSScriptRoot
python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
