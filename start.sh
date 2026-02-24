#!/bin/bash
# Chạy backend API. PORT mặc định 8000 nếu không set.
PORT=${PORT:-8000}
uvicorn src.api.app:app --host 0.0.0.0 --port $PORT
