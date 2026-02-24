# Config cho chạy tối ưu từ web: assets, start_date, end_date, risk_appetite

import json
from pathlib import Path
from typing import Optional

PROJECT_DIR = Path(__file__).resolve().parents[2]
WEB_RUN_CONFIG_PATH = PROJECT_DIR / "data_processed" / "web_run_config.json"


def load_web_run_config() -> Optional[dict]:
    """Đọc config do API ghi khi user gửi form. None nếu không có."""
    if not WEB_RUN_CONFIG_PATH.exists():
        return None
    try:
        with open(WEB_RUN_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_web_run_config(
    assets: list,
    start_date: str,
    end_date: str,
    risk_appetite: str,
    initial_capital: float = 10000.0,
) -> None:
    """API gọi trước khi chạy pipeline."""
    WEB_RUN_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WEB_RUN_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "assets": assets,
            "start_date": start_date,
            "end_date": end_date,
            "risk_appetite": risk_appetite,
            "initial_capital": initial_capital,
        }, f, ensure_ascii=False, indent=2)
