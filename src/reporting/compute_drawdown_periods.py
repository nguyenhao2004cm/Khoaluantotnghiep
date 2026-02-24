import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]

PORT_FILE = PROJECT_DIR / "data_processed/powerbi/portfolio_timeseries.csv"
OUT_DIR = PROJECT_DIR / "data_processed/reporting"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "drawdown_periods.csv"

df = pd.read_csv(PORT_FILE, parse_dates=["date"])

drawdowns = []
in_dd = False
start = None
min_dd = 0

for _, row in df.iterrows():
    if row["drawdown"] < 0 and not in_dd:
        in_dd = True
        start = row["date"]
        min_dd = row["drawdown"]

    elif row["drawdown"] < 0 and in_dd:
        min_dd = min(min_dd, row["drawdown"])

    elif row["drawdown"] == 0 and in_dd:
        end = row["date"]
        length = (end - start).days
        drawdowns.append([start, end, length, min_dd])
        in_dd = False

dd_df = pd.DataFrame(
    drawdowns,
    columns=[
        "Ngày bắt đầu",
        "Ngày kết thúc",
        "Thời gian sụt giảm (ngày)",
        "Mức sụt giảm tối đa"
    ]
)

# 🔹 Sắp xếp & lấy top 10 drawdown lớn nhất
dd_df = dd_df.sort_values("Mức sụt giảm tối đa").head(10)

# 🔹 Làm tròn mức sụt giảm
dd_df["Mức sụt giảm tối đa"] = dd_df["Mức sụt giảm tối đa"].round(3)

dd_df.to_csv(OUT_FILE, index=False)