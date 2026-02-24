import os
import pandas as pd

# ==============================
# CONFIG
# ==============================
INPUT_PATH = "data_processed/backtest/portfolio_returns.csv"
OUT_DIR = "data_processed/benchmark"
OUT_PATH = os.path.join(OUT_DIR, "equal_weight_portfolio.csv")

os.makedirs(OUT_DIR, exist_ok=True)


def run():
    # ==============================
    # LOAD BACKTEST RESULT
    # ==============================
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])

    # ==============================
    # EXTRACT BENCHMARK
    # ==============================
    benchmark = df[[
        "date",
        "equal_return",
        "equal_cum"
    ]].rename(columns={
        "equal_return": "benchmark_return",
        "equal_cum": "benchmark_cum"
    })

    # ==============================
    # SAVE
    # ==============================
    benchmark.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f" Benchmark saved to {OUT_PATH}")
    print(benchmark.head())


if __name__ == "__main__":
    run()
