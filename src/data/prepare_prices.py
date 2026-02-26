import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# CONFIG – ULTRA LENIENT
# =====================================================
RAW_PRICE_DIR = "data_raw/stocks"
OUTPUT_DIR = "data_processed/prices"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRICE_COL_CANDIDATES = ["adj_close", "adjusted_close", "adjclose", "close"]

MAX_LOG_RETURN = np.log(11)     
MAX_MISSING_PCT = 0.95
MAX_FFILL_DAYS = 30
MAX_BFILL_DAYS = 3              
MIN_DATA_DAYS = 60

# =====================================================
# FIX ONLY OBVIOUS DATA CORRUPTION
# =====================================================
def fix_obvious_errors(df, symbol):
    """
    Fix ONLY undeniable data errors.
    Do NOT smooth volatility or truncate tails.
    """
    df = df.sort_values("date").copy()

    # --- 1. Negative prices (impossible)
    neg_mask = df["close"] < 0
    if neg_mask.any():
        df.loc[neg_mask, "close"] = df.loc[neg_mask, "close"].abs()
        print(f" {symbol}: fixed {neg_mask.sum()} negative prices")

    # --- 2. Zero prices (treated as missing)
    zero_mask = df["close"] == 0
    if zero_mask.any():
        df.loc[zero_mask, "close"] = np.nan
        print(f" {symbol}: replaced {zero_mask.sum()} zero prices with NaN")

    # --- 3. Detect extreme corruption (decimal shift ~10x, 100x, 1000x)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    insane_mask = df["log_ret"].abs() > MAX_LOG_RETURN

    if insane_mask.any():
        print(f" {symbol}: {insane_mask.sum()} extreme jumps detected")

        for idx in df[insane_mask].index:
            i = df.index.get_loc(idx)
            if i == 0:
                continue

            prev_idx = df.index[i - 1]
            prev_p = df.loc[prev_idx, "close"]
            curr_p = df.loc[idx, "close"]
            ratio = curr_p / prev_p

            # curr quá cao (ratio ~ 10, 100, 1000) → chia curr
            if np.isclose(ratio, 10, rtol=0.2):
                df.loc[idx, "close"] = curr_p / 10
            elif np.isclose(ratio, 100, rtol=0.2):
                df.loc[idx, "close"] = curr_p / 100
            elif np.isclose(ratio, 1000, rtol=0.2):
                df.loc[idx, "close"] = curr_p / 1000
            # prev quá cao (ratio ~ 0.1, 0.01, 0.001) → chia prev
            elif ratio < 0.02 and ratio > 0:
                if np.isclose(ratio, 0.1, rtol=0.3):
                    df.loc[prev_idx, "close"] = prev_p / 10
                elif np.isclose(ratio, 0.01, rtol=0.3):
                    df.loc[prev_idx, "close"] = prev_p / 100
                elif np.isclose(ratio, 0.001, rtol=0.3):
                    df.loc[prev_idx, "close"] = prev_p / 1000
            # curr quá thấp (1/ratio ~ 10, 100, 1000) → nhân curr
            elif np.isclose(1 / ratio, 10, rtol=0.2):
                df.loc[idx, "close"] = curr_p * 10
            elif np.isclose(1 / ratio, 100, rtol=0.2):
                df.loc[idx, "close"] = curr_p * 100
            elif np.isclose(1 / ratio, 1000, rtol=0.2):
                df.loc[idx, "close"] = curr_p * 1000

        # recompute và lặp tối đa 2 lần (fix prev có thể tạo jump mới)
        for _ in range(2):
            df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
            insane_mask = df["log_ret"].abs() > MAX_LOG_RETURN
            if not insane_mask.any():
                break
            for idx in df[insane_mask].index:
                i = df.index.get_loc(idx)
                if i == 0:
                    continue
                prev_idx = df.index[i - 1]
                prev_p = df.loc[prev_idx, "close"]
                curr_p = df.loc[idx, "close"]
                ratio = curr_p / prev_p
                if np.isclose(ratio, 1000, rtol=0.2):
                    df.loc[idx, "close"] = curr_p / 1000
                elif np.isclose(ratio, 100, rtol=0.2):
                    df.loc[idx, "close"] = curr_p / 100
                elif np.isclose(ratio, 10, rtol=0.2):
                    df.loc[idx, "close"] = curr_p / 10
                elif ratio < 0.02 and ratio > 0:
                    if ratio < 0.002:
                        df.loc[prev_idx, "close"] = prev_p / 1000
                    elif ratio < 0.02:
                        df.loc[prev_idx, "close"] = prev_p / 100
                elif ratio > 50:
                    if ratio > 500:
                        df.loc[idx, "close"] = curr_p / 1000
                    else:
                        df.loc[idx, "close"] = curr_p / 100

        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        insane_mask = df["log_ret"].abs() > MAX_LOG_RETURN
        if insane_mask.any():
            print(f" {symbol}: dropped {insane_mask.sum()} unrecoverable points")
            df = df.loc[~insane_mask]

    return df.drop(columns=["log_ret"], errors="ignore")


# =====================================================
# MINIMAL SYMBOL CLEANING
# =====================================================
def minimal_clean_symbol(df, symbol):
    if len(df) == 0:
        return None, "empty"

    df = fix_obvious_errors(df, symbol)

    if len(df) < MIN_DATA_DAYS:
        return None, "too_short"

    return df, "ok"


# =====================================================
# MAIN PIPELINE
# =====================================================
def main():
    print("\n PRICE PREPARATION\n")

    all_frames = []
    all_dates = set()
    stats = {"ok": [], "too_short": [], "empty": [], "error": []}

    for file in sorted(os.listdir(RAW_PRICE_DIR)):
        if not file.endswith(".csv"):
            continue

        symbol = file.replace(".csv", "").upper()
        try:
            df = pd.read_csv(os.path.join(RAW_PRICE_DIR, file))
            df.columns = [c.lower().strip() for c in df.columns]

            price_col = next((c for c in PRICE_COL_CANDIDATES if c in df.columns), None)
            if price_col is None or "date" not in df.columns:
                stats["error"].append(symbol)
                continue

            df = df[["date", price_col]].rename(columns={price_col: "close"})
            df["symbol"] = symbol
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna().sort_values("date")

            cleaned, status = minimal_clean_symbol(df, symbol)
            if cleaned is None:
                stats[status].append(symbol)
                continue

            all_frames.append(cleaned)
            all_dates.update(cleaned["date"].unique())
            stats["ok"].append(symbol)

        except Exception as e:
            print(f" {symbol}: {e}")
            stats["error"].append(symbol)

    # --- common calendar
    panel = []

    for df in all_frames:
        symbol = df["symbol"].iloc[0]

        df = df.sort_values("date").copy()

        missing_pct = df["close"].isna().mean()
        if missing_pct > MAX_MISSING_PCT:
            stats["too_short"].append(symbol)
            continue

        df["close"] = df["close"].ffill(limit=MAX_FFILL_DAYS)
        df["close"] = df["close"].bfill(limit=MAX_BFILL_DAYS)
        df = df.dropna()

        if len(df) < MIN_DATA_DAYS:
            continue

        panel.append(df.reset_index(drop=True))

    prices = pd.concat(panel, ignore_index=True).sort_values(["symbol", "date"])
    prices.to_csv(os.path.join(OUTPUT_DIR, "prices.csv"), index=False)

    print("\n FINAL SUMMARY")
    print(f"   Symbols kept: {prices['symbol'].nunique()}")
    print(f"   Date range : {prices['date'].min():%Y-%m-%d} to {prices['date'].max():%Y-%m-%d}")
    print("   Philosophy preserved: NO smoothing, NO truncation, FULL tails")

if __name__ == "__main__":
    main()
