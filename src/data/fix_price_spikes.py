# =====================================================
# FIX PRICE SPIKES (decimal shift 10x, 100x, 1000x)
# Chạy trực tiếp trên prices.csv đã có
# =====================================================

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
PRICE_FILE = PROJECT_DIR / "data_processed" / "prices" / "prices.csv"


def fix_decimal_shift(df):
    """Sửa lỗi dịch chuyển thập phân (176000->176, 96700->96.7)."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    total_fixed = 0

    for _ in range(15):
        df["prev_close"] = df.groupby("symbol")["close"].shift(1)
        df["next_close"] = df.groupby("symbol")["close"].shift(-1)
        df["ratio"] = df["close"] / df["prev_close"]
        df["ratio_next"] = df["close"] / df["next_close"]
        # Mở rộng: ratio > 20 hoặc < 0.05
        mask = (df["ratio"] > 20) | ((df["ratio"] < 0.05) & (df["ratio"] > 0))
        # Hàng đầu tiên của mỗi mã: so sánh với next
        first_row_mask = df.groupby("symbol").cumcount() == 0
        mask_first = first_row_mask & (
            (df["ratio_next"] > 20) | ((df["ratio_next"] < 0.05) & (df["ratio_next"] > 0))
        )
        mask = mask | mask_first
        if not mask.any():
            break

        fixed_this_round = 0
        fixed_indices = set()
        bad_rows = df[mask]
        for idx in bad_rows.index:
            pos = df.index.get_loc(idx)
            prev_idx = df.index[pos - 1] if pos > 0 else None
            next_idx = df.index[pos + 1] if pos < len(df) - 1 else None
            curr_p = df.loc[idx, "close"]
            sym = df.loc[idx, "symbol"]

            # So sánh với hàng trước (cùng mã)
            if prev_idx is not None and df.loc[prev_idx, "symbol"] == sym:
                prev_p = df.loc[prev_idx, "close"]
                ratio = curr_p / prev_p if prev_p > 0 else 1

                # curr quá cao (ratio > 20)
                if ratio > 500 and curr_p > 100 and idx not in fixed_indices:
                    df.loc[idx, "close"] = curr_p / 1000
                    fixed_this_round += 1
                    fixed_indices.add(idx)
                elif ratio > 50 and curr_p > 50 and idx not in fixed_indices:
                    df.loc[idx, "close"] = curr_p / 100
                    fixed_this_round += 1
                    fixed_indices.add(idx)
                elif ratio > 20 and curr_p > 20 and idx not in fixed_indices:
                    df.loc[idx, "close"] = curr_p / 10
                    fixed_this_round += 1
                    fixed_indices.add(idx)
                # prev quá cao (ratio < 0.05)
                elif ratio < 0.002 and ratio > 0 and prev_p > curr_p * 100 and prev_idx not in fixed_indices:
                    df.loc[prev_idx, "close"] = prev_p / 1000
                    fixed_this_round += 1
                    fixed_indices.add(prev_idx)
                elif ratio < 0.02 and ratio > 0 and prev_p > curr_p * 50 and prev_idx not in fixed_indices:
                    df.loc[prev_idx, "close"] = prev_p / 100
                    fixed_this_round += 1
                    fixed_indices.add(prev_idx)
                elif ratio < 0.05 and ratio > 0 and prev_p > curr_p * 10 and prev_idx not in fixed_indices:
                    df.loc[prev_idx, "close"] = prev_p / 10
                    fixed_this_round += 1
                    fixed_indices.add(prev_idx)

            # Hàng đầu tiên của mã: so sánh với hàng tiếp theo (cùng mã)
            if next_idx is not None and df.loc[next_idx, "symbol"] == sym and idx not in fixed_indices:
                next_p = df.loc[next_idx, "close"]
                ratio_next = curr_p / next_p if next_p > 0 else 1
                # curr quá cao so với next (curr/next > 20)
                if ratio_next > 500 and curr_p > 100:
                    df.loc[idx, "close"] = curr_p / 1000
                    fixed_this_round += 1
                    fixed_indices.add(idx)
                elif ratio_next > 50 and curr_p > 50:
                    df.loc[idx, "close"] = curr_p / 100
                    fixed_this_round += 1
                    fixed_indices.add(idx)
                elif ratio_next > 20 and curr_p > 20:
                    df.loc[idx, "close"] = curr_p / 10
                    fixed_this_round += 1
                    fixed_indices.add(idx)
                # curr quá thấp so với next (curr/next < 0.05) → next sai
                elif ratio_next < 0.002 and ratio_next > 0 and next_p > curr_p * 100 and next_idx not in fixed_indices:
                    df.loc[next_idx, "close"] = next_p / 1000
                    fixed_this_round += 1
                    fixed_indices.add(next_idx)
                elif ratio_next < 0.02 and ratio_next > 0 and next_p > curr_p * 50 and next_idx not in fixed_indices:
                    df.loc[next_idx, "close"] = next_p / 100
                    fixed_this_round += 1
                    fixed_indices.add(next_idx)
                elif ratio_next < 0.05 and ratio_next > 0 and next_p > curr_p * 10 and next_idx not in fixed_indices:
                    df.loc[next_idx, "close"] = next_p / 10
                    fixed_this_round += 1
                    fixed_indices.add(next_idx)

        total_fixed += fixed_this_round
        if fixed_this_round == 0:
            break

    # Fallback: median(prev,next) cho curr sai; median(prev_prev, curr) cho prev sai
    for _ in range(5):
        df["prev_close"] = df.groupby("symbol")["close"].shift(1)
        df["next_close"] = df.groupby("symbol")["close"].shift(-1)
        df["ratio"] = df["close"] / df["prev_close"]
        same_sym = df["symbol"] == df.groupby("symbol")["symbol"].shift(1)
        fixed_fallback = 0
        # curr quá cao
        mask_curr = (df["ratio"] > 20) & same_sym
        for idx in df.loc[mask_curr].index:
            prev = df.loc[idx, "prev_close"]
            nxt = df.loc[idx, "next_close"]
            v = np.nanmedian([prev, nxt]) if pd.notna(prev) and pd.notna(nxt) else (prev if pd.notna(prev) else nxt)
            if pd.notna(v) and v > 0:
                df.loc[idx, "close"] = v
                total_fixed += 1
                fixed_fallback += 1
        # prev quá cao (curr đúng)
        df["prev_close"] = df.groupby("symbol")["close"].shift(1)
        df["ratio"] = df["close"] / df["prev_close"]
        mask_prev = (df["ratio"] < 0.05) & (df["ratio"] > 0) & same_sym
        for idx in df.loc[mask_prev].index:
            pos = df.index.get_loc(idx)
            if pos == 0:
                continue
            prev_idx = df.index[pos - 1]
            if df.loc[prev_idx, "symbol"] != df.loc[idx, "symbol"]:
                continue
            prev_prev = df.loc[prev_idx, "prev_close"]
            curr = df.loc[idx, "close"]
            if pd.isna(curr) or curr <= 0:
                continue
            # prev_prev hợp lý: dùng median; không thì thay prev = curr
            if pd.notna(prev_prev) and prev_prev > 0 and curr * 0.5 <= prev_prev <= curr * 2:
                v = np.nanmedian([prev_prev, curr])
            else:
                v = curr
            if pd.notna(v) and v > 0:
                df.loc[prev_idx, "close"] = v
                total_fixed += 1
                fixed_fallback += 1
        if fixed_fallback == 0:
            break

    return df.drop(columns=["prev_close", "next_close", "ratio"], errors="ignore"), total_fixed


def main():
    if not PRICE_FILE.exists():
        print(f" Not found: {PRICE_FILE}")
        return

    df = pd.read_csv(PRICE_FILE)
    if "close" not in df.columns:
        print(" No 'close' column")
        return

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close", "date", "symbol"])

    out, total_fixed = fix_decimal_shift(df.copy())

    if total_fixed > 0:
        out.to_csv(PRICE_FILE, index=False)
        print(f" Fixed {total_fixed} price spikes in {PRICE_FILE}")
    else:
        print(" No price spikes to fix")


if __name__ == "__main__":
    main()
