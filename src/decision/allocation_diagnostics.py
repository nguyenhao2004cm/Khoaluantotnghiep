# =====================================================
# ALLOCATION DIAGNOSTICS — Problem Diagnosis
# Theo đề xuất thầy: xác định vấn đề bằng dữ liệu
#
# Bước 1: Kiểm tra dispersion của signal
# Bước 2: Kiểm tra cấu trúc rủi ro VIC–VCB (vol, correlation, covariance)
# Bước 5: Signal có đóng góp gì? Correlation(signal, future volatility/drawdown)
# =====================================================

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
import numpy as np

# =====================================
# PATHS
# =====================================
SIGNAL_FILE = PROJECT_DIR / "data_processed/decision/signal.csv"
PRICE_FILE = PROJECT_DIR / "data_processed/prices/prices.csv"
RISK_NORM_DIR = PROJECT_DIR / "data_processed/risk_normalized"
OUT_DIR = PROJECT_DIR / "data_processed/diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROLLING_WINDOW = 60
TRADING_DAYS = 252


def load_prices_pivot(symbols):
    """Load prices: index=date, columns=symbols."""
    df = pd.read_csv(PRICE_FILE)
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["symbol"].isin(symbols)]
    if df.empty:
        return None
    df = df.sort_values("date").groupby(["date", "symbol"], as_index=False).last()
    return df.pivot(index="date", columns="symbol", values="close").sort_index()


# =====================================
# BƯỚC 1: SIGNAL DISPERSION
# =====================================
def analyze_signal_dispersion():
    """
    Khi chỉ có 2 tài sản: signal chênh lệch bao nhiêu?
    Std(signal) theo từng ngày?
    Nếu dispersion thấp → softmax luôn tiệm cận 50-50.
    """
    print("\n" + "=" * 60)
    print(" BUOC 1 - SIGNAL DISPERSION")
    print("=" * 60)

    df = pd.read_csv(SIGNAL_FILE, parse_dates=["date"])

    # Cross-sectional stats per date
    stats = df.groupby("date")["signal"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    stats.columns = ["date", "mean", "std", "min", "max", "count"]

    stats["range"] = stats["max"] - stats["min"]
    stats["dispersion"] = stats["std"].fillna(0)

    # Case 2 assets
    two_asset_dates = stats[stats["count"] == 2]
    if not two_asset_dates.empty:
        merged = two_asset_dates.merge(df, on="date")
        pivot = merged.pivot(index="date", columns="symbol", values="signal")
        pivot["signal_diff"] = pivot.iloc[:, 1] - pivot.iloc[:, 0]
        pivot["signal_diff_abs"] = pivot["signal_diff"].abs()

        print("\n--- When only 2 assets ---")
        print(f"  Days: {len(two_asset_dates)}")
        print(f"  Mean |signal_diff|: {pivot['signal_diff_abs'].mean():.4f}")
        print(f"  Std(signal) mean: {two_asset_dates['std'].mean():.4f}")
        print(f"  Min dispersion: {two_asset_dates['std'].min():.4f}")
        print(f"  Max dispersion: {two_asset_dates['std'].max():.4f}")

        if pivot["signal_diff_abs"].mean() < 0.1:
            print("\n  [!] DISPERSION THAP -> Softmax se tiem can 50-50")
            print("     Data issue, not just algorithm.")

    print("\n--- Dispersion overview ---")
    print(stats["std"].describe().to_string())

    stats.to_csv(OUT_DIR / "signal_dispersion_by_date.csv", index=False)
    print(f"\n  Saved: {OUT_DIR / 'signal_dispersion_by_date.csv'}")
    return stats


# =====================================
# BƯỚC 2: CẤU TRÚC RỦI RO VIC–VCB
# =====================================
def analyze_risk_structure(symbols=None):
    """
    Volatility riêng lẻ, correlation, covariance rolling 60 ngày.
    Nếu correlation cao + vol gần nhau → 50-50 có thể hợp lý (risk parity).
    Nếu vol khác nhau → weight nên khác 50-50.
    """
    if symbols is None:
        symbols = ["VIC", "VCB"]

    print("\n" + "=" * 60)
    print(f" BUOC 2 - CAU TRUC RUI RO ({', '.join(symbols)})")
    print("=" * 60)

    prices = load_prices_pivot(symbols)
    if prices is None or len(prices) < ROLLING_WINDOW:
        print(f"  [!] Insufficient data for {symbols}")
        return None

    ret = prices.pct_change(fill_method=None).dropna(how="all")

    # Rolling 60 ngày
    vol_rolling = ret.rolling(ROLLING_WINDOW).std() * np.sqrt(TRADING_DAYS)
    corr_rolling = ret.rolling(ROLLING_WINDOW).corr()

    # Lấy ngày gần nhất
    last_date = vol_rolling.dropna(how="all").index[-1]
    vol_last = vol_rolling.loc[last_date].dropna()
    ret_last = ret.loc[ret.index <= last_date].tail(ROLLING_WINDOW)

    cov = ret_last.cov().values * TRADING_DAYS
    if len(symbols) == 2:
        corr = ret_last.corr().iloc[0, 1]
    else:
        corr = ret_last.corr()

    print(f"\n  Date: {last_date.date()}")
    print(f"  Rolling window: {ROLLING_WINDOW} days")
    print("\n  --- Volatility (annualized) ---")
    for s in symbols:
        if s in vol_last.index:
            print(f"    {s}: {vol_last[s]:.4f} ({vol_last[s]*100:.2f}%)")

    print("\n  --- Correlation ---")
    if len(symbols) == 2:
        print(f"    {symbols[0]}-{symbols[1]}: {corr:.4f}")
    else:
        print(corr.to_string())

    # ERC weight (2 assets): w1 = sigma2 / (sigma1 + sigma2)
    if len(symbols) == 2 and all(s in vol_last.index for s in symbols):
        sig1, sig2 = vol_last[symbols[0]], vol_last[symbols[1]]
        w1_erc = sig2 / (sig1 + sig2) if (sig1 + sig2) > 0 else 0.5
        w2_erc = 1 - w1_erc
        print("\n  --- ERC weights (Equal Risk Contribution) ---")
        print(f"    {symbols[0]}: {w1_erc:.4f} ({w1_erc*100:.1f}%)")
        print(f"    {symbols[1]}: {w2_erc:.4f} ({w2_erc*100:.1f}%)")
        if abs(w1_erc - 0.5) < 0.05:
            print("    -> Vol gan nhau -> 50-50 la hop ly theo risk parity")
        else:
            print("    -> Vol khac nhau -> weight nen khac 50-50")

    return {"vol": vol_last, "corr": corr, "cov": cov}


# =====================================
# BƯỚC 5: SIGNAL USEFULNESS
# =====================================
def analyze_signal_usefulness():
    """
    Correlation(signal, future volatility)?
    Correlation(signal, drawdown)?
    Nếu signal không dự báo risk dynamics → allocation không nên phụ thuộc mạnh.
    """
    print("\n" + "=" * 60)
    print(" BUOC 5 - SIGNAL USEFULNESS")
    print("=" * 60)

    signal_df = pd.read_csv(SIGNAL_FILE, parse_dates=["date"])
    prices = load_prices_pivot(signal_df["symbol"].unique().tolist())
    if prices is None:
        print("  [!] No price data")
        return

    ret = prices.pct_change(fill_method=None).dropna(how="all")
    # Future vol: rolling 20 ngày forward
    future_vol = ret.rolling(20).std().shift(-20) * np.sqrt(TRADING_DAYS)
    # Drawdown proxy: cumulative min of returns
    cumret = (1 + ret).cumprod()
    dd = (cumret - cumret.cummax()) / cumret.cummax()
    future_dd = dd.shift(-20)

    results = []
    for symbol in ret.columns:
        if symbol not in signal_df["symbol"].values:
            continue
        sig = signal_df[signal_df["symbol"] == symbol].set_index("date")["signal"]
        common = ret[symbol].dropna().index.intersection(sig.index)
        if len(common) < 60:
            continue
        fv = future_vol.loc[common, symbol] if symbol in future_vol.columns else pd.Series(dtype=float)
        fd = future_dd.loc[common, symbol] if symbol in future_dd.columns else pd.Series(dtype=float)
        s = sig.reindex(common).ffill().bfill()

        corr_vol = s.corr(fv) if fv.notna().sum() > 30 else np.nan
        corr_dd = s.corr(fd) if fd.notna().sum() > 30 else np.nan
        results.append({"symbol": symbol, "corr_signal_future_vol": corr_vol, "corr_signal_future_dd": corr_dd})

    res_df = pd.DataFrame(results).dropna(how="all")
    if res_df.empty:
        print("  [!] Insufficient data")
        return

    print("\n  Correlation(signal, future 20d volatility):")
    print(f"    Mean: {res_df['corr_signal_future_vol'].mean():.4f}")
    print(f"    Std:  {res_df['corr_signal_future_vol'].std():.4f}")
    print("\n  Correlation(signal, future drawdown):")
    print(f"    Mean: {res_df['corr_signal_future_dd'].mean():.4f}")
    print(f"    Std:  {res_df['corr_signal_future_dd'].std():.4f}")

    if abs(res_df["corr_signal_future_vol"].mean()) < 0.1:
        print("\n  [!] Signal khong tuong quan manh voi future volatility")
        print("     -> Allocation khong nen phu thuoc qua nhieu vao signal ranking")

    res_df.to_csv(OUT_DIR / "signal_usefulness.csv", index=False)
    print(f"\n  Saved: {OUT_DIR / 'signal_usefulness.csv'}")


# =====================================
# MAIN
# =====================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" ALLOCATION DIAGNOSTICS - Problem Diagnosis")
    print("=" * 60)

    analyze_signal_dispersion()
    analyze_risk_structure(["VIC", "VCB"])
    analyze_signal_usefulness()

    print("\n" + "=" * 60)
    print(" DIAGNOSTICS COMPLETED")
    print("=" * 60 + "\n")
