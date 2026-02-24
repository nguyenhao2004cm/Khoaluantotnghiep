import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize

# =====================================
# CONFIG
# =====================================
PROJECT_DIR = Path(__file__).resolve().parents[2]
PRICE_FILE = PROJECT_DIR / "data_processed" / "prices" / "prices.csv"
TRADING_DAYS = 252


# =====================================
# LOAD CLEAN ASSET RETURNS
# =====================================
def load_asset_returns(symbols):
    df = pd.read_csv(PRICE_FILE, parse_dates=["date"])

    df["symbol"] = df["symbol"].str.upper()
    df = df[df["symbol"].isin(symbols)]
    df = df.dropna(subset=["date", "symbol", "close"])

    # FIX duplicate (date, symbol) — chuẩn Portfolio Visualizer
    df = (
        df.sort_values("date")
          .groupby(["date", "symbol"], as_index=False)
          .last()
    )

    prices = (
        df.pivot(index="date", columns="symbol", values="close")
          .sort_index()
    )

    returns = prices.pct_change(fill_method=None).dropna(how="all")
    return returns


# =====================================
# COMPUTE ASSET STATS
# =====================================
def compute_asset_stats(asset_returns):
    mean_returns = asset_returns.mean() * TRADING_DAYS
    volatilities = asset_returns.std() * np.sqrt(TRADING_DAYS)

    return pd.DataFrame({
        "Expected Return": mean_returns,
        "Volatility": volatilities
    })


# =====================================
# EFFICIENT FRONTIER — QUADRATIC OPTIMIZATION
# =====================================
def compute_efficient_frontier(mean_returns, cov_matrix, n_points=50):
    n_assets = len(mean_returns)

    target_returns = np.linspace(
        mean_returns.min(),
        mean_returns.max(),
        n_points
    )

    frontier_vols = []
    frontier_rets = []

    for target in target_returns:

        def portfolio_volatility(weights):
            return np.sqrt(weights.T @ cov_matrix @ weights)

        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w: w @ mean_returns - target},
        )

        bounds = tuple((0, 1) for _ in range(n_assets))
        x0 = np.ones(n_assets) / n_assets

        result = minimize(
            portfolio_volatility,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            frontier_vols.append(result.fun)
            frontier_rets.append(target)

    return np.array(frontier_vols), np.array(frontier_rets)


# =====================================
# PLOT — PV STYLE
# =====================================
def plot_asset_risk_return(asset_stats, frontier_vols, frontier_rets):
    plt.figure(figsize=(9, 6))

    # Efficient Frontier (LINE)
    plt.plot(
        frontier_vols,
        frontier_rets,
        color="steelblue",
        linewidth=2.5,
        label="Efficient Frontier"
    )

    # Asset points
    plt.scatter(
        asset_stats["Volatility"],
        asset_stats["Expected Return"],
        color="black",
        s=40,
        zorder=5,
        label="Assets"
    )

    # Annotate asset names
    for symbol, row in asset_stats.iterrows():
        plt.annotate(
            symbol,
            (row["Volatility"], row["Expected Return"]),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.xlabel("Standard Deviation (Risk)")
    plt.ylabel("Expected Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.close()


# =====================================
# RUN
# =====================================
if __name__ == "__main__":
    # Lấy danh mục hiện tại (top K cổ phiếu)
    from holdings import current_holdings

    _, holdings = current_holdings()
    symbols = holdings["symbol"].tolist()

    # Load returns
    asset_returns = load_asset_returns(symbols)

    # Asset stats
    asset_stats = compute_asset_stats(asset_returns)

    # Efficient frontier
    mean_returns = asset_returns.mean() * TRADING_DAYS
    cov_matrix = asset_returns.cov() * TRADING_DAYS

    frontier_vols, frontier_rets = compute_efficient_frontier(
        mean_returns.values,
        cov_matrix.values
    )

    # Plot
    plot_asset_risk_return(asset_stats, frontier_vols, frontier_rets)
