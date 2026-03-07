"""
Fractional Kelly position sizing.

f* = kelly_fraction × (μ − rf) / σ²

Uses composite_score → μ mapping from composite.py.
σ is historical annualized volatility from trailing 252 days.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core Kelly formula
# ---------------------------------------------------------------------------

def kelly_fraction(
    mu: float,
    sigma: float,
    rf: float,
    fraction: float = 0.25,
) -> float:
    """
    Fractional Kelly weight.
    f* = fraction × (μ − rf) / σ²
    Clamped to [0, 1] (long-only, no leverage).
    """
    if sigma <= 0 or mu <= rf:
        return 0.0
    raw = fraction * (mu - rf) / (sigma ** 2)
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# σ estimation
# ---------------------------------------------------------------------------

def estimate_sigma(
    ticker: str,
    conn,
    as_of_date: str,
    window_days: int = 252,
) -> tuple[float, bool]:
    """
    Annualized volatility from trailing log returns.
    Returns (sigma, is_valid).
    is_valid = False if fewer than 60 data points available.
    """
    from src.data.db import get_prices
    from src.data.prices import compute_log_returns
    from datetime import datetime, timedelta

    start = (datetime.strptime(as_of_date, "%Y-%m-%d")
             - timedelta(days=window_days + 60)).strftime("%Y-%m-%d")
    price_df = get_prices(conn, [ticker], start, as_of_date)

    if price_df.empty or ticker not in price_df.columns:
        return 0.35, False  # fallback vol

    returns = compute_log_returns(price_df[[ticker]])[ticker].dropna().tail(window_days)
    n = len(returns)
    if n < 30:
        return 0.35, False

    sigma = float(returns.std() * np.sqrt(252))
    return max(0.05, sigma), True  # floor at 5% to avoid division by tiny number


# ---------------------------------------------------------------------------
# Compute Kelly weights for scored DataFrame
# ---------------------------------------------------------------------------

def compute_kelly_weights(
    scored_df: pd.DataFrame,
    conn,
    params: dict,
    as_of_date: str,
    universe_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each stock with entry_signal=True, compute Kelly position size.
    Returns DataFrame with: ticker, mu_estimate, sigma_estimate, rf_rate,
    kelly_fraction, kelly_25pct, primary_bucket, composite_score.
    """
    from src.data.macro import get_risk_free_rate

    # Merge region info
    universe_map = universe_df.set_index("ticker")[["region", "primary_bucket"]]
    candidates = scored_df[scored_df["entry_signal"] == True].copy()

    if candidates.empty:
        return pd.DataFrame()

    rows = []
    kelly_fraction_param = params["kelly"]["fraction"]

    for _, row in candidates.iterrows():
        ticker = row["ticker"]
        region_info = universe_map.loc[ticker] if ticker in universe_map.index else None
        region = region_info["region"] if region_info is not None else "US"
        bucket = region_info["primary_bucket"] if region_info is not None else None

        rf = get_risk_free_rate(conn, region, as_of_date, params)

        mu = row.get("mu_estimate")
        if mu is None or pd.isna(mu):
            continue  # skip if no mu estimate

        sigma, sigma_valid = estimate_sigma(ticker, conn, as_of_date)

        f = kelly_fraction(mu, sigma, rf, fraction=kelly_fraction_param)

        rows.append({
            "ticker":          ticker,
            "mu_estimate":     round(mu, 4),
            "sigma_estimate":  round(sigma, 4),
            "rf_rate":         round(rf, 4),
            "kelly_raw":       round(f, 4),
            "kelly_25pct":     round(f, 4),  # fraction is already applied above
            "primary_bucket":  bucket,
            "composite_score": row.get("composite_score"),
            "sigma_valid":     sigma_valid,
        })

    return pd.DataFrame(rows)
