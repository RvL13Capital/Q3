"""
Fractional Kelly position sizing — EARKE eq 12.

Full objective (without σ_epist, which requires deep ensembles):

  f* = argmax_f [
      (μ − rf) · f
      − ½ · σ² · f²
      − c · σ · |f − f_old|^1.5 · √(W / V)
  ]

  c  = impact_scaling (≈1.0, dimensionless)
  W  = AUM in currency units (params: kelly.aum_eur)
  V  = average daily dollar volume of the stock (last 60 days)
  f_old = last Kelly fraction for this ticker (from portfolio_snapshots)

When W/V is small (liquid stock, small fund), the impact term is negligible and
f* ≈ fraction × (μ−rf)/σ² (classical fractional Kelly).
As AUM grows, the impact penalty rises, progressively reducing optimal f* and
ultimately capping W_max — implementing Goodhart immunity through physics.

Optimised via scipy.optimize.minimize_scalar on [0, 1].
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


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
    """
    from src.data.db import get_prices
    from src.data.prices import compute_log_returns

    start = (datetime.strptime(as_of_date, "%Y-%m-%d")
             - timedelta(days=window_days + 60)).strftime("%Y-%m-%d")
    price_df = get_prices(conn, [ticker], start, as_of_date)

    if price_df.empty or ticker not in price_df.columns:
        return 0.35, False

    returns = compute_log_returns(price_df[[ticker]])[ticker].dropna().tail(window_days)
    n = len(returns)
    if n < 30:
        return 0.35, False

    sigma = float(returns.std() * np.sqrt(252))
    return max(0.05, sigma), True


# ---------------------------------------------------------------------------
# Daily dollar volume estimation
# ---------------------------------------------------------------------------

def estimate_daily_dollar_volume(
    ticker: str,
    conn,
    as_of_date: str,
    window_days: int = 60,
) -> float:
    """
    Average daily dollar volume over the trailing window (price × volume).
    Returns 0.0 if unavailable, which disables the impact term.
    """
    from src.data.db import get_prices

    start = (datetime.strptime(as_of_date, "%Y-%m-%d")
             - timedelta(days=window_days + 10)).strftime("%Y-%m-%d")

    # Fetch raw OHLCV (need volume column, not in the wide adj_close-only format)
    try:
        df = conn.execute("""
            SELECT date, adj_close, volume
            FROM prices
            WHERE ticker = ? AND date >= ? AND date <= ?
            ORDER BY date
        """, [ticker, start, as_of_date]).df()
    except Exception:
        return 0.0

    if df.empty or df["volume"].isna().all():
        return 0.0

    df = df.dropna(subset=["adj_close", "volume"]).tail(window_days)
    if df.empty:
        return 0.0

    daily_vol = (df["adj_close"] * df["volume"]).mean()
    return float(daily_vol) if daily_vol > 0 else 0.0


# ---------------------------------------------------------------------------
# Last Kelly fraction from previous portfolio snapshot
# ---------------------------------------------------------------------------

def get_last_kelly_fraction(conn, ticker: str) -> float:
    """Return kelly_25pct from the most recent portfolio snapshot for this ticker."""
    try:
        result = conn.execute("""
            SELECT kelly_25pct
            FROM portfolio_snapshots
            WHERE ticker = ?
            ORDER BY snapshot_date DESC
            LIMIT 1
        """, [ticker]).fetchone()
        if result and result[0] is not None:
            return float(result[0])
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Core Kelly optimisation (eq 12, without σ_epist)
# ---------------------------------------------------------------------------

def kelly_fraction(
    mu: float,
    sigma: float,
    rf: float,
    fraction: float = 0.25,
    f_old: float = 0.0,
    aum: float = 0.0,
    daily_dollar_volume: float = 0.0,
    impact_scaling: float = 1.0,
) -> float:
    """
    Optimise the Kelly objective with market impact penalty.

    When daily_dollar_volume == 0 or aum == 0, degrades gracefully to the
    classic fractional Kelly: fraction × (μ−rf) / σ².
    """
    if sigma <= 0 or mu <= rf:
        return 0.0

    # Impact term weight √(W/V); 0 when either unknown
    if daily_dollar_volume > 0 and aum > 0:
        sqrt_wv = np.sqrt(aum / daily_dollar_volume)
    else:
        sqrt_wv = 0.0

    if sqrt_wv < 1e-8:
        # No impact data — fall back to classical fractional Kelly
        raw = fraction * (mu - rf) / (sigma ** 2)
        return float(max(0.0, min(1.0, raw)))

    mu_robust = mu - rf  # excess return

    def neg_objective(f: float) -> float:
        growth    = mu_robust * f - 0.5 * sigma ** 2 * f ** 2
        turnover  = abs(f - f_old)
        impact    = impact_scaling * sigma * (turnover ** 1.5) * sqrt_wv
        return -(growth - impact)

    # Optimise over [0, 1/fraction] — equivalent range to the full Kelly,
    # so fraction scaling afterward gives the same result as classic fractional Kelly
    # when impact is negligible.
    upper = min(4.0, 1.0 / max(fraction, 0.05))  # cap at 4× full Kelly
    result = minimize_scalar(neg_objective, bounds=(0.0, upper), method="bounded")
    f_full = float(result.x) if result.success else mu_robust / sigma ** 2

    return float(max(0.0, min(1.0, fraction * f_full)))


# ---------------------------------------------------------------------------
# AUM capacity ceiling (derived from eq 12)
# ---------------------------------------------------------------------------

def compute_w_max(
    mu_robust: float,
    sigma: float,
    daily_dollar_volume: float,
    f_old: float,
    target_turnover: float = 0.10,
    impact_scaling: float = 1.0,
) -> float:
    """
    Hard AUM ceiling beyond which alpha is fully absorbed by market impact.

    W_max = V × ( μ_robust / (c · σ · |Δf|^1.5) )²

    Returns 0.0 if underdetermined.
    """
    if daily_dollar_volume <= 0 or sigma <= 0 or mu_robust <= 0:
        return 0.0
    if target_turnover < 1e-6:
        return 0.0
    denominator = impact_scaling * sigma * (target_turnover ** 1.5)
    if denominator < 1e-10:
        return 0.0
    return float(daily_dollar_volume * (mu_robust / denominator) ** 2)


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
    kelly_fraction, kelly_25pct, w_max, primary_bucket, composite_score.
    """
    from src.data.macro import get_risk_free_rate

    universe_map = universe_df.set_index("ticker")[["region", "primary_bucket"]]
    candidates   = scored_df[scored_df["entry_signal"] == True].copy()

    if candidates.empty:
        return pd.DataFrame()

    kelly_params    = params["kelly"]
    frac            = kelly_params["fraction"]
    aum             = float(kelly_params.get("aum_eur", 0.0))
    impact_scaling  = float(kelly_params.get("impact_scaling", 1.0))

    rows = []
    for _, row in candidates.iterrows():
        ticker      = row["ticker"]
        region_info = universe_map.loc[ticker] if ticker in universe_map.index else None
        region      = region_info["region"]        if region_info is not None else "US"
        bucket      = region_info["primary_bucket"] if region_info is not None else None

        rf = get_risk_free_rate(conn, region, as_of_date, params)
        mu = row.get("mu_estimate")
        if mu is None or pd.isna(mu):
            continue

        sigma, sigma_valid = estimate_sigma(ticker, conn, as_of_date)
        f_old              = get_last_kelly_fraction(conn, ticker)
        daily_vol          = estimate_daily_dollar_volume(ticker, conn, as_of_date)

        f = kelly_fraction(
            mu=mu, sigma=sigma, rf=rf, fraction=frac,
            f_old=f_old, aum=aum,
            daily_dollar_volume=daily_vol,
            impact_scaling=impact_scaling,
        )

        w_max = compute_w_max(
            mu_robust=max(0.0, mu - rf),
            sigma=sigma,
            daily_dollar_volume=daily_vol,
            f_old=f_old,
            impact_scaling=impact_scaling,
        )

        rows.append({
            "ticker":          ticker,
            "mu_estimate":     round(mu, 4),
            "sigma_estimate":  round(sigma, 4),
            "rf_rate":         round(rf, 4),
            "kelly_raw":       round(f, 4),
            "kelly_25pct":     round(f, 4),
            "w_max_eur":       round(w_max, 0) if w_max > 0 else None,
            "primary_bucket":  bucket,
            "composite_score": row.get("composite_score"),
            "sigma_valid":     sigma_valid,
        })

    return pd.DataFrame(rows)
