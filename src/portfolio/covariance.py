"""
Ledoit-Wolf (2004) analytical shrinkage estimator for covariance matrices.

Implements the closed-form optimal shrinkage toward a scaled identity target:
    Σ_shrunk = δ·F + (1−δ)·S

where S is the sample covariance, F = mean(diag(S))·I is the shrinkage target,
and δ ∈ [0,1] is the analytically optimal shrinkage intensity.

Pure numpy implementation — avoids a scikit-learn dependency for a single function.
Reference: Ledoit & Wolf, "A well-conditioned estimator for large-dimensional
covariance matrices", Journal of Multivariate Analysis 88 (2004) 365–411.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core shrinkage estimator
# ---------------------------------------------------------------------------

def ledoit_wolf_shrink(
    returns: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf analytical shrinkage toward scaled identity.

    Parameters
    ----------
    returns : pd.DataFrame
        T×N matrix of demeaned or raw returns (rows=observations, cols=assets).
        Must have T ≥ 2 and N ≥ 1.

    Returns
    -------
    shrunk_cov : np.ndarray
        N×N positive semi-definite shrunk covariance matrix.
    delta : float
        Optimal shrinkage intensity in [0, 1].
        δ → 1 means high shrinkage (few observations relative to assets).
        δ → 0 means the sample covariance is reliable.
    """
    X = returns.values.astype(np.float64)
    T, N = X.shape

    # Demean
    X = X - X.mean(axis=0, keepdims=True)

    # Sample covariance (unbiased)
    S = (X.T @ X) / (T - 1)

    # Shrinkage target: scaled identity F = μ·I where μ = mean of diagonal
    mu = np.trace(S) / N
    F = mu * np.eye(N)

    # Frobenius norm terms for the optimal shrinkage intensity
    # δ* = min(1, (β̄²) / (d̄²))  where:
    #   d̄² = ||S − F||²_F             (distance from target)
    #   β̄² = (1/T²) Σ_t ||x_t x_t' − S||²_F  (estimation error)

    d_sq = np.sum((S - F) ** 2)

    # β̄² computation: sum of squared deviations of outer products from S
    beta_sq = 0.0
    for t in range(T):
        x_t = X[t:t + 1, :]  # 1×N
        outer_t = x_t.T @ x_t  # N×N (unnormalized outer product for single obs)
        beta_sq += np.sum((outer_t - S) ** 2)
    beta_sq /= T ** 2

    # Optimal shrinkage intensity
    if d_sq < 1e-15:
        delta = 1.0  # S ≈ F already; shrink fully
    else:
        delta = float(min(1.0, max(0.0, beta_sq / d_sq)))

    shrunk = delta * F + (1.0 - delta) * S
    return shrunk, delta


# ---------------------------------------------------------------------------
# High-level estimator with price fetching and annualization
# ---------------------------------------------------------------------------

def estimate_covariance_matrix(
    tickers: list[str],
    conn,
    as_of_date: str,
    window_days: int = 252,
    min_obs: int = 60,
    sigma_floor: float = 0.05,
) -> tuple[Optional[pd.DataFrame], float, bool]:
    """Fetch prices, compute log returns, apply Ledoit-Wolf shrinkage.

    Parameters
    ----------
    tickers : list[str]
        Tickers to include in the covariance matrix.
    conn : duckdb.DuckDBPyConnection
        Database connection.
    as_of_date : str
        YYYY-MM-DD date for the estimation window end.
    window_days : int
        Number of trading days for the estimation window (default 252 ≈ 1 year).
    min_obs : int
        Minimum number of non-NaN return observations required (default 60).
    sigma_floor : float
        Floor for annualized per-asset volatility (default 5%).

    Returns
    -------
    cov_df : pd.DataFrame or None
        N×N annualized covariance matrix indexed/columned by ticker.
        None if estimation fails (insufficient data).
    shrinkage_intensity : float
        Ledoit-Wolf optimal δ. Higher = more shrinkage needed.
    is_valid : bool
        True if matrix was successfully estimated.
    """
    from src.data.db import get_prices
    from src.data.prices import compute_log_returns

    if len(tickers) < 1:
        return None, 0.0, False

    # Fetch extra buffer for lead-in to log returns
    start = (datetime.strptime(as_of_date, "%Y-%m-%d")
             - timedelta(days=window_days + 90)).strftime("%Y-%m-%d")

    price_df = get_prices(conn, tickers, start, as_of_date)
    if price_df.empty:
        return None, 0.0, False

    # Keep only requested tickers that exist in price data
    available = [t for t in tickers if t in price_df.columns]
    if len(available) < 1:
        return None, 0.0, False

    returns = compute_log_returns(price_df[available]).tail(window_days)

    # Drop tickers with too many missing values
    valid_cols = returns.columns[returns.notna().sum() >= min_obs]
    if len(valid_cols) < 1:
        return None, 0.0, False

    clean = returns[valid_cols].dropna()
    if len(clean) < min_obs:
        return None, 0.0, False

    # Apply Ledoit-Wolf shrinkage
    shrunk_daily, delta = ledoit_wolf_shrink(clean)

    # Annualize: Σ_annual = 252 × Σ_daily
    shrunk_annual = shrunk_daily * 252

    # Enforce sigma floor on diagonal
    floor_var = sigma_floor ** 2
    for i in range(shrunk_annual.shape[0]):
        if shrunk_annual[i, i] < floor_var:
            shrunk_annual[i, i] = floor_var

    cov_df = pd.DataFrame(
        shrunk_annual,
        index=list(valid_cols),
        columns=list(valid_cols),
    )

    logger.info(
        "Covariance matrix: %d×%d, %d obs, δ=%.3f (shrinkage intensity)",
        len(valid_cols), len(valid_cols), len(clean), delta,
    )

    return cov_df, delta, True


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix.

    Parameters
    ----------
    cov : np.ndarray
        N×N covariance matrix.

    Returns
    -------
    corr : np.ndarray
        N×N correlation matrix with 1.0 on diagonal.
    """
    std = np.sqrt(np.diag(cov))
    std[std < 1e-15] = 1e-15  # prevent division by zero
    outer_std = np.outer(std, std)
    corr = cov / outer_std
    # Ensure exact 1.0 on diagonal (numerical hygiene)
    np.fill_diagonal(corr, 1.0)
    return corr
