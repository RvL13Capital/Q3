"""
Integration tests for the crowding signal (Signal 3 — CSD approach).

Tests use in-memory DuckDB with synthetic price data; no network calls.

The CSD crowding score has two sub-components:
  1. Δρ₁  — change in lag-1 autocorrelation of stock returns
  2. Δ(λ_max/Σλ) — change in absorption ratio (universe-level)
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.db import upsert_prices
from src.signals.crowding import (
    compute_lag1_autocorr,
    compute_absorption_ratio,
    compute_crowding_score,
    batch_crowding_scores,
)

AS_OF = "2023-01-13"   # last date inside the 400-day synthetic window


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(ticker: str, n_days: int = 400, start_price: float = 100.0,
                 vol: float = 0.015, currency: str = "USD",
                 autocorr: float = 0.0) -> pd.DataFrame:
    """
    Synthetic OHLCV prices.  autocorr > 0 → AR(1) trending process.
    """
    rng = np.random.default_rng(seed=abs(hash(ticker)) % (2 ** 32))
    dates = [date(2022, 1, 3) + timedelta(days=i) for i in range(n_days)]
    shocks = rng.normal(0.0003, vol, n_days)
    returns = np.zeros(n_days)
    returns[0] = shocks[0]
    for i in range(1, n_days):
        returns[i] = autocorr * returns[i - 1] + shocks[i]

    prices = [start_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))

    return pd.DataFrame({
        "ticker":    ticker,
        "date":      dates,
        "open":      [p * 0.99 for p in prices],
        "high":      [p * 1.01 for p in prices],
        "low":       [p * 0.98 for p in prices],
        "close":     prices,
        "adj_close": prices,
        "volume":    [int(1e6)] * n_days,
        "currency":  currency,
        "source":    "test",
    })


# ---------------------------------------------------------------------------
# Lag-1 autocorrelation delta
# ---------------------------------------------------------------------------

def test_lag1_autocorr_no_data(conn, params):
    """Missing ticker → returns zeros with zero confidence."""
    rho_cur, rho_base, delta, conf = compute_lag1_autocorr(
        "MISSING", conn, params, AS_OF
    )
    assert rho_cur == 0.0
    assert delta == 0.0
    assert conf == 0.0


def test_lag1_autocorr_returns_tuple(conn, params):
    """With real price data, should return a 4-tuple of floats."""
    df = _make_prices("ETN", vol=0.015)
    upsert_prices(conn, df)
    result = compute_lag1_autocorr("ETN", conn, params, AS_OF)
    assert len(result) == 4
    rho_cur, rho_base, delta, conf = result
    assert isinstance(delta, float)
    assert 0.0 <= conf <= 1.0


def test_lag1_autocorr_trending_has_positive_rho(conn, params):
    """A strongly trending AR(1) series should have positive current autocorrelation."""
    df_trending = _make_prices("TREND", vol=0.005, autocorr=0.6)
    upsert_prices(conn, df_trending)
    rho_t, _, _, conf_t = compute_lag1_autocorr("TREND", conn, params, AS_OF)
    assert conf_t > 0
    assert rho_t > 0, "Trending series should have positive current autocorrelation"


# ---------------------------------------------------------------------------
# Absorption ratio
# ---------------------------------------------------------------------------

def test_absorption_ratio_no_data(conn, params):
    """No prices → returns neutral values with zero confidence."""
    _, _, _, conf = compute_absorption_ratio(
        ["ETN", "RHM.DE", "CCJ"], conn, params, AS_OF
    )
    assert conf == 0.0


def test_absorption_ratio_correlated_universe(conn, params):
    """Near-perfectly correlated prices → high absorption ratio."""
    base = _make_prices("BASE", vol=0.015)
    tickers = ["T1", "T2", "T3", "T4"]
    rng = np.random.default_rng(42)
    for t in tickers:
        noise = rng.normal(0, 0.001, len(base))
        df = base.copy()
        df["ticker"]    = t
        df["adj_close"] = df["adj_close"] * (1 + noise)
        df["close"]     = df["adj_close"]
        upsert_prices(conn, df)

    ratio_cur, _, _, conf = compute_absorption_ratio(tickers, conn, params, AS_OF)
    assert conf > 0
    assert ratio_cur > 0.5, f"Correlated universe → high absorption, got {ratio_cur}"


def test_absorption_ratio_independent_universe(conn, params):
    """Independent random walks → absorption ratio between 0 and 1."""
    tickers = ["I1", "I2", "I3", "I4", "I5"]
    for t in tickers:
        upsert_prices(conn, _make_prices(t, vol=0.02, autocorr=0.0))

    ratio_cur, _, _, conf = compute_absorption_ratio(tickers, conn, params, AS_OF)
    assert conf > 0
    assert 0.0 < ratio_cur < 1.0


# ---------------------------------------------------------------------------
# Composite CSD crowding score
# ---------------------------------------------------------------------------

def test_crowding_score_bounds(conn, params):
    """crowding_score must always be in [0, 1]."""
    upsert_prices(conn, _make_prices("ETN"))
    result = compute_crowding_score("ETN", conn, params, AS_OF,
                                    absorption_delta=0.05, absorption_conf=0.8)
    assert 0.0 <= result["crowding_score"] <= 1.0
    assert 0.0 <= result["crowding_confidence"] <= 1.0


def test_crowding_score_keys(conn, params):
    """Result dict must contain expected keys."""
    upsert_prices(conn, _make_prices("ETN"))
    result = compute_crowding_score("ETN", conn, params, AS_OF)
    for key in ["ticker", "crowding_score", "crowding_confidence",
                "autocorr_delta", "absorption_delta",
                "etf_corr_score", "short_interest_score"]:
        assert key in result, f"Missing key: {key}"


def test_crowding_score_no_data(conn, params):
    """No prices → safe neutral result, zero confidence."""
    result = compute_crowding_score("MISSING", conn, params, AS_OF)
    assert 0.0 <= result["crowding_score"] <= 1.0
    assert result["crowding_confidence"] == 0.0


def test_crowding_score_high_absorption_raises_score(conn, params):
    """Higher absorption delta → higher crowding score."""
    upsert_prices(conn, _make_prices("ETN"))
    low  = compute_crowding_score("ETN", conn, params, AS_OF,
                                   absorption_delta=-0.05, absorption_conf=0.8)
    high = compute_crowding_score("ETN", conn, params, AS_OF,
                                   absorption_delta=+0.08, absorption_conf=0.8)
    assert high["crowding_score"] > low["crowding_score"]


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def test_batch_crowding_scores_shape(conn, sample_universe, all_prices, params):
    """Batch should return one row per universe stock."""
    upsert_prices(conn, all_prices)
    result = batch_crowding_scores(sample_universe, conn, params, AS_OF)
    assert len(result) == len(sample_universe)
    assert set(result["ticker"]) == set(sample_universe["ticker"])


def test_batch_crowding_scores_bounds(conn, sample_universe, all_prices, params):
    """All crowding scores must be in [0, 1]."""
    upsert_prices(conn, all_prices)
    result = batch_crowding_scores(sample_universe, conn, params, AS_OF)
    assert (result["crowding_score"].between(0, 1)).all()
