"""
Unit + integration tests for the swing momentum timing layer (signals/momentum.py).

Tests use synthetic in-memory DuckDB price data; no network calls.

Three sub-scores under test:
  1. RS Score   — percentile rank vs universe peers
  2. Breakout Score — pivot breakout proximity × volume confirmation
  3. VCP Score  — ATR compression (volatility contraction)

Composite swing_score and its gating effect on composite entry_signal also tested.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.db import initialize_schema, upsert_prices
from src.signals.momentum import (
    compute_rs_score,
    compute_breakout_score,
    compute_vcp_score,
    compute_momentum_score,
    batch_momentum_scores,
)
from src.signals.composite import compute_composite_score

# ────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ────────────────────────────────────────────────────────────────────────────

AS_OF = "2023-06-30"
N_DAYS = 300   # enough for 252-day windows

PARAMS = {
    "signals": {
        "entry_threshold":          0.30,
        "crowding_entry_max":       0.40,
        "crowding_exit_threshold":  0.75,
        "quality_exit_threshold":   0.25,
        "composite_decay_pct":      0.20,
        "min_composite_confidence": 0.40,
    },
    "return_estimation": {
        "equity_risk_premium": 0.05,
        "theta_risk_premium":  0.30,
    },
    "momentum": {
        "rs_lookback_long":       252,
        "rs_lookback_short":       63,
        "rs_skip_days":             5,
        "breakout_window":         252,
        "breakout_volume_window":   50,
        "breakout_volume_ratio":   1.5,
        "vcp_atr_window_recent":   15,
        "vcp_atr_window_baseline": 60,
        "swing_weight_rs":        0.40,
        "swing_weight_breakout":  0.40,
        "swing_weight_vcp":       0.20,
        "swing_entry_threshold":  0.40,
    },
}


def _make_price_df(
    ticker: str,
    n_days: int = N_DAYS,
    start_price: float = 100.0,
    trend: float = 0.001,     # daily drift (positive = uptrend)
    vol: float = 0.010,       # daily volatility
    vol_ratio_end: float = 1.0,  # final N days: vol × this factor
    vol_contraction_days: int = 0,  # apply vol_ratio_end to last N days
    avg_volume: int = 1_000_000,
    volume_spike_end: bool = False,  # spike volume in last 5 days
    currency: str = "USD",
) -> pd.DataFrame:
    """Build a synthetic price DataFrame suitable for upsert_prices."""
    rng = np.random.default_rng(seed=abs(hash(ticker)) % (2 ** 32))
    end_date   = date.fromisoformat(AS_OF)
    start_date = end_date - timedelta(days=n_days - 1)
    all_dates  = [start_date + timedelta(days=i) for i in range(n_days)]

    vols = np.full(n_days, vol)
    if vol_contraction_days > 0:
        vols[-vol_contraction_days:] = vol * vol_ratio_end

    returns   = rng.normal(trend, vols, n_days)
    prices    = [start_price]
    for r in returns[1:]:
        prices.append(max(prices[-1] * (1 + r), 0.01))

    volumes = np.full(n_days, avg_volume, dtype=float)
    if volume_spike_end:
        volumes[-5:] = avg_volume * 2.0

    return pd.DataFrame({
        "ticker":    ticker,
        "date":      all_dates,
        "open":      [p * 0.99 for p in prices],
        "high":      [p * 1.01 for p in prices],
        "low":       [p * 0.98 for p in prices],
        "close":     prices,
        "adj_close": prices,
        "volume":    volumes.astype(int),
        "currency":  currency,
        "source":    "test",
    })


@pytest.fixture
def conn():
    """In-memory DuckDB connection with schema initialised."""
    c = duckdb.connect(":memory:")
    initialize_schema(c)
    yield c
    c.close()


# ────────────────────────────────────────────────────────────────────────────
# RS Score tests
# ────────────────────────────────────────────────────────────────────────────

class TestRSScore:
    def test_top_performer_ranks_high(self, conn):
        """Stock with strong uptrend should rank near 1.0 vs flat peers."""
        tickers = ["LEADER", "FLAT1", "FLAT2", "FLAT3"]
        upsert_prices(conn, _make_price_df("LEADER", trend=0.003))
        for t in ["FLAT1", "FLAT2", "FLAT3"]:
            upsert_prices(conn, _make_price_df(t, trend=0.0, vol=0.005))

        rs, conf = compute_rs_score("LEADER", tickers, conn, PARAMS, AS_OF)
        assert rs > 0.70, f"Leader RS should be > 0.70, got {rs}"
        assert conf > 0.5

    def test_bottom_performer_ranks_low(self, conn):
        """Stock with strong downtrend should rank near 0.0 vs rising peers."""
        tickers = ["LAGGARD", "BULL1", "BULL2", "BULL3"]
        upsert_prices(conn, _make_price_df("LAGGARD", trend=-0.003))
        for t in ["BULL1", "BULL2", "BULL3"]:
            upsert_prices(conn, _make_price_df(t, trend=0.002))

        rs, conf = compute_rs_score("LAGGARD", tickers, conn, PARAMS, AS_OF)
        assert rs < 0.30, f"Laggard RS should be < 0.30, got {rs}"

    def test_missing_ticker_returns_neutral(self, conn):
        """Ticker with no price data should return 0.5 neutral + 0.0 confidence."""
        upsert_prices(conn, _make_price_df("PEER1"))
        rs, conf = compute_rs_score("GHOST", ["GHOST", "PEER1"], conn, PARAMS, AS_OF)
        assert rs == pytest.approx(0.5)
        assert conf == pytest.approx(0.0)

    def test_single_ticker_universe_returns_neutral(self, conn):
        """Universe of 1 cannot rank — should return neutral."""
        upsert_prices(conn, _make_price_df("SOLO"))
        rs, conf = compute_rs_score("SOLO", ["SOLO"], conn, PARAMS, AS_OF)
        # Only 1 ticker → len(rs_map) < 3 → neutral
        assert rs == pytest.approx(0.5)
        assert conf == pytest.approx(0.0)


# ────────────────────────────────────────────────────────────────────────────
# Breakout Score tests
# ────────────────────────────────────────────────────────────────────────────

class TestBreakoutScore:
    def test_at_52w_high_high_volume_scores_high(self, conn):
        """Stock making new N-day high on volume spike → high breakout score."""
        # Strong uptrend → last price is near/at the period high
        upsert_prices(conn, _make_price_df("BREAKOUT", trend=0.002, volume_spike_end=True))
        score, conf = compute_breakout_score("BREAKOUT", conn, PARAMS, AS_OF)
        # Price should be near the N-day high (monotone uptrend), volume spiking
        assert score > 0.3, f"Breakout score expected > 0.3, got {score}"
        assert conf > 0.5

    def test_far_from_high_scores_near_zero(self, conn):
        """Stock well below its N-day high should score near zero."""
        # Build a series that peaks early, then drops significantly
        rng = np.random.default_rng(42)
        n = N_DAYS
        end_date   = date.fromisoformat(AS_OF)
        all_dates  = [end_date - timedelta(days=n - 1 - i) for i in range(n)]
        # Sharp rally first half, big selloff second half
        prices = [100.0]
        for i in range(1, n):
            if i < n // 3:
                prices.append(prices[-1] * 1.003)  # rally
            else:
                prices.append(prices[-1] * 0.998)  # selloff

        df = pd.DataFrame({
            "ticker": "FALLEN", "date": all_dates,
            "open": prices, "high": prices, "low": prices,
            "close": prices, "adj_close": prices,
            "volume": [1_000_000] * n, "currency": "USD", "source": "test",
        })
        upsert_prices(conn, df)
        score, conf = compute_breakout_score("FALLEN", conn, PARAMS, AS_OF)
        assert score < 0.3, f"Fallen score should be < 0.3, got {score}"

    def test_missing_ticker_returns_zero(self, conn):
        """No price data → score = 0, confidence = 0."""
        score, conf = compute_breakout_score("GHOST", conn, PARAMS, AS_OF)
        assert score == pytest.approx(0.0)
        assert conf == pytest.approx(0.0)


# ────────────────────────────────────────────────────────────────────────────
# VCP Score tests
# ────────────────────────────────────────────────────────────────────────────

class TestVCPScore:
    def test_tight_base_scores_high(self, conn):
        """Recent ATR much smaller than baseline → vcp_score near 1."""
        # High baseline vol, then contract sharply in last 15 days
        upsert_prices(conn, _make_price_df(
            "TIGHT", vol=0.020, vol_ratio_end=0.15, vol_contraction_days=15
        ))
        score, conf = compute_vcp_score("TIGHT", conn, PARAMS, AS_OF)
        assert score > 0.50, f"Tight base VCP should be > 0.50, got {score}"

    def test_expanding_volatility_scores_low(self, conn):
        """Recent ATR much larger than baseline → vcp_score near 0."""
        upsert_prices(conn, _make_price_df(
            "EXPAND", vol=0.005, vol_ratio_end=4.0, vol_contraction_days=15
        ))
        score, conf = compute_vcp_score("EXPAND", conn, PARAMS, AS_OF)
        assert score < 0.20, f"Expanding vol VCP should be < 0.20, got {score}"

    def test_neutral_baseline_vol_scores_near_one_third(self, conn):
        """Constant vol → atr_ratio ≈ 1 → vcp_score ≈ 1 − 1/1.5 ≈ 0.33."""
        upsert_prices(conn, _make_price_df("FLAT_VOL", vol=0.010))
        score, conf = compute_vcp_score("FLAT_VOL", conn, PARAMS, AS_OF)
        assert 0.15 < score < 0.55, f"Neutral VCP should be ~0.33, got {score}"

    def test_missing_ticker_returns_neutral(self, conn):
        """No data → neutral 0.5 with 0.0 confidence."""
        score, conf = compute_vcp_score("GHOST", conn, PARAMS, AS_OF)
        assert score == pytest.approx(0.5)
        assert conf == pytest.approx(0.0)


# ────────────────────────────────────────────────────────────────────────────
# Composite swing score
# ────────────────────────────────────────────────────────────────────────────

class TestCompositeSwingScore:
    def test_leader_scores_higher_than_laggard(self, conn):
        """RS + breakout + VCP all favour the uptrending stock."""
        tickers = ["LEADER2", "LAGGARD2", "PEER_A", "PEER_B"]
        upsert_prices(conn, _make_price_df(
            "LEADER2", trend=0.003, vol_ratio_end=0.2, vol_contraction_days=15,
            volume_spike_end=True
        ))
        upsert_prices(conn, _make_price_df("LAGGARD2", trend=-0.002))
        for t in ["PEER_A", "PEER_B"]:
            upsert_prices(conn, _make_price_df(t, trend=0.0005))

        leader  = compute_momentum_score("LEADER2",  tickers, conn, PARAMS, AS_OF)
        laggard = compute_momentum_score("LAGGARD2", tickers, conn, PARAMS, AS_OF)

        assert leader["swing_score"] > laggard["swing_score"]

    def test_output_keys_present(self, conn):
        """compute_momentum_score must return all required keys."""
        upsert_prices(conn, _make_price_df("X"))
        result = compute_momentum_score("X", ["X"], conn, PARAMS, AS_OF)
        for key in ("ticker", "swing_score", "swing_confidence",
                    "rs_rank", "breakout_score", "vcp_score"):
            assert key in result, f"Missing key: {key}"

    def test_scores_in_unit_range(self, conn):
        """All sub-scores and composite must be in [0, 1]."""
        tickers = ["A2", "B2", "C2"]
        for t in tickers:
            upsert_prices(conn, _make_price_df(t))
        result = compute_momentum_score("A2", tickers, conn, PARAMS, AS_OF)
        for key in ("swing_score", "rs_rank", "breakout_score", "vcp_score"):
            assert 0.0 <= result[key] <= 1.0, f"{key} = {result[key]} out of [0,1]"

    def test_batch_momentum_scores_length(self, conn):
        """batch_momentum_scores returns one row per universe stock."""
        universe = pd.DataFrame([
            {"ticker": "T1", "region": "US"},
            {"ticker": "T2", "region": "US"},
            {"ticker": "T3", "region": "EU"},
        ])
        for t in ["T1", "T2", "T3"]:
            upsert_prices(conn, _make_price_df(t))

        df = batch_momentum_scores(universe, conn, PARAMS, AS_OF)
        assert len(df) == 3
        assert set(df["ticker"]) == {"T1", "T2", "T3"}


# ────────────────────────────────────────────────────────────────────────────
# Integration: swing gate in composite entry_signal
# ────────────────────────────────────────────────────────────────────────────

class TestSwingGateInComposite:
    """
    Verify that compute_composite_score gates entry_signal via swing_score.
    No DB required — swing dict injected directly.
    """

    GOOD_FUNDAMENTAL = (
        {"ticker": "T", "physical_norm": 0.70, "physical_confidence": 1.0},
        {"ticker": "T", "quality_score":  0.70,  "quality_confidence": 1.0},
        {"ticker": "T", "crowding_score": 0.20, "crowding_confidence": 1.0},
    )

    def test_strong_swing_allows_entry(self):
        phys, qual, crowd = self.GOOD_FUNDAMENTAL
        swing = {"swing_score": 0.75, "swing_confidence": 0.90,
                 "rs_rank": 0.85, "breakout_score": 0.70, "vcp_score": 0.60}
        result = compute_composite_score(phys, qual, crowd, PARAMS, rf=0.04, swing=swing)
        assert result["entry_signal"] is True

    def test_weak_swing_suppresses_entry(self):
        """Good fundamentals but weak swing setup → entry_signal = False."""
        phys, qual, crowd = self.GOOD_FUNDAMENTAL
        swing = {"swing_score": 0.10, "swing_confidence": 0.80,
                 "rs_rank": 0.15, "breakout_score": 0.05, "vcp_score": 0.20}
        result = compute_composite_score(phys, qual, crowd, PARAMS, rf=0.04, swing=swing)
        assert result["entry_signal"] is False

    def test_no_swing_data_leaves_entry_unchanged(self):
        """When swing=None, the fundamental entry logic is unchanged (backward compat)."""
        phys, qual, crowd = self.GOOD_FUNDAMENTAL
        result_no_swing = compute_composite_score(phys, qual, crowd, PARAMS, rf=0.04, swing=None)
        # Without swing gate, good fundamentals → entry_signal True
        assert result_no_swing["entry_signal"] is True

    def test_zero_threshold_disables_gate(self):
        """swing_entry_threshold=0.0 → swing_score irrelevant, entry fires on fundamentals."""
        params_no_gate = {**PARAMS, "momentum": {**PARAMS["momentum"], "swing_entry_threshold": 0.0}}
        phys, qual, crowd = self.GOOD_FUNDAMENTAL
        swing = {"swing_score": 0.05, "swing_confidence": 0.80,
                 "rs_rank": 0.10, "breakout_score": 0.05, "vcp_score": 0.10}
        result = compute_composite_score(phys, qual, crowd, params_no_gate, rf=0.04, swing=swing)
        assert result["entry_signal"] is True

    def test_swing_score_stored_in_output(self):
        """swing_score and sub-scores must be present in compute_composite_score output."""
        phys, qual, crowd = self.GOOD_FUNDAMENTAL
        swing = {"swing_score": 0.60, "swing_confidence": 0.85,
                 "rs_rank": 0.80, "breakout_score": 0.55, "vcp_score": 0.45}
        result = compute_composite_score(phys, qual, crowd, PARAMS, rf=0.04, swing=swing)
        assert result["swing_score"]    == pytest.approx(0.60, abs=1e-4)
        assert result["rs_rank"]        == pytest.approx(0.80, abs=1e-4)
        assert result["breakout_score"] == pytest.approx(0.55, abs=1e-4)
        assert result["vcp_score"]      == pytest.approx(0.45, abs=1e-4)

    def test_swing_none_output_is_none(self):
        """With no swing data, swing columns in output should be None."""
        phys, qual, crowd = self.GOOD_FUNDAMENTAL
        result = compute_composite_score(phys, qual, crowd, PARAMS, rf=0.04, swing=None)
        assert result["swing_score"] is None
        assert result["rs_rank"]     is None
