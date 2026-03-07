"""
Tests for dashboard query / transformation logic.

streamlit is not installed in the test environment, so we stub it out
before importing anything from src.reporting.dashboard.
The tests cover:
  - _color_crowding / _color_composite helper functions
  - Portfolio-tab merge logic (Tab 1)
  - Universe-scanner filter + sort logic (Tab 3)
  - Exit-monitor status thresholds and alert counts (Tab 4)
"""
import sys
import types
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Stub the streamlit module before the dashboard can import it
# ---------------------------------------------------------------------------

def _make_streamlit_stub():
    st = types.ModuleType("streamlit")
    # cache decorators must return the function unchanged
    st.cache_resource = lambda f=None, **kw: (f if f else lambda fn: fn)
    st.cache_data = lambda f=None, ttl=None, **kw: (f if f else lambda fn: fn)
    # everything else is a no-op MagicMock
    st.__getattr__ = lambda self, _: MagicMock()
    return st


sys.modules.setdefault("streamlit", _make_streamlit_stub())

from src.reporting.dashboard import _color_crowding, _color_composite  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers shared across test sections
# ---------------------------------------------------------------------------

def _make_scores(*tickers, composite=0.60, quality=0.50, crowding=0.30,
                 physical=0.70, entry=True) -> pd.DataFrame:
    """Build a minimal signal_scores-like DataFrame."""
    return pd.DataFrame([
        {
            "ticker":              t,
            "score_date":          date(2025, 3, 1),
            "composite_score":     composite,
            "quality_score":       quality,
            "crowding_score":      crowding,
            "physical_norm":       physical,
            "entry_signal":        entry,
            "roic_wacc_spread":    0.08,
            "margin_snr":          4.5,
            "inflation_convexity": 0.03,
            "etf_correlation":     0.25,
            "trends_norm":         0.40,
            "short_pct":           0.05,
            "mu_estimate":         0.12,
            "sigma_estimate":      0.20,
            "kelly_25pct":         0.06,
        }
        for t in tickers
    ])


def _make_portfolio(*tickers, weight=0.08) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "ticker":          t,
            "weight":          weight,
            "snapshot_date":   date(2025, 3, 1),
            "composite_score": 0.60,
            "kelly_25pct":     0.06,
            "bucket_id":       "grid",
            "is_new_position": True,
            "rationale":       "test",
        }
        for t in tickers
    ])


def _make_universe() -> pd.DataFrame:
    return pd.DataFrame([
        {"ticker": "ETN",    "region": "US", "primary_bucket": "grid"},
        {"ticker": "RHM.DE", "region": "EU", "primary_bucket": "defense"},
        {"ticker": "CCJ",    "region": "US", "primary_bucket": "nuclear"},
        {"ticker": "AWK",    "region": "US", "primary_bucket": "water"},
        {"ticker": "NVDA",   "region": "US", "primary_bucket": "ai_infra"},
    ])


# ===========================================================================
# 1. Color helper functions
# ===========================================================================

class TestColorCrowding:
    """_color_crowding: red ≥ 0.75, yellow ≥ 0.55, green otherwise."""

    @pytest.mark.parametrize("val", [float("nan"), pd.NA])
    def test_nan_returns_empty_string(self, val):
        assert _color_crowding(val) == ""

    @pytest.mark.parametrize("val", [0.75, 0.80, 1.00])
    def test_above_exit_threshold_is_red(self, val):
        assert "ffcccc" in _color_crowding(val)

    @pytest.mark.parametrize("val", [0.55, 0.60, 0.74])
    def test_watch_zone_is_yellow(self, val):
        assert "fff3cc" in _color_crowding(val)

    @pytest.mark.parametrize("val", [0.00, 0.30, 0.54])
    def test_below_watch_threshold_is_green(self, val):
        assert "ccffcc" in _color_crowding(val)

    def test_exact_boundaries(self):
        assert "ffcccc" in _color_crowding(0.75)   # boundary → red
        assert "fff3cc" in _color_crowding(0.55)   # boundary → yellow
        assert "ccffcc" in _color_crowding(0.54)   # just below → green


class TestColorComposite:
    """_color_composite: green ≥ 0.65, light-blue ≥ 0.55, empty otherwise."""

    @pytest.mark.parametrize("val", [float("nan"), pd.NA])
    def test_nan_returns_empty_string(self, val):
        assert _color_composite(val) == ""

    @pytest.mark.parametrize("val", [0.65, 0.80, 1.00])
    def test_strong_signal_is_green(self, val):
        assert "ccffcc" in _color_composite(val)

    @pytest.mark.parametrize("val", [0.55, 0.60, 0.64])
    def test_entry_zone_is_light_blue(self, val):
        assert "e6f7ff" in _color_composite(val)

    @pytest.mark.parametrize("val", [0.00, 0.40, 0.54])
    def test_below_threshold_is_empty(self, val):
        assert _color_composite(val) == ""

    def test_exact_boundaries(self):
        assert "ccffcc" in _color_composite(0.65)   # boundary → green
        assert "e6f7ff" in _color_composite(0.55)   # boundary → blue
        assert _color_composite(0.54) == ""          # just below → empty


# ===========================================================================
# 2. Portfolio-tab merge logic (Tab 1)
# ===========================================================================

class TestPortfolioTabMerge:
    """Replicate Tab 1: portfolio ← left-join ← scores (or fallback copy)."""

    _SCORE_COLS = ["ticker", "composite_score", "quality_score",
                   "crowding_score", "physical_norm"]

    def test_merge_adds_signal_columns(self):
        portfolio = _make_portfolio("ETN", "RHM.DE")
        scores    = _make_scores("ETN", "RHM.DE", composite=0.72, crowding=0.25)
        merged = portfolio.merge(scores[self._SCORE_COLS], on="ticker", how="left")
        assert "crowding_score" in merged.columns
        assert "quality_score"  in merged.columns
        assert len(merged) == 2

    def test_merge_keeps_portfolio_rows_when_score_missing(self):
        portfolio = _make_portfolio("ETN", "UNKNOWN")
        scores    = _make_scores("ETN")
        merged = portfolio.merge(scores[self._SCORE_COLS], on="ticker", how="left")
        assert len(merged) == 2
        assert pd.isna(merged.loc[merged["ticker"] == "UNKNOWN", "crowding_score"].iloc[0])

    def test_scores_empty_falls_back_to_portfolio_copy(self):
        """Tab 1 code: if scores.empty: merged = portfolio.copy()"""
        portfolio = _make_portfolio("ETN")
        scores    = pd.DataFrame()
        merged = portfolio.merge(
            scores[self._SCORE_COLS] if not scores.empty else pd.DataFrame(),
            on="ticker" if not scores.empty else None,
            how="left",
        ) if not scores.empty else portfolio.copy()
        assert list(merged["ticker"]) == ["ETN"]
        # Signal columns are absent when scores were empty
        assert "crowding_score" not in merged.columns

    def test_sort_by_composite_desc(self):
        # Drop composite_score from portfolio to avoid merge suffix collision;
        # the scores slice provides the authoritative composite_score column.
        portfolio = _make_portfolio("A", "B", "C").drop(columns=["composite_score"])
        scores_df = pd.DataFrame([
            {"ticker": "A", "composite_score": 0.80, "quality_score": 0.5,
             "crowding_score": 0.2, "physical_norm": 0.7},
            {"ticker": "B", "composite_score": 0.55, "quality_score": 0.4,
             "crowding_score": 0.3, "physical_norm": 0.6},
            {"ticker": "C", "composite_score": 0.68, "quality_score": 0.6,
             "crowding_score": 0.1, "physical_norm": 0.8},
        ])
        merged    = portfolio.merge(scores_df, on="ticker", how="left")
        assert "composite_score" in merged.columns  # no suffix collision
        sorted_df = merged.sort_values("composite_score", ascending=False)
        assert sorted_df["ticker"].tolist() == ["A", "C", "B"]

    def test_display_columns_filtered_to_present_only(self):
        """Tab 1 filters display_cols to those actually in merged."""
        portfolio  = _make_portfolio("ETN").drop(columns=["composite_score"])
        scores_df  = pd.DataFrame([{
            "ticker": "ETN", "composite_score": 0.70,
            "quality_score": 0.55, "crowding_score": 0.30, "physical_norm": 0.65,
        }])
        merged      = portfolio.merge(scores_df, on="ticker", how="left")
        desired     = ["ticker", "weight", "composite_score", "quality_score", "crowding_score"]
        display_col = [c for c in desired if c in merged.columns]
        assert set(display_col) == set(desired)

    def test_weight_formatted_as_percentage(self):
        portfolio = _make_portfolio("ETN")
        portfolio["weight"] = 0.075
        portfolio["weight"] = portfolio["weight"].map(lambda x: f"{x:.1%}")
        assert portfolio["weight"].iloc[0] == "7.5%"

    def test_bucket_col_prefers_primary_bucket(self):
        """Tab 1 uses 'primary_bucket' if present, else falls back to 'bucket_id'."""
        port_with_primary = _make_portfolio("ETN")
        port_with_primary["primary_bucket"] = "grid"
        bucket_col = ("primary_bucket" if "primary_bucket" in port_with_primary.columns
                      else "bucket_id")
        assert bucket_col == "primary_bucket"

    def test_bucket_col_falls_back_to_bucket_id(self):
        portfolio  = _make_portfolio("ETN")   # has bucket_id, not primary_bucket
        bucket_col = ("primary_bucket" if "primary_bucket" in portfolio.columns
                      else "bucket_id")
        assert bucket_col == "bucket_id"

    def test_bucket_aggregation(self):
        portfolio = pd.DataFrame([
            {"ticker": "ETN",  "weight": 0.10, "bucket_id": "grid"},
            {"ticker": "NVDA", "weight": 0.08, "bucket_id": "ai_infra"},
            {"ticker": "CCJ",  "weight": 0.06, "bucket_id": "grid"},
        ])
        bucket_agg = portfolio.groupby("bucket_id")["weight"].sum()
        assert pytest.approx(bucket_agg["grid"],     abs=1e-6) == 0.16
        assert pytest.approx(bucket_agg["ai_infra"], abs=1e-6) == 0.08


# ===========================================================================
# 3. Universe-scanner filter + sort logic (Tab 3)
# ===========================================================================

class TestUniverseScannerFilter:
    """Replicate Tab 3 merge → filter → sort pipeline."""

    def _build_scanner_df(self):
        scores   = _make_scores("ETN", "RHM.DE", "CCJ", "AWK", "NVDA")
        universe = _make_universe()
        return scores.merge(
            universe[["ticker", "region", "primary_bucket"]],
            on="ticker", how="left",
        )

    def test_region_filter_us_only(self):
        filtered = self._build_scanner_df()
        filtered = filtered[filtered["region"].isin(["US"])]
        assert set(filtered["ticker"]) == {"ETN", "CCJ", "AWK", "NVDA"}

    def test_region_filter_eu_only(self):
        filtered = self._build_scanner_df()[lambda df: df["region"].isin(["EU"])]
        assert set(filtered["ticker"]) == {"RHM.DE"}

    def test_bucket_filter_grid_only(self):
        filtered = self._build_scanner_df()[lambda df: df["primary_bucket"].isin(["grid"])]
        assert set(filtered["ticker"]) == {"ETN"}

    def test_combined_region_and_bucket_filter(self):
        df = self._build_scanner_df()
        filtered = df[df["region"].isin(["US"]) & df["primary_bucket"].isin(["nuclear", "water"])]
        assert set(filtered["ticker"]) == {"CCJ", "AWK"}

    def test_no_filter_returns_all(self):
        df         = self._build_scanner_df()
        all_regions = ["EU", "US", "CA"]
        all_buckets = ["grid", "nuclear", "defense", "water", "critical_materials", "ai_infra"]
        filtered   = df[df["region"].isin(all_regions) & df["primary_bucket"].isin(all_buckets)]
        assert len(filtered) == 5

    def test_sort_by_composite_descending(self):
        scores = pd.DataFrame([
            {"ticker": "ETN",  "composite_score": 0.80, "region": "US", "primary_bucket": "grid"},
            {"ticker": "NVDA", "composite_score": 0.62, "region": "US", "primary_bucket": "ai_infra"},
            {"ticker": "CCJ",  "composite_score": 0.71, "region": "US", "primary_bucket": "nuclear"},
        ])
        sorted_df = scores.sort_values("composite_score", ascending=False)
        assert sorted_df["ticker"].tolist() == ["ETN", "CCJ", "NVDA"]

    def test_display_columns_all_present_after_merge(self):
        merged  = self._build_scanner_df()
        desired = ["ticker", "region", "primary_bucket",
                   "composite_score", "physical_norm",
                   "quality_score", "crowding_score", "entry_signal"]
        present = [c for c in desired if c in merged.columns]
        assert set(present) == set(desired)

    def test_ticker_missing_from_universe_has_nan_bucket(self):
        scores  = _make_scores("GHOST_TICKER")
        merged  = scores.merge(_make_universe()[["ticker", "region", "primary_bucket"]],
                               on="ticker", how="left")
        assert pd.isna(merged["primary_bucket"].iloc[0])


# ===========================================================================
# 4. Exit-monitor threshold / status logic (Tab 4)
# ===========================================================================

class TestExitMonitorLogic:
    """Replicate the status_label and alert-count logic from Tab 4."""

    CROWD_EXIT = 0.75   # matches dashboard hard-coded value
    WATCH_THR  = 0.55   # matches dashboard hard-coded value

    def _status(self, crowd):
        """Mirror the status_label closure in run_dashboard()."""
        if pd.isna(crowd):
            return "⚪ No data"
        if crowd >= self.CROWD_EXIT:
            return "🔴 EXIT"
        if crowd >= self.WATCH_THR:
            return "🟡 WATCH"
        return "🟢 OK"

    # --- status_label parametrized ---

    @pytest.mark.parametrize("crowd,expected", [
        (0.75, "🔴 EXIT"),
        (0.90, "🔴 EXIT"),
        (1.00, "🔴 EXIT"),
        (0.55, "🟡 WATCH"),
        (0.74, "🟡 WATCH"),
        (0.00, "🟢 OK"),
        (0.30, "🟢 OK"),
        (0.54, "🟢 OK"),
    ])
    def test_status_label(self, crowd, expected):
        assert self._status(crowd) == expected

    def test_nan_crowding_returns_no_data(self):
        assert self._status(float("nan")) == "⚪ No data"

    # --- alert counts ---

    def test_red_count_correct(self):
        scores = _make_scores("A", "B", "C", "D")
        scores.loc[scores["ticker"].isin(["A", "B"]), "crowding_score"] = 0.80
        scores.loc[scores["ticker"].isin(["C"]),      "crowding_score"] = 0.60
        scores.loc[scores["ticker"].isin(["D"]),      "crowding_score"] = 0.20
        assert (scores["crowding_score"] >= self.CROWD_EXIT).sum() == 2

    def test_yellow_count_correct(self):
        scores = _make_scores("A", "B", "C", "D")
        scores.loc[scores["ticker"].isin(["A"]),      "crowding_score"] = 0.80
        scores.loc[scores["ticker"].isin(["B", "C"]), "crowding_score"] = 0.60
        scores.loc[scores["ticker"].isin(["D"]),      "crowding_score"] = 0.20
        n_yel = (
            (scores["crowding_score"] >= self.WATCH_THR) &
            (scores["crowding_score"] <  self.CROWD_EXIT)
        ).sum()
        assert n_yel == 2

    def test_no_alerts_when_all_ok(self):
        scores = _make_scores("A", "B", "C", crowding=0.30)
        assert (scores["crowding_score"] >= self.CROWD_EXIT).sum() == 0
        assert (
            (scores["crowding_score"] >= self.WATCH_THR) &
            (scores["crowding_score"] <  self.CROWD_EXIT)
        ).sum() == 0

    def test_portfolio_filter_to_held_tickers(self):
        portfolio = _make_portfolio("ETN", "NVDA")
        scores    = _make_scores("ETN", "NVDA", "CCJ", "AWK")
        port_scores = scores[scores["ticker"].isin(portfolio["ticker"].tolist())]
        assert set(port_scores["ticker"]) == {"ETN", "NVDA"}
        assert len(port_scores) == 2

    def test_status_column_applied_to_each_row(self):
        scores = _make_scores("A", "B", "C")
        scores.loc[scores["ticker"] == "A", "crowding_score"] = 0.80
        scores.loc[scores["ticker"] == "B", "crowding_score"] = 0.60
        scores.loc[scores["ticker"] == "C", "crowding_score"] = 0.20
        scores["status"] = scores["crowding_score"].apply(self._status)
        assert scores.loc[scores["ticker"] == "A", "status"].iloc[0] == "🔴 EXIT"
        assert scores.loc[scores["ticker"] == "B", "status"].iloc[0] == "🟡 WATCH"
        assert scores.loc[scores["ticker"] == "C", "status"].iloc[0] == "🟢 OK"

    def test_exit_monitor_display_columns(self):
        """Tab 4 restricts display to exactly these 5 columns."""
        scores = _make_scores("ETN", "NVDA")
        scores["status"] = scores["crowding_score"].apply(self._status)
        expected = ["ticker", "crowding_score", "quality_score", "composite_score", "status"]
        display  = scores[expected]
        assert list(display.columns) == expected

    def test_sort_by_crowding_descending(self):
        scores = _make_scores("A", "B", "C")
        scores.loc[scores["ticker"] == "A", "crowding_score"] = 0.20
        scores.loc[scores["ticker"] == "B", "crowding_score"] = 0.80
        scores.loc[scores["ticker"] == "C", "crowding_score"] = 0.55
        sorted_df = scores.sort_values("crowding_score", ascending=False)
        assert sorted_df["ticker"].tolist() == ["B", "C", "A"]

    def test_empty_portfolio_produces_no_alerts(self):
        portfolio   = pd.DataFrame()
        scores      = _make_scores("ETN", "NVDA")
        port_tickers = [] if portfolio.empty else portfolio["ticker"].tolist()
        port_scores  = scores[scores["ticker"].isin(port_tickers)]
        assert port_scores.empty
