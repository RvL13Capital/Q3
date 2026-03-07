"""
Tests for dashboard query / transformation logic.

streamlit is not installed in the test environment, so we stub it out
before importing anything from src.reporting.dashboard.
The tests cover:
  - _color_crowding / _color_composite helper functions
  - Portfolio-tab merge logic
  - Universe-scanner filter + sort logic
  - Exit-monitor status thresholds and alert counts
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
    # Anything accessed on the stub returns a MagicMock so @st.cache_data etc. work
    st.__getattr__ = lambda self, _: MagicMock()
    for attr in [
        "cache_resource", "cache_data", "set_page_config", "title", "tabs",
        "columns", "metric", "dataframe", "subheader", "table", "info",
        "selectbox", "multiselect", "bar_chart", "error", "warning",
    ]:
        setattr(st, attr, MagicMock(return_value=MagicMock()))
    # cache decorators must return the function unchanged
    st.cache_resource = lambda f=None, **kw: (f if f else lambda fn: fn)
    st.cache_data = lambda f=None, ttl=None, **kw: (f if f else lambda fn: fn)
    return st


sys.modules.setdefault("streamlit", _make_streamlit_stub())

# Now we can safely import from the dashboard
from src.reporting.dashboard import _color_crowding, _color_composite  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers shared across test sections
# ---------------------------------------------------------------------------

def _make_scores(*tickers, composite=0.60, quality=0.50, crowding=0.30,
                 physical=0.70, entry=True) -> pd.DataFrame:
    """Build a minimal signal_scores-like DataFrame."""
    rows = []
    for t in tickers:
        rows.append({
            "ticker":            t,
            "score_date":        date(2025, 3, 1),
            "composite_score":   composite,
            "quality_score":     quality,
            "crowding_score":    crowding,
            "physical_norm":     physical,
            "entry_signal":      entry,
            "roic_wacc_spread":  0.08,
            "margin_snr":        4.5,
            "inflation_convexity": 0.03,
            "etf_correlation":   0.25,
            "trends_norm":       0.40,
            "short_pct":         0.05,
            "mu_estimate":       0.12,
            "sigma_estimate":    0.20,
            "kelly_25pct":       0.06,
        })
    return pd.DataFrame(rows)


def _make_portfolio(*tickers, weight=0.08) -> pd.DataFrame:
    rows = []
    for t in tickers:
        rows.append({
            "ticker":          t,
            "weight":          weight,
            "snapshot_date":   date(2025, 3, 1),
            "composite_score": 0.60,
            "kelly_25pct":     0.06,
            "bucket_id":       "grid",
            "is_new_position": True,
            "rationale":       "test",
        })
    return pd.DataFrame(rows)


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

    def test_nan_returns_empty_string(self):
        assert _color_crowding(float("nan")) == ""
        assert _color_crowding(pd.NA) == ""

    def test_above_exit_threshold_is_red(self):
        for val in (0.75, 0.80, 1.00):
            assert "ffcccc" in _color_crowding(val), f"expected red for {val}"

    def test_watch_zone_is_yellow(self):
        for val in (0.55, 0.60, 0.74):
            assert "fff3cc" in _color_crowding(val), f"expected yellow for {val}"

    def test_below_watch_threshold_is_green(self):
        for val in (0.00, 0.30, 0.54):
            assert "ccffcc" in _color_crowding(val), f"expected green for {val}"

    def test_exact_boundaries(self):
        # 0.75 → red, 0.55 → yellow, 0.54 → green
        assert "ffcccc" in _color_crowding(0.75)
        assert "fff3cc" in _color_crowding(0.55)
        assert "ccffcc" in _color_crowding(0.54)


class TestColorComposite:
    """_color_composite: green ≥ 0.65, light-blue ≥ 0.55, empty otherwise."""

    def test_nan_returns_empty_string(self):
        assert _color_composite(float("nan")) == ""
        assert _color_composite(pd.NA) == ""

    def test_strong_signal_is_green(self):
        for val in (0.65, 0.80, 1.00):
            assert "ccffcc" in _color_composite(val), f"expected green for {val}"

    def test_entry_zone_is_light_blue(self):
        for val in (0.55, 0.60, 0.64):
            assert "e6f7ff" in _color_composite(val), f"expected blue for {val}"

    def test_below_threshold_is_empty(self):
        for val in (0.00, 0.40, 0.54):
            assert _color_composite(val) == "", f"expected empty for {val}"

    def test_exact_boundaries(self):
        assert "ccffcc" in _color_composite(0.65)
        assert "e6f7ff" in _color_composite(0.55)
        assert _color_composite(0.54) == ""


# ===========================================================================
# 2. Portfolio-tab merge logic
# ===========================================================================

class TestPortfolioTabMerge:
    """Replicate Tab 1 merge: portfolio ← left-join ← scores."""

    def test_merge_adds_signal_columns(self):
        portfolio = _make_portfolio("ETN", "RHM.DE")
        scores    = _make_scores("ETN", "RHM.DE", composite=0.72, crowding=0.25)

        merged = portfolio.merge(
            scores[["ticker", "composite_score", "quality_score",
                    "crowding_score", "physical_norm"]],
            on="ticker", how="left",
        )
        # composite_score from portfolio is overwritten by scores; rename handled by suffix
        assert "crowding_score" in merged.columns
        assert "quality_score"  in merged.columns
        assert len(merged) == 2

    def test_merge_keeps_all_portfolio_rows_when_score_missing(self):
        portfolio = _make_portfolio("ETN", "UNKNOWN")
        scores    = _make_scores("ETN")

        merged = portfolio.merge(
            scores[["ticker", "composite_score", "quality_score",
                    "crowding_score", "physical_norm"]],
            on="ticker", how="left",
        )
        assert len(merged) == 2  # UNKNOWN retained with NaN signals
        unknown_row = merged[merged["ticker"] == "UNKNOWN"]
        assert pd.isna(unknown_row["crowding_score"].iloc[0])

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
        merged = portfolio.merge(scores_df, on="ticker", how="left")
        assert "composite_score" in merged.columns   # no suffix collision
        sorted_df = merged.sort_values("composite_score", ascending=False)
        assert sorted_df["ticker"].tolist() == ["A", "C", "B"]

    def test_weight_formatted_as_percentage(self):
        portfolio = _make_portfolio("ETN")
        portfolio["weight"] = 0.075
        portfolio["weight"] = portfolio["weight"].map(lambda x: f"{x:.1%}")
        assert portfolio["weight"].iloc[0] == "7.5%"

    def test_empty_portfolio_handled(self):
        portfolio = pd.DataFrame()
        assert portfolio.empty

    def test_bucket_aggregation(self):
        portfolio = pd.DataFrame([
            {"ticker": "ETN",    "weight": 0.10, "bucket_id": "grid"},
            {"ticker": "NVDA",   "weight": 0.08, "bucket_id": "ai_infra"},
            {"ticker": "CCJ",    "weight": 0.06, "bucket_id": "grid"},
        ])
        bucket_agg = portfolio.groupby("bucket_id")["weight"].sum()
        assert pytest.approx(bucket_agg["grid"],     abs=1e-6) == 0.16
        assert pytest.approx(bucket_agg["ai_infra"], abs=1e-6) == 0.08


# ===========================================================================
# 3. Universe-scanner filter + sort logic
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
        merged   = self._build_scanner_df()
        filtered = merged[merged["region"].isin(["US"])]
        assert set(filtered["ticker"]) == {"ETN", "CCJ", "AWK", "NVDA"}

    def test_region_filter_eu_only(self):
        merged   = self._build_scanner_df()
        filtered = merged[merged["region"].isin(["EU"])]
        assert set(filtered["ticker"]) == {"RHM.DE"}

    def test_bucket_filter_grid_only(self):
        merged   = self._build_scanner_df()
        filtered = merged[merged["primary_bucket"].isin(["grid"])]
        assert set(filtered["ticker"]) == {"ETN"}

    def test_combined_region_and_bucket_filter(self):
        merged   = self._build_scanner_df()
        filtered = merged[
            merged["region"].isin(["US"]) &
            merged["primary_bucket"].isin(["nuclear", "water"])
        ]
        assert set(filtered["ticker"]) == {"CCJ", "AWK"}

    def test_no_filter_returns_all(self):
        merged   = self._build_scanner_df()
        all_regions  = ["EU", "US", "CA"]
        all_buckets  = ["grid", "nuclear", "defense", "water",
                        "critical_materials", "ai_infra"]
        filtered = merged[
            merged["region"].isin(all_regions) &
            merged["primary_bucket"].isin(all_buckets)
        ]
        assert len(filtered) == 5

    def test_sort_by_composite_descending(self):
        scores = pd.DataFrame([
            {"ticker": "ETN",    "composite_score": 0.80,
             "region": "US", "primary_bucket": "grid"},
            {"ticker": "NVDA",   "composite_score": 0.62,
             "region": "US", "primary_bucket": "ai_infra"},
            {"ticker": "CCJ",    "composite_score": 0.71,
             "region": "US", "primary_bucket": "nuclear"},
        ])
        sorted_df = scores.sort_values("composite_score", ascending=False)
        assert sorted_df["ticker"].tolist() == ["ETN", "CCJ", "NVDA"]

    def test_display_columns_subset(self):
        merged = self._build_scanner_df()
        desired = ["ticker", "region", "primary_bucket",
                   "composite_score", "physical_norm",
                   "quality_score", "crowding_score", "entry_signal"]
        display_cols = [c for c in desired if c in merged.columns]
        assert set(display_cols) == set(desired)   # all present in merged

    def test_ticker_missing_from_universe_has_nan_bucket(self):
        scores   = _make_scores("GHOST_TICKER")
        universe = _make_universe()
        merged   = scores.merge(
            universe[["ticker", "region", "primary_bucket"]],
            on="ticker", how="left",
        )
        assert pd.isna(merged["primary_bucket"].iloc[0])


# ===========================================================================
# 4. Exit-monitor threshold / status logic
# ===========================================================================

class TestExitMonitorLogic:
    """Replicate the status_label and alert-count logic from Tab 4."""

    CROWD_EXIT = 0.75
    WATCH_THR  = 0.55

    def _status(self, crowd):
        """Mirror the status_label closure in run_dashboard()."""
        if pd.isna(crowd):
            return "⚪ No data"
        if crowd >= self.CROWD_EXIT:
            return "🔴 EXIT"
        if crowd >= self.WATCH_THR:
            return "🟡 WATCH"
        return "🟢 OK"

    # --- status_label ---

    def test_exit_threshold_exact(self):
        assert self._status(0.75) == "🔴 EXIT"

    def test_exit_above_threshold(self):
        assert self._status(0.90) == "🔴 EXIT"

    def test_watch_zone_lower_bound(self):
        assert self._status(0.55) == "🟡 WATCH"

    def test_watch_zone_upper_bound(self):
        assert self._status(0.74) == "🟡 WATCH"

    def test_ok_zone(self):
        assert self._status(0.30) == "🟢 OK"
        assert self._status(0.54) == "🟢 OK"
        assert self._status(0.00) == "🟢 OK"

    def test_nan_crowding(self):
        assert self._status(float("nan")) == "⚪ No data"

    # --- alert counts ---

    def test_red_count_correct(self):
        scores = _make_scores("A", "B", "C", "D")
        scores.loc[scores["ticker"].isin(["A", "B"]), "crowding_score"] = 0.80
        scores.loc[scores["ticker"].isin(["C"]),      "crowding_score"] = 0.60
        scores.loc[scores["ticker"].isin(["D"]),      "crowding_score"] = 0.20
        n_red = (scores["crowding_score"] >= self.CROWD_EXIT).sum()
        assert n_red == 2

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
        n_red = (scores["crowding_score"] >= self.CROWD_EXIT).sum()
        n_yel = (
            (scores["crowding_score"] >= self.WATCH_THR) &
            (scores["crowding_score"] <  self.CROWD_EXIT)
        ).sum()
        assert n_red == 0
        assert n_yel == 0

    def test_portfolio_filter_to_held_tickers(self):
        portfolio = _make_portfolio("ETN", "NVDA")
        scores    = _make_scores("ETN", "NVDA", "CCJ", "AWK")
        port_tickers = portfolio["ticker"].tolist()
        port_scores  = scores[scores["ticker"].isin(port_tickers)]
        assert set(port_scores["ticker"]) == {"ETN", "NVDA"}
        assert len(port_scores) == 2

    def test_status_applied_to_each_row(self):
        scores = _make_scores("A", "B", "C")
        scores.loc[scores["ticker"] == "A", "crowding_score"] = 0.80
        scores.loc[scores["ticker"] == "B", "crowding_score"] = 0.60
        scores.loc[scores["ticker"] == "C", "crowding_score"] = 0.20
        scores["status"] = scores["crowding_score"].apply(self._status)
        assert scores.loc[scores["ticker"] == "A", "status"].iloc[0] == "🔴 EXIT"
        assert scores.loc[scores["ticker"] == "B", "status"].iloc[0] == "🟡 WATCH"
        assert scores.loc[scores["ticker"] == "C", "status"].iloc[0] == "🟢 OK"

    def test_sort_by_crowding_descending(self):
        scores = _make_scores("A", "B", "C")
        scores.loc[scores["ticker"] == "A", "crowding_score"] = 0.20
        scores.loc[scores["ticker"] == "B", "crowding_score"] = 0.80
        scores.loc[scores["ticker"] == "C", "crowding_score"] = 0.55
        sorted_df = scores.sort_values("crowding_score", ascending=False)
        assert sorted_df["ticker"].tolist() == ["B", "C", "A"]

    def test_empty_portfolio_produces_no_alerts(self):
        portfolio = pd.DataFrame()
        scores    = _make_scores("ETN", "NVDA")
        if portfolio.empty:
            port_tickers = []
        else:
            port_tickers = portfolio["ticker"].tolist()
        port_scores = scores[scores["ticker"].isin(port_tickers)]
        assert port_scores.empty
