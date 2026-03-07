"""
Tests for dashboard query / transformation logic.

streamlit is not installed in the test environment, so we stub it out
before importing anything from src.reporting.dashboard.
The tests cover:
  - _color_crowding / _color_composite styling helpers
  - _fmt_sizing helper (Tab 2)
  - status_label (Tab 4, now module-level)
  - build_portfolio_display (Tab 1 pure data function)
  - build_scanner_display (Tab 3 pure data function)
  - build_exit_monitor_display (Tab 4 pure data function)
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

from src.reporting.dashboard import (  # noqa: E402
    _color_crowding,
    _color_composite,
    _fmt_sizing,
    status_label,
    build_portfolio_display,
    build_scanner_display,
    build_exit_monitor_display,
    _CROWD_EXIT,
    _WATCH_THR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scores(*tickers, composite=0.60, quality=0.50, crowding=0.30,
                 physical=0.70, entry=True) -> pd.DataFrame:
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
# 1. Color styling helpers
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


# ===========================================================================
# 2. _fmt_sizing (Tab 2 sizing table)
# ===========================================================================

class TestFmtSizing:
    """_fmt_sizing: formats as percentage; "—" for None/NaN; 0.0 is valid."""

    @pytest.mark.parametrize("val,expected", [
        (0.12,  "12.0%"),
        (0.05,  "5.0%"),
        (0.003, "0.3%"),
    ])
    def test_positive_value_formats_as_pct(self, val, expected):
        assert _fmt_sizing(val) == expected

    def test_zero_formats_as_zero_pct(self):
        # 0.0 is a valid value — must NOT render as "—"
        assert _fmt_sizing(0.0) == "0.0%"

    @pytest.mark.parametrize("val", [None, float("nan")])
    def test_missing_returns_dash(self, val):
        assert _fmt_sizing(val) == "—"


# ===========================================================================
# 3. status_label (Tab 4, now module-level)
# ===========================================================================

class TestStatusLabel:
    """status_label uses module constants by default; accepts override."""

    @pytest.mark.parametrize("crowd,expected", [
        (_CROWD_EXIT,        "🔴 EXIT"),
        (_CROWD_EXIT + 0.10, "🔴 EXIT"),
        (1.00,               "🔴 EXIT"),
        (_WATCH_THR,         "🟡 WATCH"),
        (_CROWD_EXIT - 0.01, "🟡 WATCH"),
        (0.00,               "🟢 OK"),
        (0.30,               "🟢 OK"),
        (_WATCH_THR - 0.01,  "🟢 OK"),
    ])
    def test_default_thresholds(self, crowd, expected):
        assert status_label(crowd) == expected

    def test_nan_returns_no_data(self):
        assert status_label(float("nan")) == "⚪ No data"

    def test_custom_thresholds(self):
        assert status_label(0.65, crowd_exit=0.60, watch_thr=0.40) == "🔴 EXIT"
        assert status_label(0.45, crowd_exit=0.60, watch_thr=0.40) == "🟡 WATCH"
        assert status_label(0.35, crowd_exit=0.60, watch_thr=0.40) == "🟢 OK"


# ===========================================================================
# 4. build_portfolio_display (Tab 1)
# ===========================================================================

class TestBuildPortfolioDisplay:
    """build_portfolio_display merges scores into portfolio with no column collision."""

    def test_signal_columns_present_after_merge(self):
        df = build_portfolio_display(_make_portfolio("ETN"), _make_scores("ETN"))
        assert "crowding_score" in df.columns
        assert "quality_score"  in df.columns

    def test_no_composite_score_suffix_collision(self):
        # portfolio has composite_score; scores also has it — must produce single column
        df = build_portfolio_display(_make_portfolio("ETN"), _make_scores("ETN"))
        assert "composite_score"   in df.columns
        assert "composite_score_x" not in df.columns
        assert "composite_score_y" not in df.columns

    def test_sorted_by_composite_desc(self):
        scores = pd.DataFrame([
            {"ticker": "A", "composite_score": 0.80, "quality_score": 0.5,
             "crowding_score": 0.2, "physical_norm": 0.7},
            {"ticker": "B", "composite_score": 0.55, "quality_score": 0.4,
             "crowding_score": 0.3, "physical_norm": 0.6},
            {"ticker": "C", "composite_score": 0.68, "quality_score": 0.6,
             "crowding_score": 0.1, "physical_norm": 0.8},
        ])
        df = build_portfolio_display(_make_portfolio("A", "B", "C"), scores)
        assert df["ticker"].tolist() == ["A", "C", "B"]

    def test_index_reset_after_sort(self):
        df = build_portfolio_display(
            _make_portfolio("A", "B"),
            _make_scores("A", "B"),
        )
        assert df.index.tolist() == list(range(len(df)))

    def test_weight_formatted_as_percentage(self):
        df = build_portfolio_display(_make_portfolio("ETN", weight=0.075), _make_scores("ETN"))
        assert df.loc[df["ticker"] == "ETN", "weight"].iloc[0] == "7.5%"

    def test_scores_empty_returns_portfolio_columns_only(self):
        df = build_portfolio_display(_make_portfolio("ETN"), pd.DataFrame())
        assert "ticker" in df.columns
        assert "crowding_score" not in df.columns

    def test_missing_ticker_in_scores_gets_nan_signals(self):
        df = build_portfolio_display(
            _make_portfolio("ETN", "UNKNOWN"),
            _make_scores("ETN"),
        )
        assert pd.isna(df.loc[df["ticker"] == "UNKNOWN", "crowding_score"].iloc[0])

    def test_bucket_aggregation(self):
        portfolio = pd.DataFrame([
            {"ticker": "ETN",  "weight": 0.10, "bucket_id": "grid",
             "composite_score": 0.65, "kelly_25pct": 0.06,
             "snapshot_date": date(2025, 3, 1), "is_new_position": True, "rationale": ""},
            {"ticker": "NVDA", "weight": 0.08, "bucket_id": "ai_infra",
             "composite_score": 0.70, "kelly_25pct": 0.07,
             "snapshot_date": date(2025, 3, 1), "is_new_position": True, "rationale": ""},
            {"ticker": "CCJ",  "weight": 0.06, "bucket_id": "grid",
             "composite_score": 0.60, "kelly_25pct": 0.05,
             "snapshot_date": date(2025, 3, 1), "is_new_position": False, "rationale": ""},
        ])
        agg = portfolio.groupby("bucket_id")["weight"].sum()
        assert pytest.approx(agg["grid"],     abs=1e-6) == 0.16
        assert pytest.approx(agg["ai_infra"], abs=1e-6) == 0.08


# ===========================================================================
# 5. build_scanner_display (Tab 3)
# ===========================================================================

class TestBuildScannerDisplay:
    """build_scanner_display merges scores with universe and sorts by composite."""

    def _df(self):
        return build_scanner_display(
            _make_scores("ETN", "RHM.DE", "CCJ", "AWK", "NVDA"),
            _make_universe(),
        )

    def test_expected_columns_present(self):
        df = self._df()
        for col in ["ticker", "region", "primary_bucket", "composite_score",
                    "physical_norm", "quality_score", "crowding_score", "entry_signal"]:
            assert col in df.columns

    def test_sorted_by_composite_descending(self):
        scores = pd.DataFrame([
            {"ticker": "ETN",  "composite_score": 0.80, "quality_score": 0.5,
             "crowding_score": 0.2, "physical_norm": 0.7, "entry_signal": True,
             "score_date": date(2025, 3, 1), "roic_wacc_spread": 0.08,
             "margin_snr": 4.5, "inflation_convexity": 0.03, "etf_correlation": 0.25,
             "trends_norm": 0.40, "short_pct": 0.05, "mu_estimate": 0.12,
             "sigma_estimate": 0.20, "kelly_25pct": 0.06},
            {"ticker": "CCJ",  "composite_score": 0.71, "quality_score": 0.6,
             "crowding_score": 0.1, "physical_norm": 0.8, "entry_signal": True,
             "score_date": date(2025, 3, 1), "roic_wacc_spread": 0.05,
             "margin_snr": 3.0, "inflation_convexity": 0.02, "etf_correlation": 0.15,
             "trends_norm": 0.35, "short_pct": 0.03, "mu_estimate": 0.10,
             "sigma_estimate": 0.18, "kelly_25pct": 0.05},
        ])
        universe = pd.DataFrame([
            {"ticker": "ETN", "region": "US", "primary_bucket": "grid"},
            {"ticker": "CCJ", "region": "US", "primary_bucket": "nuclear"},
        ])
        df = build_scanner_display(scores, universe)
        assert df["ticker"].tolist() == ["ETN", "CCJ"]

    def test_index_reset(self):
        df = self._df()
        assert df.index.tolist() == list(range(len(df)))

    def test_region_filter_us(self):
        df      = self._df()
        result  = df[df["region"].isin(["US"])]
        assert set(result["ticker"]) == {"ETN", "CCJ", "AWK", "NVDA"}

    def test_region_filter_eu(self):
        df     = self._df()
        result = df[df["region"].isin(["EU"])]
        assert set(result["ticker"]) == {"RHM.DE"}

    def test_bucket_filter(self):
        df     = self._df()
        result = df[df["primary_bucket"].isin(["nuclear", "water"])]
        assert set(result["ticker"]) == {"CCJ", "AWK"}

    def test_combined_filter(self):
        df     = self._df()
        result = df[df["region"].isin(["US"]) & df["primary_bucket"].isin(["grid"])]
        assert set(result["ticker"]) == {"ETN"}

    def test_unknown_ticker_has_nan_bucket(self):
        df = build_scanner_display(_make_scores("GHOST"), _make_universe())
        assert pd.isna(df.loc[df["ticker"] == "GHOST", "primary_bucket"].iloc[0])


# ===========================================================================
# 6. build_exit_monitor_display (Tab 4)
# ===========================================================================

class TestBuildExitMonitorDisplay:
    """build_exit_monitor_display produces status column and correct alert counts."""

    def test_returns_tuple_of_three(self):
        result = build_exit_monitor_display(_make_portfolio("ETN"), _make_scores("ETN"))
        assert len(result) == 3

    def test_status_column_present(self):
        df, _, _ = build_exit_monitor_display(_make_portfolio("ETN"), _make_scores("ETN"))
        assert "status" in df.columns

    def test_display_columns_exact(self):
        df, _, _ = build_exit_monitor_display(_make_portfolio("ETN"), _make_scores("ETN"))
        assert list(df.columns) == [
            "ticker", "crowding_score", "quality_score", "composite_score", "status"
        ]

    def test_sorted_by_crowding_desc(self):
        scores = _make_scores("A", "B", "C")
        scores.loc[scores["ticker"] == "A", "crowding_score"] = 0.20
        scores.loc[scores["ticker"] == "B", "crowding_score"] = 0.80
        scores.loc[scores["ticker"] == "C", "crowding_score"] = 0.55
        df, _, _ = build_exit_monitor_display(_make_portfolio("A", "B", "C"), scores)
        assert df["ticker"].tolist() == ["B", "C", "A"]

    def test_index_reset(self):
        df, _, _ = build_exit_monitor_display(_make_portfolio("ETN"), _make_scores("ETN"))
        assert df.index.tolist() == list(range(len(df)))

    def test_red_count(self):
        scores = _make_scores("A", "B", "C", "D")
        scores.loc[scores["ticker"].isin(["A", "B"]), "crowding_score"] = 0.80
        scores.loc[scores["ticker"].isin(["C"]),      "crowding_score"] = 0.60
        scores.loc[scores["ticker"].isin(["D"]),      "crowding_score"] = 0.20
        _, n_red, _ = build_exit_monitor_display(_make_portfolio("A", "B", "C", "D"), scores)
        assert n_red == 2

    def test_yellow_count(self):
        scores = _make_scores("A", "B", "C", "D")
        scores.loc[scores["ticker"].isin(["A"]),      "crowding_score"] = 0.80
        scores.loc[scores["ticker"].isin(["B", "C"]), "crowding_score"] = 0.60
        scores.loc[scores["ticker"].isin(["D"]),      "crowding_score"] = 0.20
        _, _, n_yel = build_exit_monitor_display(_make_portfolio("A", "B", "C", "D"), scores)
        assert n_yel == 2

    def test_no_alerts_when_all_ok(self):
        _, n_red, n_yel = build_exit_monitor_display(
            _make_portfolio("A", "B"),
            _make_scores("A", "B", crowding=0.30),
        )
        assert n_red == 0
        assert n_yel == 0

    def test_filters_to_held_tickers_only(self):
        portfolio = _make_portfolio("ETN", "NVDA")
        scores    = _make_scores("ETN", "NVDA", "CCJ", "AWK")
        df, _, _  = build_exit_monitor_display(portfolio, scores)
        assert set(df["ticker"]) == {"ETN", "NVDA"}

    def test_empty_portfolio_returns_empty(self):
        df, n_red, n_yel = build_exit_monitor_display(pd.DataFrame(), _make_scores("ETN"))
        assert df.empty
        assert n_red == 0
        assert n_yel == 0

    def test_empty_scores_returns_empty(self):
        df, n_red, n_yel = build_exit_monitor_display(_make_portfolio("ETN"), pd.DataFrame())
        assert df.empty
        assert n_red == 0
        assert n_yel == 0

    def test_ticker_held_but_not_in_scores_returns_empty(self):
        df, n_red, n_yel = build_exit_monitor_display(
            _make_portfolio("ORPHAN"),
            _make_scores("ETN"),  # ORPHAN not scored
        )
        assert df.empty
        assert n_red == 0
        assert n_yel == 0

    def test_custom_thresholds_respected(self):
        scores = _make_scores("A")
        scores.loc[scores["ticker"] == "A", "crowding_score"] = 0.65
        # With default thresholds (0.75/0.55): yellow
        _, n_red_def, n_yel_def = build_exit_monitor_display(
            _make_portfolio("A"), scores
        )
        assert n_red_def == 0 and n_yel_def == 1
        # With tighter exit threshold (0.60): red
        _, n_red_tight, _ = build_exit_monitor_display(
            _make_portfolio("A"), scores, crowd_exit=0.60, watch_thr=0.40
        )
        assert n_red_tight == 1
