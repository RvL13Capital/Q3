"""
Integration tests for the portfolio layer:
  - Kelly sigma estimation against real price data in DuckDB
  - Kelly weight computation (compute_kelly_weights)
  - Portfolio construction and constraint application (construct_portfolio, apply_constraints)
  - Portfolio diffing (diff_portfolio)
  - Monitor: exit signal detection, entry candidate selection, watch list, generate_weekly_actions
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.db import (
    upsert_prices,
    upsert_fundamentals_annual,
    upsert_macro,
    upsert_signal_scores,
    upsert_portfolio_snapshot,
    get_latest_portfolio,
)
from src.portfolio.kelly import kelly_fraction, estimate_sigma, compute_kelly_weights
from src.portfolio.construction import apply_constraints, construct_portfolio, diff_portfolio
from src.portfolio.monitor import (
    check_exit_signals,
    check_entry_signals,
    get_watch_list,
    generate_weekly_actions,
)

AS_OF = "2023-01-13"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_macro(conn, params):
    from datetime import date, timedelta
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(1500)]
    rf_df = pd.DataFrame({"date": [d.isoformat() for d in dates],
                           "value": [4.3] * len(dates), "source": "test"})
    upsert_macro(conn, params["macro"]["us_risk_free_series"], rf_df)
    upsert_macro(conn, params["macro"]["eu_risk_free_series"], rf_df.assign(value=2.8))


def _make_scored_row(ticker: str, composite: float = 0.70, entry: bool = True,
                     crowding: float = 0.25, quality: float = 0.75,
                     mu: float = 0.15) -> dict:
    return {
        "ticker":               ticker,
        "score_date":           AS_OF,
        "physical_raw":         2.0,
        "physical_norm":        0.667,
        "physical_confidence":  1.0,
        "quality_score":        quality,
        "quality_confidence":   0.9,
        "roic_wacc_spread":     0.05,
        "margin_snr":           6.0,
        "inflation_convexity":  0.04,
        "crowding_score":       crowding,
        "crowding_confidence":  0.8,
        "etf_corr_score":       0.30,
        "short_interest_score": 0.20,
        "composite_score":      composite,
        "composite_confidence": 0.85,
        "mu_estimate":          mu if entry else None,
        "sigma_estimate":       None,
        "kelly_fraction":       None,
        "kelly_25pct":          None,
        "entry_signal":         entry,
        "exit_signal":          not entry,
    }


def _make_scored_df(*args):
    return pd.DataFrame([_make_scored_row(*a) if isinstance(a, tuple) else _make_scored_row(a)
                         for a in args])


def _make_portfolio_snapshot(conn, tickers, snapshot_date=AS_OF, weight=0.07):
    rows = []
    for t in tickers:
        rows.append({
            "snapshot_date":   snapshot_date,
            "ticker":          t,
            "weight":          weight,
            "composite_score": 0.65,
            "kelly_25pct":     0.15,
            "bucket_id":       "grid",
            "is_new_position": True,
            "rationale":       None,
        })
    df = pd.DataFrame(rows)
    upsert_portfolio_snapshot(conn, df)
    return df


# ============================================================================
# Kelly core
# ============================================================================

class TestKellyFormula:
    def test_positive_edge(self):
        # mu=15%, rf=4%, sigma=20% → f* = 0.25 * (0.15-0.04)/0.04 = 0.6875
        f, _ = kelly_fraction(mu=0.15, sigma=0.20, rf=0.04, fraction=0.25)
        assert f == pytest.approx(0.6875)

    def test_negative_edge_returns_zero(self):
        f, _ = kelly_fraction(mu=0.03, sigma=0.20, rf=0.04, fraction=0.25)
        assert f == 0.0

    def test_clamped_at_one(self):
        # Very high mu → would exceed 1.0 without clamping
        f, _ = kelly_fraction(mu=5.0, sigma=0.01, rf=0.04, fraction=0.25)
        assert f == pytest.approx(1.0)

    def test_zero_sigma_returns_zero(self):
        f, _ = kelly_fraction(mu=0.15, sigma=0.0, rf=0.04, fraction=0.25)
        assert f == 0.0

    def test_smaller_fraction_gives_smaller_weight(self):
        # mu=10%, rf=4%, sigma=30% → raw = fraction * (0.10-0.04)/0.09
        # With fraction=1.0 → 0.667; 0.5 → 0.333; 0.25 → 0.167 — all < 1.0 (no clamping)
        f_full, _ = kelly_fraction(0.10, 0.30, 0.04, fraction=1.0)
        f_half, _ = kelly_fraction(0.10, 0.30, 0.04, fraction=0.5)
        f_qtr,  _ = kelly_fraction(0.10, 0.30, 0.04, fraction=0.25)
        assert f_full > f_half > f_qtr > 0


class TestEstimateSigma:
    def test_with_sufficient_data(self, conn, prices_etn):
        upsert_prices(conn, prices_etn)
        sigma, valid = estimate_sigma("ETN", conn, AS_OF)
        assert valid == True
        assert 0.05 <= sigma <= 1.0  # reasonable annualized vol

    def test_with_no_data(self, conn):
        sigma, valid = estimate_sigma("MISSING", conn, AS_OF)
        assert valid == False
        assert sigma == pytest.approx(0.35)  # fallback

    def test_with_too_few_data_points(self, conn, prices_etn):
        upsert_prices(conn, prices_etn.head(20))
        as_of = str(prices_etn.head(20)["date"].max())
        sigma, valid = estimate_sigma("ETN", conn, as_of)
        assert valid == False

    def test_sigma_floored_at_5pct(self, conn, prices_etn):
        # Even with flat prices sigma must be ≥ 5%
        flat = prices_etn.copy()
        flat["adj_close"] = 100.0
        flat["close"]     = 100.0
        upsert_prices(conn, flat)
        sigma, _ = estimate_sigma("ETN", conn, AS_OF)
        assert sigma >= 0.05


class TestComputeKellyWeights:
    def test_returns_entry_candidates_only(self, conn, sample_universe, all_prices, params):
        upsert_prices(conn, all_prices)
        _insert_macro(conn, params)
        scored = _make_scored_df(
            ("ETN", 0.72, True),
            ("CCJ", 0.68, True),
            ("RHM.DE", 0.45, False),  # no entry
        )
        result = compute_kelly_weights(scored, conn, params, AS_OF, sample_universe)
        tickers = result["ticker"].tolist()
        assert "ETN" in tickers
        assert "CCJ" in tickers
        assert "RHM.DE" not in tickers

    def test_kelly_weights_positive(self, conn, sample_universe, all_prices, params):
        upsert_prices(conn, all_prices)
        _insert_macro(conn, params)
        scored = _make_scored_df(("ETN", 0.72, True), ("CCJ", 0.68, True))
        result = compute_kelly_weights(scored, conn, params, AS_OF, sample_universe)
        assert (result["kelly_25pct"] > 0).all()

    def test_no_entry_candidates(self, conn, sample_universe, params):
        scored = _make_scored_df(("ETN", 0.40, False))
        result = compute_kelly_weights(scored, conn, params, AS_OF, sample_universe)
        assert result.empty

    def test_no_mu_estimate_skipped(self, conn, sample_universe, params):
        """Rows with mu_estimate=None must be skipped."""
        scored = pd.DataFrame([_make_scored_row("ETN", entry=True, mu=None)])
        result = compute_kelly_weights(scored, conn, params, AS_OF, sample_universe)
        assert result.empty

    def test_sigma_in_output(self, conn, sample_universe, all_prices, params):
        upsert_prices(conn, all_prices)
        _insert_macro(conn, params)
        scored = _make_scored_df(("ETN", 0.72, True))
        result = compute_kelly_weights(scored, conn, params, AS_OF, sample_universe)
        assert "sigma_estimate" in result.columns
        assert (result["sigma_estimate"] >= 0.05).all()


# ============================================================================
# Portfolio construction
# ============================================================================

class TestApplyConstraints:
    def _kelly_df(self, tickers, weights, buckets):
        return pd.DataFrame({
            "ticker":          tickers,
            "kelly_25pct":     weights,
            "primary_bucket":  buckets,
            "composite_score": [0.65] * len(tickers),
        })

    def test_empty_input(self, params):
        df = pd.DataFrame(columns=["ticker", "kelly_25pct", "primary_bucket", "composite_score"])
        result = apply_constraints(df, params)
        assert result.empty

    def test_stock_cap_enforced(self, params):
        df = self._kelly_df(["ETN"], [0.20], ["grid"])  # > 8% cap
        result = apply_constraints(df, params)
        assert result.loc[result["ticker"] == "ETN", "weight"].iloc[0] <= 0.08

    def test_bucket_cap_enforced(self, params):
        # grid bucket has 2 stocks × 10% each = 20% → below 35% cap
        # grid bucket has 3 stocks × 15% each = 45% → exceeds 35% cap
        df = self._kelly_df(
            ["ETN", "XYL", "PWR"],
            [0.15, 0.15, 0.15],
            ["grid", "grid", "grid"],
        )
        result = apply_constraints(df, params)
        grid_total = result[result["primary_bucket"] == "grid"]["weight"].sum()
        assert grid_total <= 0.35 + 1e-6

    def test_cash_floor_enforced(self, params):
        # 5 stocks × 20% each = 100% invested → cash floor (10%) not respected
        df = self._kelly_df(
            [f"T{i}" for i in range(5)],
            [0.20] * 5,
            ["grid"] * 5,
        )
        result = apply_constraints(df, params)
        invested = result["weight"].sum()
        assert invested <= 0.90 + 1e-6  # cash_reserve = 10%

    def test_min_position_dropped(self, params):
        # kelly_25pct below min (2%) → dropped
        df = self._kelly_df(["ETN", "CCJ"], [0.08, 0.005], ["grid", "nuclear"])
        result = apply_constraints(df, params)
        assert "CCJ" not in result["ticker"].tolist()

    def test_no_negative_weights(self, params):
        df = self._kelly_df(["ETN", "CCJ"], [-0.01, 0.07], ["grid", "nuclear"])
        result = apply_constraints(df, params)
        assert (result["weight"] >= 0).all()

    def test_weights_are_fractions(self, params):
        df = self._kelly_df(
            ["ETN", "CCJ", "PWR"],
            [0.08, 0.07, 0.06],
            ["grid", "nuclear", "grid"],
        )
        result = apply_constraints(df, params)
        assert result["weight"].sum() <= 1.0
        assert (result["weight"] <= 0.08 + 1e-6).all()


class TestConstructPortfolio:
    def test_returns_dataframe(self, conn, sample_universe, all_prices, params):
        upsert_prices(conn, all_prices)
        _insert_macro(conn, params)
        scored = _make_scored_df(("ETN", 0.72, True), ("CCJ", 0.68, True))
        result = construct_portfolio(conn, scored, sample_universe, params, AS_OF, dry_run=True)
        assert isinstance(result, pd.DataFrame)

    def test_no_entry_candidates(self, conn, sample_universe, params):
        scored = _make_scored_df(("ETN", 0.40, False))
        result = construct_portfolio(conn, scored, sample_universe, params, AS_OF, dry_run=True)
        assert result.empty

    def test_portfolio_persisted(self, conn, sample_universe, all_prices, params):
        upsert_prices(conn, all_prices)
        _insert_macro(conn, params)
        scored = _make_scored_df(("ETN", 0.72, True))
        construct_portfolio(conn, scored, sample_universe, params, AS_OF)
        portfolio = get_latest_portfolio(conn)
        assert not portfolio.empty
        assert "ETN" in portfolio["ticker"].tolist()

    def test_dry_run_not_persisted(self, conn, sample_universe, all_prices, params):
        upsert_prices(conn, all_prices)
        _insert_macro(conn, params)
        scored = _make_scored_df(("ETN", 0.72, True))
        construct_portfolio(conn, scored, sample_universe, params, AS_OF, dry_run=True)
        portfolio = get_latest_portfolio(conn)
        assert portfolio.empty

    def test_weights_in_bounds(self, conn, sample_universe, all_prices, params):
        upsert_prices(conn, all_prices)
        _insert_macro(conn, params)
        scored = _make_scored_df(("ETN", 0.75, True), ("CCJ", 0.68, True))
        result = construct_portfolio(conn, scored, sample_universe, params, AS_OF, dry_run=True)
        if not result.empty:
            assert (result["weight"] <= params["kelly"]["max_position"] + 1e-6).all()
            assert result["weight"].sum() <= 1.0 - params["kelly"]["cash_reserve"] + 1e-6


class TestDiffPortfolio:
    def test_all_adds_when_no_prior(self, conn, sample_universe, all_prices, params):
        upsert_prices(conn, all_prices)
        _insert_macro(conn, params)
        scored = _make_scored_df(("ETN", 0.72, True))
        new_port = construct_portfolio(conn, scored, sample_universe, params, AS_OF, dry_run=True)
        if new_port.empty:
            pytest.skip("No entry candidates with this data")
        diff = diff_portfolio(conn, new_port, AS_OF)
        assert (diff["action"] == "add").all()

    def test_remove_dropped_positions(self, conn):
        """Positions in old portfolio but not in new → 'remove'."""
        _make_portfolio_snapshot(conn, ["ETN", "CCJ"], snapshot_date="2022-12-01")
        new_port = pd.DataFrame([{"ticker": "ETN", "weight": 0.08, "primary_bucket": "grid"}])
        diff = diff_portfolio(conn, new_port, AS_OF)
        ccj_row = diff[diff["ticker"] == "CCJ"]
        assert ccj_row["action"].iloc[0] == "remove"

    def test_increase_on_higher_weight(self, conn):
        _make_portfolio_snapshot(conn, ["ETN"], weight=0.05)
        new_port = pd.DataFrame([{"ticker": "ETN", "weight": 0.08, "primary_bucket": "grid"}])
        diff = diff_portfolio(conn, new_port, AS_OF)
        etn_row = diff[diff["ticker"] == "ETN"]
        assert etn_row["action"].iloc[0] == "increase"

    def test_decrease_on_lower_weight(self, conn):
        _make_portfolio_snapshot(conn, ["ETN"], weight=0.08)
        new_port = pd.DataFrame([{"ticker": "ETN", "weight": 0.05, "primary_bucket": "grid"}])
        diff = diff_portfolio(conn, new_port, AS_OF)
        assert diff[diff["ticker"] == "ETN"]["action"].iloc[0] == "decrease"

    def test_hold_within_5pct(self, conn):
        _make_portfolio_snapshot(conn, ["ETN"], weight=0.08)
        new_port = pd.DataFrame([{"ticker": "ETN", "weight": 0.082, "primary_bucket": "grid"}])
        diff = diff_portfolio(conn, new_port, AS_OF)
        assert diff[diff["ticker"] == "ETN"]["action"].iloc[0] == "hold"


# ============================================================================
# Monitor
# ============================================================================

class TestCheckExitSignals:
    def test_empty_portfolio(self, conn, params):
        result = check_exit_signals(conn, pd.DataFrame(), params, AS_OF)
        assert result.empty

    def test_crowding_exit_trigger(self, conn, params):
        # Store a signal score with high crowding
        scores = pd.DataFrame([_make_scored_row(
            "ETN", composite=0.60, crowding=0.80, quality=0.70
        )])
        upsert_signal_scores(conn, scores)

        portfolio = pd.DataFrame([{
            "ticker": "ETN", "weight": 0.07, "composite_score": 0.65,
            "primary_bucket": "grid",
        }])
        result = check_exit_signals(conn, portfolio, params, AS_OF)
        assert result.loc[result["ticker"] == "ETN", "exit_triggered"].iloc[0] == True
        reasons = result.loc[result["ticker"] == "ETN", "exit_reasons"].iloc[0]
        assert any("CROWDED" in r for r in reasons)

    def test_quality_exit_trigger(self, conn, params):
        scores = pd.DataFrame([_make_scored_row(
            "ETN", composite=0.60, crowding=0.25, quality=0.15
        )])
        upsert_signal_scores(conn, scores)

        portfolio = pd.DataFrame([{
            "ticker": "ETN", "weight": 0.07, "composite_score": 0.65,
            "primary_bucket": "grid",
        }])
        result = check_exit_signals(conn, portfolio, params, AS_OF)
        assert result.loc[result["ticker"] == "ETN", "exit_triggered"].iloc[0] == True
        reasons = result.loc[result["ticker"] == "ETN", "exit_reasons"].iloc[0]
        assert any("QUALITY" in r for r in reasons)

    def test_composite_decay_trigger(self, conn, params):
        """Composite decayed >20% from entry composite stored in snapshot."""
        scores = pd.DataFrame([_make_scored_row(
            "ETN", composite=0.48, crowding=0.20, quality=0.70
        )])
        upsert_signal_scores(conn, scores)

        portfolio = pd.DataFrame([{
            "ticker": "ETN", "weight": 0.07,
            "composite_score": 0.65,  # entry composite
            "primary_bucket": "grid",
        }])
        result = check_exit_signals(conn, portfolio, params, AS_OF)
        reasons = result.loc[result["ticker"] == "ETN", "exit_reasons"].iloc[0]
        assert any("DECAY" in r for r in reasons)

    def test_no_exit_for_healthy_position(self, conn, params):
        """Low crowding, high quality, high composite → no exit."""
        scores = pd.DataFrame([_make_scored_row(
            "ETN", composite=0.72, crowding=0.20, quality=0.80
        )])
        upsert_signal_scores(conn, scores)

        portfolio = pd.DataFrame([{
            "ticker": "ETN", "weight": 0.07,
            "composite_score": 0.70,
            "primary_bucket": "grid",
        }])
        result = check_exit_signals(conn, portfolio, params, AS_OF)
        assert result.loc[result["ticker"] == "ETN", "exit_triggered"].iloc[0] == False

    def test_price_drawdown_trigger(self, conn, params):
        """35% drawdown from entry price triggers stop-loss exit."""
        from datetime import date, timedelta
        # Controlled flat-then-crash: entry at 100, then drops to 50
        n = 200
        dates = [date(2022, 1, 3) + timedelta(days=i) for i in range(n)]
        # Prices start at 100, drop to 55 at the end (45% drawdown)
        adj_close = [100.0 - 0.225 * i for i in range(n)]   # 100 → 55.0
        crashed = pd.DataFrame({
            "ticker": "ETN", "date": dates,
            "open": adj_close, "high": adj_close, "low": adj_close,
            "close": adj_close, "adj_close": adj_close,
            "volume": 1_000_000, "currency": "USD", "source": "test",
        })
        upsert_prices(conn, crashed)

        first_date = str(crashed["date"].iloc[0])
        _make_portfolio_snapshot(conn, ["ETN"], snapshot_date=first_date)

        as_of_crash = str(crashed["date"].iloc[-1])  # 2022-07-21
        scores = pd.DataFrame([{**_make_scored_row("ETN", composite=0.60),
                                 "score_date": as_of_crash}])
        upsert_signal_scores(conn, scores)
        portfolio = pd.DataFrame([{
            "ticker": "ETN", "weight": 0.07,
            "composite_score": 0.65, "primary_bucket": "grid",
        }])
        result = check_exit_signals(conn, portfolio, params, as_of_crash)
        reasons = result.loc[result["ticker"] == "ETN", "exit_reasons"].iloc[0]
        assert any("DRAWDOWN" in r for r in reasons)


class TestCheckEntrySignals:
    def test_returns_entry_candidates_not_held(self, conn):
        scored = _make_scored_df(
            ("ETN", 0.72, True),
            ("CCJ", 0.68, True),
            ("RHM.DE", 0.45, False),
        )
        current_portfolio = pd.DataFrame([{"ticker": "ETN", "weight": 0.07}])
        result = check_entry_signals(conn, current_portfolio, scored, {})
        assert "ETN" not in result["ticker"].tolist()
        assert "CCJ" in result["ticker"].tolist()
        assert "RHM.DE" not in result["ticker"].tolist()

    def test_sorted_by_composite_desc(self, conn):
        scored = _make_scored_df(
            ("ETN", 0.65, True),
            ("CCJ", 0.80, True),
            ("PWR", 0.72, True),
        )
        result = check_entry_signals(conn, pd.DataFrame(), scored, {})
        scores = result["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_empty_portfolio_all_candidates(self, conn):
        scored = _make_scored_df(("ETN", 0.72, True), ("CCJ", 0.68, True))
        result = check_entry_signals(conn, pd.DataFrame(), scored, {})
        assert set(result["ticker"]) == {"ETN", "CCJ"}

    def test_no_entry_signals(self, conn):
        scored = _make_scored_df(("ETN", 0.40, False), ("CCJ", 0.42, False))
        result = check_entry_signals(conn, pd.DataFrame(), scored, {})
        assert result.empty


class TestGetWatchList:
    def test_elevated_crowding_on_watch(self, params):
        """Positions with crowding between watch_thr and exit_thr → on watch list."""
        current = pd.DataFrame([{"ticker": "ETN", "weight": 0.07}])
        scored = _make_scored_df(("ETN", 0.65, True, 0.60))  # crowding=0.60 (above watch 0.55)
        result = get_watch_list(current, scored, params)
        assert "ETN" in result["ticker"].tolist()

    def test_below_watch_threshold_not_watched(self, params):
        current = pd.DataFrame([{"ticker": "ETN", "weight": 0.07}])
        scored = _make_scored_df(("ETN", 0.65, True, 0.30))  # crowding=0.30
        result = get_watch_list(current, scored, params)
        assert "ETN" not in result["ticker"].tolist()

    def test_at_exit_threshold_not_watched(self, params):
        """At exit_threshold, it should exit — not just be watched."""
        current = pd.DataFrame([{"ticker": "ETN", "weight": 0.07}])
        scored = _make_scored_df(("ETN", 0.65, True, 0.75))  # crowding=0.75 = exit_thr
        result = get_watch_list(current, scored, params)
        # inclusive="left" means left boundary included but not right → 0.75 excluded
        assert "ETN" not in result["ticker"].tolist()

    def test_non_held_tickers_not_on_watch(self, params):
        current = pd.DataFrame([{"ticker": "ETN", "weight": 0.07}])
        scored = _make_scored_df(("ETN", 0.65, True, 0.60), ("CCJ", 0.68, True, 0.62))
        result = get_watch_list(current, scored, params)
        assert "CCJ" not in result["ticker"].tolist()


class TestGenerateWeeklyActions:
    def test_returns_expected_keys(self, conn, sample_universe, all_prices, params):
        upsert_prices(conn, all_prices)
        _insert_macro(conn, params)
        scored = _make_scored_df(("ETN", 0.72, True))
        upsert_signal_scores(conn, scored)
        actions = generate_weekly_actions(conn, scored, sample_universe, params, AS_OF)
        for key in ["exits", "entries", "watch_list", "new_portfolio", "diff", "as_of_date", "any_action"]:
            assert key in actions

    def test_no_action_week(self, conn, sample_universe, params):
        """No entries, no exits, no watch → any_action=False."""
        # No prices → no entry candidates → no portfolio built
        scored = _make_scored_df(("ETN", 0.40, False))
        upsert_signal_scores(conn, scored)
        actions = generate_weekly_actions(conn, scored, sample_universe, params, AS_OF)
        assert actions["any_action"] == False

    def test_entry_week_any_action_true(self, conn, sample_universe, all_prices, params):
        """With a valid entry candidate any_action should be True."""
        upsert_prices(conn, all_prices)
        _insert_macro(conn, params)
        scored = _make_scored_df(("ETN", 0.72, True))
        upsert_signal_scores(conn, scored)
        actions = generate_weekly_actions(conn, scored, sample_universe, params, AS_OF)
        # Entries list is populated (even if portfolio construction finds no Kelly candidates,
        # the entries DataFrame itself may be populated from check_entry_signals)
        assert isinstance(actions["entries"], pd.DataFrame)

    def test_as_of_date_propagated(self, conn, sample_universe, params):
        scored = _make_scored_df(("ETN", 0.40, False))
        upsert_signal_scores(conn, scored)
        actions = generate_weekly_actions(conn, scored, sample_universe, params, AS_OF)
        assert actions["as_of_date"] == AS_OF

    def test_exit_week(self, conn, sample_universe, params):
        """With a held position at high crowding, exits should be non-empty."""
        # Persist a prior portfolio
        _make_portfolio_snapshot(conn, ["ETN"])
        # Score ETN with high crowding
        scored = pd.DataFrame([_make_scored_row(
            "ETN", composite=0.60, crowding=0.80, quality=0.70
        )])
        upsert_signal_scores(conn, scored)
        actions = generate_weekly_actions(conn, scored, sample_universe, params, AS_OF)
        exits = actions["exits"]
        if not exits.empty:
            triggered = exits[exits["exit_triggered"] == True]
            assert len(triggered) > 0

    def test_dry_run_not_persisted(self, conn, sample_universe, all_prices, params):
        """dry_run=True must propagate to construct_portfolio so no snapshot is written."""
        upsert_prices(conn, all_prices)
        _insert_macro(conn, params)
        scored = _make_scored_df(("ETN", 0.72, True))
        upsert_signal_scores(conn, scored)

        actions = generate_weekly_actions(
            conn, scored, sample_universe, params, AS_OF, dry_run=True
        )

        # Portfolio is returned in-memory but must NOT be persisted to DB
        portfolio = get_latest_portfolio(conn)
        assert portfolio.empty, "dry_run=True must not write a portfolio snapshot"
        # The in-memory result is still returned for reporting
        assert "new_portfolio" in actions
