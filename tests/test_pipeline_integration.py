"""
End-to-end integration tests for run_weekly_scoring.

Inserts all required data (prices, fundamentals, macro) into an in-memory
DuckDB and verifies the full scoring pipeline produces valid, internally
consistent results.
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
    get_latest_signal_scores,
)
from src.signals.composite import run_weekly_scoring

AS_OF = "2023-01-13"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_macro(conn, params):
    from datetime import date, timedelta
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(1500)]
    rf_df = pd.DataFrame({"date": [d.isoformat() for d in dates],
                           "value": [4.3] * len(dates), "source": "test"})
    ppi_df = pd.DataFrame({"date": [d.isoformat() for d in dates],
                            "value": [3.0] * len(dates), "source": "test"})
    upsert_macro(conn, params["macro"]["us_risk_free_series"], rf_df)
    upsert_macro(conn, params["macro"]["eu_risk_free_series"], rf_df.assign(value=2.8))
    upsert_macro(conn, params["macro"]["us_ppi_series"], ppi_df)
    upsert_macro(conn, params["macro"]["eu_ppi_series"], ppi_df)


def _full_setup(conn, sample_universe, all_prices,
                fundamentals_etn, fundamentals_rhm, params):
    """Populate DB with everything run_weekly_scoring needs."""
    upsert_prices(conn, all_prices)
    upsert_fundamentals_annual(conn, fundamentals_etn)
    upsert_fundamentals_annual(conn, fundamentals_rhm)
    _insert_macro(conn, params)


# ---------------------------------------------------------------------------
# Output shape and schema
# ---------------------------------------------------------------------------

def test_run_weekly_scoring_returns_dataframe(
    conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params
):
    _full_setup(conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params)
    result = run_weekly_scoring(conn, sample_universe, params, AS_OF)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_universe)


def test_run_weekly_scoring_expected_columns(
    conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params
):
    _full_setup(conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params)
    result = run_weekly_scoring(conn, sample_universe, params, AS_OF)
    required_cols = [
        "ticker", "score_date",
        "physical_norm", "physical_confidence",
        "quality_score", "quality_confidence",
        "crowding_score", "crowding_confidence",
        "composite_score", "composite_confidence",
        "entry_signal", "exit_signal",
    ]
    for col in required_cols:
        assert col in result.columns, f"Missing column '{col}'"


def test_run_weekly_scoring_score_date(
    conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params
):
    _full_setup(conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params)
    result = run_weekly_scoring(conn, sample_universe, params, AS_OF)
    assert (result["score_date"] == AS_OF).all()


def test_run_weekly_scoring_all_tickers_present(
    conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params
):
    _full_setup(conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params)
    result = run_weekly_scoring(conn, sample_universe, params, AS_OF)
    assert set(result["ticker"]) == set(sample_universe["ticker"])


# ---------------------------------------------------------------------------
# Score bounds
# ---------------------------------------------------------------------------

def test_composite_scores_in_bounds(
    conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params
):
    _full_setup(conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params)
    result = run_weekly_scoring(conn, sample_universe, params, AS_OF)
    scored = result.dropna(subset=["composite_score"])
    assert (scored["composite_score"].between(0, 1)).all()
    assert (scored["composite_confidence"].between(0, 1)).all()


def test_quality_scores_in_bounds(
    conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params
):
    _full_setup(conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params)
    result = run_weekly_scoring(conn, sample_universe, params, AS_OF)
    assert (result["quality_score"].between(0, 1)).all()


def test_crowding_scores_in_bounds(
    conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params
):
    _full_setup(conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params)
    result = run_weekly_scoring(conn, sample_universe, params, AS_OF)
    assert (result["crowding_score"].between(0, 1)).all()


def test_physical_scores_in_bounds(
    conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params
):
    _full_setup(conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params)
    result = run_weekly_scoring(conn, sample_universe, params, AS_OF)
    assert (result["physical_norm"].between(0, 1)).all()


# ---------------------------------------------------------------------------
# Entry / exit signal logic
# ---------------------------------------------------------------------------

def test_entry_signal_requires_high_composite(
    conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params
):
    """No entry signal should have composite < entry_threshold."""
    _full_setup(conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params)
    result = run_weekly_scoring(conn, sample_universe, params, AS_OF)
    threshold = params["signals"]["entry_threshold"]
    entries = result[result["entry_signal"] == True]
    if not entries.empty:
        assert (entries["composite_score"] >= threshold).all()


def test_entry_signal_requires_low_crowding(
    conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params
):
    """No entry signal should have crowding above the entry ceiling."""
    _full_setup(conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params)
    result = run_weekly_scoring(conn, sample_universe, params, AS_OF)
    crowd_max = params["signals"]["crowding_entry_max"]
    entries = result[result["entry_signal"] == True]
    if not entries.empty:
        assert (entries["crowding_score"] <= crowd_max).all()


def test_exit_signal_logic(
    conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params
):
    """Rows with exit_signal=True must satisfy at least one exit condition."""
    _full_setup(conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params)
    result = run_weekly_scoring(conn, sample_universe, params, AS_OF)
    exits = result[result["exit_signal"] == True]
    if not exits.empty:
        crowd_exit = params["signals"]["crowding_exit_threshold"]
        qual_exit  = params["signals"]["quality_exit_threshold"]
        violated = exits[
            (exits["crowding_score"] < crowd_exit) &
            (exits["quality_score"] > qual_exit)
        ]
        assert len(violated) == 0, f"Exit signal without exit condition: {violated}"


def test_signals_are_mutually_exclusive(
    conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params
):
    """A row cannot have both entry_signal=True and exit_signal=True."""
    _full_setup(conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params)
    result = run_weekly_scoring(conn, sample_universe, params, AS_OF)
    both = result[result["entry_signal"] & result["exit_signal"]]
    assert both.empty, f"Rows with both signals: {both['ticker'].tolist()}"


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------

def test_scores_persisted_to_db(
    conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params
):
    """run_weekly_scoring must upsert results to signal_scores table."""
    _full_setup(conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params)
    run_weekly_scoring(conn, sample_universe, params, AS_OF)
    stored = get_latest_signal_scores(conn)
    assert len(stored) == len(sample_universe)
    assert set(stored["ticker"]) == set(sample_universe["ticker"])


def test_second_run_updates_scores(
    conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params
):
    """Running scoring twice for the same date must not duplicate rows."""
    _full_setup(conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params)
    run_weekly_scoring(conn, sample_universe, params, AS_OF)
    run_weekly_scoring(conn, sample_universe, params, AS_OF)
    count = conn.execute(
        f"SELECT count(*) FROM signal_scores WHERE score_date = '{AS_OF}'"
    ).fetchone()[0]
    assert count == len(sample_universe), "Duplicate rows written on second run"


def test_mu_estimate_only_for_entry_candidates(
    conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params
):
    """mu_estimate is computed for any stock with valid composite (rf + θ·composite).
    Stocks with null composite_score (insufficient data) should have null mu_estimate."""
    _full_setup(conn, sample_universe, all_prices, fundamentals_etn, fundamentals_rhm, params)
    result = run_weekly_scoring(conn, sample_universe, params, AS_OF)
    # Rows where composite_score is null must also have null mu_estimate
    null_composite = result[result["composite_score"].isna()]
    if not null_composite.empty:
        assert null_composite["mu_estimate"].isna().all(), \
            "mu_estimate must be null when composite_score is null"
    # Rows with valid composite must have a valid mu_estimate (rf + θ·composite > 0)
    valid_composite = result[result["composite_score"].notna()]
    if not valid_composite.empty:
        assert valid_composite["mu_estimate"].notna().all(), \
            "mu_estimate must be non-null for any stock with valid composite"


# ---------------------------------------------------------------------------
# Empty universe edge case
# ---------------------------------------------------------------------------

def test_run_weekly_scoring_empty_universe(conn, params):
    _insert_macro(conn, params)
    empty_universe = pd.DataFrame(columns=[
        "ticker", "name", "exchange", "region", "currency",
        "accounting_std", "buckets", "primary_bucket", "isin", "trends_keyword",
    ])
    result = run_weekly_scoring(conn, empty_universe, params, AS_OF)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ---------------------------------------------------------------------------
# Data-sparse scenario (only 1 ticker has data)
# ---------------------------------------------------------------------------

def test_partial_data_other_tickers_still_scored(
    conn, sample_universe, prices_etn, prices_spy, fundamentals_etn, params
):
    """
    Only ETN has prices+fundamentals; CCJ and RHM.DE have no data.
    All tickers should still appear in output (with default/neutral scores).
    """
    upsert_prices(conn, prices_etn)
    upsert_prices(conn, prices_spy)
    upsert_fundamentals_annual(conn, fundamentals_etn)
    _insert_macro(conn, params)

    result = run_weekly_scoring(conn, sample_universe, params, AS_OF)
    assert len(result) == len(sample_universe)

    # ETN should have higher data confidence than the others
    etn_conf = result[result["ticker"] == "ETN"]["composite_confidence"].iloc[0]
    ccj_conf = result[result["ticker"] == "CCJ"]["composite_confidence"].iloc[0]
    assert etn_conf >= ccj_conf
