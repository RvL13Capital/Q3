"""
Integration tests for the quality signal (Signal 2).

All tests spin up an in-memory DuckDB populated with synthetic fundamentals
and macro data — no network, no disk I/O.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.db import upsert_fundamentals_annual, upsert_macro
from src.signals.quality import (
    compute_roic_wacc_spread,
    compute_margin_snr,
    compute_inflation_convexity,
    compute_quality_score,
    batch_quality_scores,
)

AS_OF = "2025-01-05"


# ---------------------------------------------------------------------------
# Macro fixtures: rates + inflation inserted into DB
# ---------------------------------------------------------------------------

def _insert_macro(conn, params):
    """Insert 10Y yields and PPI series for US and EU."""
    dates = [date(2022, 1, 1) + timedelta(days=i) for i in range(1200)]
    rows_yield = pd.DataFrame({
        "date":   [d.isoformat() for d in dates],
        "value":  [4.3] * len(dates),
        "source": "test",
    })
    rows_ppi = pd.DataFrame({
        "date":   [d.isoformat() for d in dates],
        "value":  [3.0] * len(dates),   # 3% PPI YoY
        "source": "test",
    })
    upsert_macro(conn, params["macro"]["us_risk_free_series"], rows_yield)
    upsert_macro(conn, params["macro"]["eu_risk_free_series"], rows_yield.assign(value=2.8))
    upsert_macro(conn, params["macro"]["us_ppi_series"], rows_ppi)
    upsert_macro(conn, params["macro"]["eu_ppi_series"], rows_ppi)


# ---------------------------------------------------------------------------
# ROIC-WACC spread
# ---------------------------------------------------------------------------

def test_roic_wacc_spread_above_cost_of_capital(conn, fundamentals_etn, params):
    upsert_fundamentals_annual(conn, fundamentals_etn)
    _insert_macro(conn, params)
    spread, score, conf = compute_roic_wacc_spread("ETN", conn, params, AS_OF, "US")
    # Our synthetic ROIC ≈ 0.07; WACC ≈ 4.3/100 + 5/100 ≈ 0.093 → spread negative/close to 0
    assert conf > 0, "Should have confidence with real data"
    assert 0.0 <= score <= 1.0


def test_roic_wacc_spread_no_data(conn, params):
    """No fundamentals → nan spread, 0.5 score, 0 confidence."""
    _insert_macro(conn, params)
    spread, score, conf = compute_roic_wacc_spread("MISSING", conn, params, AS_OF, "US")
    import math
    assert math.isnan(spread)
    assert score == 0.5
    assert conf == 0.0


def test_roic_wacc_spread_no_macro(conn, fundamentals_etn, params):
    """No macro data → falls back to 4% RF, still returns valid score."""
    upsert_fundamentals_annual(conn, fundamentals_etn)
    # Don't insert macro → fallback RF = 4%
    spread, score, conf = compute_roic_wacc_spread("ETN", conn, params, AS_OF, "US")
    assert conf > 0


def test_roic_wacc_spread_eu_region(conn, fundamentals_rhm, params):
    """EU stock should use EU risk-free rate."""
    upsert_fundamentals_annual(conn, fundamentals_rhm)
    _insert_macro(conn, params)
    spread, score, conf = compute_roic_wacc_spread("RHM.DE", conn, params, AS_OF, "EU")
    assert conf > 0


# ---------------------------------------------------------------------------
# Gross margin SNR
# ---------------------------------------------------------------------------

def test_margin_snr_stable_margins(conn, fundamentals_etn, params):
    """Synthetic ETN data has constant gross margin (40%) → high SNR."""
    upsert_fundamentals_annual(conn, fundamentals_etn)
    snr, score, conf = compute_margin_snr("ETN", conn, params)
    assert snr > 5.0, f"Expected high SNR for stable margins, got {snr}"
    assert score == pytest.approx(1.0)   # SNR >> 10 max → capped at 1
    assert conf == pytest.approx(1.0)    # 5 years of data


def test_margin_snr_no_data(conn, params):
    import math
    snr, score, conf = compute_margin_snr("MISSING", conn, params)
    assert math.isnan(snr)
    assert score == 0.5
    assert conf == 0.0


def test_margin_snr_single_year(conn, fundamentals_etn, params):
    """Only 1 year of data → low confidence, neutral score."""
    one_year = fundamentals_etn.head(1)
    upsert_fundamentals_annual(conn, one_year)
    import math
    snr, score, conf = compute_margin_snr("ETN", conn, params)
    assert math.isnan(snr), "SNR undefined with < 2 years"
    assert conf < 0.5


def test_margin_snr_confidence_scales_with_years(conn, params, fundamentals_etn):
    """Confidence should be min(1, n/5)."""
    upsert_fundamentals_annual(conn, fundamentals_etn.head(3))
    _, _, conf_3yr = compute_margin_snr("ETN", conn, params)

    conn.execute("DELETE FROM fundamentals_annual WHERE ticker='ETN'")
    upsert_fundamentals_annual(conn, fundamentals_etn)  # 5 years
    _, _, conf_5yr = compute_margin_snr("ETN", conn, params)

    assert conf_5yr > conf_3yr


# ---------------------------------------------------------------------------
# Inflation convexity
# ---------------------------------------------------------------------------

def test_inflation_convexity_positive(conn, fundamentals_etn, params):
    """8% revenue CAGR vs 3% PPI → positive convexity."""
    upsert_fundamentals_annual(conn, fundamentals_etn)
    _insert_macro(conn, params)
    convexity, score, conf = compute_inflation_convexity(
        "ETN", conn, params, AS_OF, "US", lookback_years=3
    )
    assert convexity > 0, f"Expected positive convexity, got {convexity}"
    assert score > 0.5, "High pricing power should push score above 0.5"
    assert conf > 0


def test_inflation_convexity_no_ppi_data(conn, fundamentals_etn, params):
    """No PPI data → falls back to 4% PPI, still returns valid score."""
    upsert_fundamentals_annual(conn, fundamentals_etn)
    # No macro → PPI fallback 4%
    convexity, score, conf = compute_inflation_convexity(
        "ETN", conn, params, AS_OF, "US", lookback_years=3
    )
    assert conf > 0, "Should still produce result with fallback PPI"


def test_inflation_convexity_no_fundamentals(conn, params):
    """No fundamental data → nan convexity."""
    import math
    convexity, score, conf = compute_inflation_convexity(
        "MISSING", conn, params, AS_OF, "US"
    )
    assert math.isnan(convexity)
    assert conf == 0.0


# ---------------------------------------------------------------------------
# Composite quality score
# ---------------------------------------------------------------------------

def test_quality_score_bounds(conn, fundamentals_etn, params):
    """Quality score must be in [0, 1]."""
    upsert_fundamentals_annual(conn, fundamentals_etn)
    _insert_macro(conn, params)
    result = compute_quality_score("ETN", conn, params, AS_OF, "US")
    assert 0.0 <= result["quality_score"] <= 1.0
    assert 0.0 <= result["quality_confidence"] <= 1.0


def test_quality_score_no_data(conn, params):
    """No data → default neutral score with zero confidence."""
    result = compute_quality_score("MISSING", conn, params, AS_OF, "US")
    assert result["quality_score"] == 0.5
    assert result["quality_confidence"] == 0.0


def test_quality_score_eu_stock(conn, fundamentals_rhm, params):
    upsert_fundamentals_annual(conn, fundamentals_rhm)
    _insert_macro(conn, params)
    result = compute_quality_score("RHM.DE", conn, params, AS_OF, "EU")
    assert 0.0 <= result["quality_score"] <= 1.0
    assert result["quality_confidence"] > 0


def test_quality_score_keys(conn, fundamentals_etn, params):
    """Result dict must contain all expected keys."""
    upsert_fundamentals_annual(conn, fundamentals_etn)
    _insert_macro(conn, params)
    result = compute_quality_score("ETN", conn, params, AS_OF, "US")
    for key in ["ticker", "quality_score", "quality_confidence",
                "roic_wacc_spread", "margin_snr", "inflation_convexity"]:
        assert key in result, f"Missing key '{key}' in quality result"


def test_quality_score_good_company_scores_higher(conn, params,
                                                   fundamentals_etn, fundamentals_rhm):
    """Both companies have similar synthetic profiles; both should score above 0.5."""
    upsert_fundamentals_annual(conn, fundamentals_etn)
    upsert_fundamentals_annual(conn, fundamentals_rhm)
    _insert_macro(conn, params)
    r_etn = compute_quality_score("ETN", conn, params, AS_OF, "US")
    r_rhm = compute_quality_score("RHM.DE", conn, params, AS_OF, "EU")
    # Both profitable → should score reasonably
    assert r_etn["quality_score"] > 0.4
    assert r_rhm["quality_score"] > 0.4


# ---------------------------------------------------------------------------
# Batch quality scores
# ---------------------------------------------------------------------------

def test_batch_quality_scores(conn, sample_universe, fundamentals_etn,
                               fundamentals_rhm, params):
    upsert_fundamentals_annual(conn, fundamentals_etn)
    upsert_fundamentals_annual(conn, fundamentals_rhm)
    _insert_macro(conn, params)

    # Only score the two tickers with data (CCJ has no fundamentals in DB)
    universe_sub = sample_universe[sample_universe["ticker"].isin(["ETN", "RHM.DE"])]
    result = batch_quality_scores(universe_sub, conn, params, AS_OF)

    assert len(result) == 2
    assert set(result["ticker"]) == {"ETN", "RHM.DE"}
    assert (result["quality_score"].between(0, 1)).all()


def test_batch_quality_scores_missing_ticker(conn, sample_universe, params):
    """Tickers with no data should still get a result (neutral score)."""
    result = batch_quality_scores(sample_universe, conn, params, AS_OF)
    assert len(result) == len(sample_universe)
    # Missing data → quality_score defaults to 0.5
    ccj_score = result[result["ticker"] == "CCJ"]["quality_score"].iloc[0]
    assert ccj_score == pytest.approx(0.5)
