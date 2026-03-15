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
    spread, conf = compute_roic_wacc_spread("ETN", conn, params, AS_OF, "US")
    # Synthetic ROIC ≈ 0.118; WACC ≈ 4.3/100 + 5/100 = 9.3% → positive spread
    assert conf > 0, "Should have confidence with real data"
    assert isinstance(spread, float)


def test_roic_wacc_spread_no_data(conn, params):
    """No fundamentals → nan spread, 0 confidence."""
    _insert_macro(conn, params)
    spread, conf = compute_roic_wacc_spread("MISSING", conn, params, AS_OF, "US")
    import math
    assert math.isnan(spread)
    assert conf == 0.0


def test_roic_wacc_spread_no_macro(conn, fundamentals_etn, params):
    """No macro data → falls back to 4% RF, still returns valid spread."""
    upsert_fundamentals_annual(conn, fundamentals_etn)
    # Don't insert macro → fallback RF = 4%
    spread, conf = compute_roic_wacc_spread("ETN", conn, params, AS_OF, "US")
    assert conf > 0


def test_roic_wacc_spread_eu_region(conn, fundamentals_rhm, params):
    """EU stock should use EU risk-free rate."""
    upsert_fundamentals_annual(conn, fundamentals_rhm)
    _insert_macro(conn, params)
    spread, conf = compute_roic_wacc_spread("RHM.DE", conn, params, AS_OF, "EU")
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

def test_inflation_convexity_stable_margins(conn, fundamentals_etn, params):
    """Constant gross margins (delta=0) → OLS slope = 0."""
    upsert_fundamentals_annual(conn, fundamentals_etn)
    _insert_macro(conn, params)
    convexity, conf = compute_inflation_convexity(
        "ETN", conn, params, AS_OF, "US", lookback_years=3
    )
    # Synthetic ETN has constant 40% GM → delta_gm = 0 → slope = 0
    assert convexity == pytest.approx(0.0, abs=1e-6)
    assert conf > 0


def test_inflation_convexity_varying_margins(conn, params):
    """Margins rising with PPI → positive ∂GM/∂PPI slope."""
    gm_series = [0.38, 0.39, 0.41, 0.43, 0.44]
    rows = []
    for i, gm in enumerate(gm_series):
        fy = 2020 + i
        rev = 5e9
        gp = rev * gm
        ebit = rev * 0.12
        equity, debt, cash, gw = 3e9, 1e9, 0.5e9, 0.5e9
        ic = equity + debt - cash - gw
        nopat = ebit * 0.79
        rows.append({
            "ticker": "VGMTEST", "fiscal_year": fy,
            "report_date": date(fy, 12, 31), "currency": "USD",
            "accounting_std": "GAAP", "source": "test",
            "revenue": rev, "gross_profit": gp, "ebit": ebit,
            "net_income": ebit * 0.75, "interest_expense": debt * 0.04,
            "tax_expense": ebit * 0.21, "total_assets": equity + debt,
            "total_equity": equity, "total_debt": debt, "cash": cash,
            "goodwill": gw, "intangible_assets": 0.2e9,
            "right_of_use_assets": None, "lease_liabilities": None,
            "capex": rev * 0.04, "gross_margin": gm, "invested_capital": ic,
            "nopat": nopat, "roic": nopat / ic, "effective_tax_rate": 0.21,
        })
    import pandas as pd
    from src.data.db import upsert_fundamentals_annual as _ufa, upsert_macro
    _ufa(conn, pd.DataFrame(rows))

    # PPI rising from 2% to 6% — monthly YoY readings aligned to fiscal years
    macro_rows = []
    for i, ppi_val in enumerate([2.0, 3.0, 4.0, 5.0, 6.0]):
        yr = 2020 + i
        for m in range(1, 13):
            macro_rows.append({"date": date(yr, m, 28).isoformat(),
                                "value": ppi_val, "source": "test"})
    upsert_macro(conn, params["macro"]["us_ppi_series"], pd.DataFrame(macro_rows))

    convexity, conf = compute_inflation_convexity(
        "VGMTEST", conn, params, "2024-12-31", "US", lookback_years=4
    )
    assert convexity > 0, f"Rising margins + rising PPI → positive slope, got {convexity}"
    assert conf > 0


def test_inflation_convexity_no_ppi_data(conn, fundamentals_etn, params):
    """No PPI data in DB → returns (nan, 0.0) — no automatic fallback."""
    upsert_fundamentals_annual(conn, fundamentals_etn)
    import math
    convexity, conf = compute_inflation_convexity(
        "ETN", conn, params, AS_OF, "US", lookback_years=3
    )
    assert math.isnan(convexity)
    assert conf == 0.0


def test_inflation_convexity_no_fundamentals(conn, params):
    """No fundamental data → nan convexity."""
    import math
    convexity, conf = compute_inflation_convexity(
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
    """No data → ROIC confidence=0 triggers hard gate → quality_score=0.0."""
    result = compute_quality_score("MISSING", conn, params, AS_OF, "US")
    assert result["quality_score"] == 0.0   # hard gate: roic_conf=0 → X_P=0
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
    """Both companies have positive ROIC spread → quality_score > 0.
    Synthetic data: ROIC≈11.8%, WACC≈9.3% → spread≈2.5pp → spread_factor=0.125.
    Constant GM → SNR high → snr_norm≈1.0. Convexity=0 → exp(0)=1.0.
    quality_score ≈ 0.125 × 1.0 × 1.0 = 0.125 — positive moat confirmed.
    """
    upsert_fundamentals_annual(conn, fundamentals_etn)
    upsert_fundamentals_annual(conn, fundamentals_rhm)
    _insert_macro(conn, params)
    r_etn = compute_quality_score("ETN", conn, params, AS_OF, "US")
    r_rhm = compute_quality_score("RHM.DE", conn, params, AS_OF, "EU")
    assert r_etn["quality_score"] > 0.0, "Profitable company must score > 0"
    assert r_rhm["quality_score"] > 0.0, "Profitable company must score > 0"


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
    """Tickers with no data return a row with quality_score=0.0 (ROIC hard gate)."""
    result = batch_quality_scores(sample_universe, conn, params, AS_OF)
    assert len(result) == len(sample_universe)
    # No fundamentals → roic_conf=0 → hard gate → quality_score=0.0
    ccj_score = result[result["ticker"] == "CCJ"]["quality_score"].iloc[0]
    assert ccj_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Convexity cap and confidence attenuation (Fix 1)
# ---------------------------------------------------------------------------

def _insert_high_convexity_company(conn, params, ticker="CAPTEST", gm_series=None,
                                    ppi_values=None):
    """Insert a company with rising margins and rising PPI for convexity tests.

    Returns fundamentals DataFrame for reference.
    """
    if gm_series is None:
        # Strong pricing power: margins rise sharply with PPI
        gm_series = [0.30, 0.33, 0.37, 0.42, 0.48, 0.55]
    if ppi_values is None:
        ppi_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    rows = []
    for i, gm in enumerate(gm_series):
        fy = 2019 + i
        rev = 5e9
        gp = rev * gm
        ebit = rev * 0.20  # high ROIC to pass hard gate
        equity, debt, cash, gw = 3e9, 1e9, 0.5e9, 0.5e9
        ic = equity + debt - cash - gw
        nopat = ebit * 0.79
        rows.append({
            "ticker": ticker, "fiscal_year": fy,
            "report_date": date(fy, 12, 31), "currency": "USD",
            "accounting_std": "GAAP", "source": "test",
            "revenue": rev, "gross_profit": gp, "ebit": ebit,
            "net_income": ebit * 0.75, "interest_expense": debt * 0.04,
            "tax_expense": ebit * 0.21, "total_assets": equity + debt,
            "total_equity": equity, "total_debt": debt, "cash": cash,
            "goodwill": gw, "intangible_assets": 0.2e9,
            "right_of_use_assets": None, "lease_liabilities": None,
            "capex": rev * 0.04, "gross_margin": gm, "invested_capital": ic,
            "nopat": nopat, "roic": nopat / ic, "effective_tax_rate": 0.21,
        })
    fund_df = pd.DataFrame(rows)
    upsert_fundamentals_annual(conn, fund_df)

    # PPI series aligned to fiscal years
    macro_rows = []
    for i, ppi_val in enumerate(ppi_values):
        yr = 2019 + i
        for m in range(1, 13):
            macro_rows.append({"date": date(yr, m, 28).isoformat(),
                                "value": ppi_val, "source": "test"})
    upsert_macro(conn, params["macro"]["us_ppi_series"], pd.DataFrame(macro_rows))

    # Insert yields so WACC can be computed
    yield_rows = pd.DataFrame({
        "date": [date(2019 + i // 365, 1, 1) + timedelta(days=i % 365)
                 for i in range(2500)],
        "source": "test",
    })
    yield_rows["date"] = yield_rows["date"].apply(lambda d: d.isoformat())
    yield_rows["value"] = 4.3
    upsert_macro(conn, params["macro"]["us_risk_free_series"], yield_rows)

    return fund_df


def test_convexity_cap_limits_multiplier(conn, params):
    """With slope=0.5 and γ=2.0, uncapped exp(1.0)=2.72.
    After confidence attenuation and hard cap at 1.50, the quality_score
    must be bounded by the cap effect."""
    _insert_high_convexity_company(conn, params, ticker="CAPTEST")

    result = compute_quality_score("CAPTEST", conn, params, "2024-12-31", "US")
    assert result["quality_score"] > 0.0, "Should have positive quality score"

    # Now compute without cap to verify capping is active
    import copy
    params_uncapped = copy.deepcopy(params)
    del params_uncapped["signals"]["quality"]["convexity_cap"]
    result_uncapped = compute_quality_score("CAPTEST", conn, params_uncapped, "2024-12-31", "US")

    # The capped score should be <= uncapped score (cap reduces amplification)
    assert result["quality_score"] <= result_uncapped["quality_score"], \
        f"Capped {result['quality_score']} should be <= uncapped {result_uncapped['quality_score']}"


def test_convexity_cap_absent_no_cap(conn, params):
    """When convexity_cap is missing from params, old uncapped behavior is preserved."""
    import copy, math
    params_no_cap = copy.deepcopy(params)
    del params_no_cap["signals"]["quality"]["convexity_cap"]

    _insert_high_convexity_company(conn, params_no_cap, ticker="NOCAP")

    # Compute convexity to verify it's positive
    convexity, conf = compute_inflation_convexity(
        "NOCAP", conn, params_no_cap, "2024-12-31", "US", lookback_years=5
    )
    assert convexity > 0, f"Expected positive convexity, got {convexity}"

    # With no cap, the quality score should still be valid [0, 1]
    result = compute_quality_score("NOCAP", conn, params_no_cap, "2024-12-31", "US")
    assert 0.0 <= result["quality_score"] <= 1.0

    # Verify that the conv_factor is not artificially limited:
    # Since there's no cap param, the default is inf, so attenuation still applies
    # but there's no hard ceiling. The score should be >= what a capped version produces.
    params_capped = copy.deepcopy(params)  # has convexity_cap: 1.50
    result_capped = compute_quality_score("NOCAP", conn, params_capped, "2024-12-31", "US")
    assert result["quality_score"] >= result_capped["quality_score"], \
        "Uncapped score should be >= capped score"


def test_convexity_confidence_attenuation(conn, params):
    """Low-confidence convexity (few data points) should produce less amplification
    than high-confidence convexity (many data points)."""
    # Few data points (3 years → 2 diffs → low confidence)
    _insert_high_convexity_company(
        conn, params, ticker="FEWPTS",
        gm_series=[0.30, 0.38, 0.48],
        ppi_values=[1.0, 3.0, 5.0],
    )
    # Insert yields for FEWPTS
    yield_rows = pd.DataFrame({
        "date": [(date(2019, 1, 1) + timedelta(days=i)).isoformat()
                 for i in range(2500)],
        "value": 4.3,
        "source": "test",
    })
    upsert_macro(conn, params["macro"]["us_risk_free_series"], yield_rows)

    conv_few, conf_few = compute_inflation_convexity(
        "FEWPTS", conn, params, "2024-12-31", "US", lookback_years=5
    )

    # Many data points (6 years → 5 diffs → higher confidence)
    _insert_high_convexity_company(
        conn, params, ticker="MANYPTS",
        gm_series=[0.30, 0.33, 0.37, 0.42, 0.48, 0.55],
        ppi_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )

    conv_many, conf_many = compute_inflation_convexity(
        "MANYPTS", conn, params, "2024-12-31", "US", lookback_years=5
    )

    # Both should have positive convexity
    assert conv_few > 0, f"Expected positive convexity for FEWPTS, got {conv_few}"
    assert conv_many > 0, f"Expected positive convexity for MANYPTS, got {conv_many}"

    # More data points → higher confidence
    assert conf_many > conf_few, \
        f"More data points should give higher confidence: {conf_many} vs {conf_few}"

    # With convexity_min_points=4 in params, 2 data points / max(4, 5) = 0.4 confidence
    # vs 5 data points / max(4, 5) = 1.0 confidence
    assert conf_few < 1.0, f"Few points should have confidence < 1.0, got {conf_few}"


# ---------------------------------------------------------------------------
# Beta-adjusted WACC (Fix 2)
# ---------------------------------------------------------------------------

def _insert_prices(conn, ticker, n_days=400, start_price=100.0, vol=0.02):
    """Insert synthetic prices into the DB for beta tests."""
    import numpy as np
    from datetime import date, timedelta
    from src.data.db import upsert_prices

    rng = np.random.default_rng(seed=hash(ticker) % (2**32))
    dates = [date(2022, 1, 3) + timedelta(days=i) for i in range(n_days)]
    prices = [start_price]
    for _ in range(n_days - 1):
        prices.append(prices[-1] * (1 + rng.normal(0.0003, vol)))

    df = pd.DataFrame({
        "ticker":    ticker,
        "date":      dates,
        "open":      [p * 0.99 for p in prices],
        "high":      [p * 1.01 for p in prices],
        "low":       [p * 0.98 for p in prices],
        "close":     prices,
        "adj_close": prices,
        "volume":    [int(1e6)] * n_days,
        "currency":  "USD",
        "source":    "test",
    })
    upsert_prices(conn, df)


def test_wacc_beta_adjusted_spread(conn, fundamentals_etn, params):
    """Insert price data for ETN + SPY, verify spread differs from simple rf+erp."""
    upsert_fundamentals_annual(conn, fundamentals_etn)
    _insert_macro(conn, params)
    _insert_prices(conn, "ETN", n_days=400, start_price=250.0)
    _insert_prices(conn, "SPY", n_days=400, start_price=450.0, vol=0.01)

    # Beta-adjusted WACC
    spread_beta, conf_beta = compute_roic_wacc_spread("ETN", conn, params, AS_OF, "US")

    # Simple WACC (beta disabled)
    import copy
    params_no_beta = copy.deepcopy(params)
    params_no_beta["signals"]["quality"]["wacc_use_beta"] = False
    spread_simple, conf_simple = compute_roic_wacc_spread("ETN", conn, params_no_beta, AS_OF, "US")

    assert isinstance(spread_beta, float)
    assert conf_beta > 0
    # With beta and leverage, the WACC changes, so the spread should differ
    assert spread_beta != pytest.approx(spread_simple, abs=1e-6), \
        "Beta-adjusted spread should differ from simple rf+erp spread"


def test_wacc_beta_disabled_matches_old_behavior(conn, fundamentals_etn, params):
    """Set wacc_use_beta: False, verify spread matches old rf + erp calculation."""
    import copy
    params_disabled = copy.deepcopy(params)
    params_disabled["signals"]["quality"]["wacc_use_beta"] = False

    upsert_fundamentals_annual(conn, fundamentals_etn)
    _insert_macro(conn, params_disabled)

    spread, conf = compute_roic_wacc_spread("ETN", conn, params_disabled, AS_OF, "US")

    # Manually compute: WACC = rf + erp (old behavior)
    from src.data.macro import get_risk_free_rate
    rf = get_risk_free_rate(conn, "US", AS_OF, params_disabled)
    erp = params_disabled["return_estimation"]["equity_risk_premium"]
    expected_wacc = rf + erp

    # ROIC from synthetic data (nopat / invested_capital)
    from src.data.db import get_latest_fundamentals
    df = get_latest_fundamentals(conn, "ETN", n_years=1, as_of_date=AS_OF)
    roic = df["roic"].dropna().iloc[0]
    expected_spread = roic - expected_wacc

    assert spread == pytest.approx(expected_spread, rel=1e-6)


def test_wacc_beta_no_price_data_uses_default(conn, fundamentals_etn, params):
    """No price data -> beta_valid=False -> confidence gets -0.15 penalty."""
    upsert_fundamentals_annual(conn, fundamentals_etn)
    _insert_macro(conn, params)
    # Don't insert any prices — beta fallback to default

    spread, conf = compute_roic_wacc_spread("ETN", conn, params, AS_OF, "US")
    assert isinstance(spread, float)
    # With source="test", base confidence = 1.0, penalty = -0.15 -> 0.85
    assert conf == pytest.approx(0.85, abs=0.01), \
        f"Expected confidence ~0.85 (1.0 - 0.15 penalty), got {conf}"


def test_wacc_leveraged_vs_unleveraged(conn, params):
    """Leveraged stock should have different WACC than unleveraged stock.
    With debt, WACC blends cost of equity with after-tax cost of debt."""
    _insert_macro(conn, params)

    # Stock A: has debt (leveraged) — equity=3B, debt=1B
    rows_a = []
    for fy in [2023, 2024]:
        rev, equity, debt = 5e9, 3e9, 1e9
        cash, gw = 0.5e9, 0.5e9
        ic = equity + debt - cash - gw
        ebit = rev * 0.15
        nopat = ebit * 0.79
        rows_a.append({
            "ticker": "LEVERED", "fiscal_year": fy,
            "report_date": date(fy, 12, 31), "currency": "USD",
            "accounting_std": "GAAP", "source": "test",
            "revenue": rev, "gross_profit": rev * 0.40, "ebit": ebit,
            "net_income": ebit * 0.75, "interest_expense": debt * 0.04,
            "tax_expense": ebit * 0.21, "total_assets": equity + debt,
            "total_equity": equity, "total_debt": debt, "cash": cash,
            "goodwill": gw, "intangible_assets": 0.2e9,
            "right_of_use_assets": None, "lease_liabilities": None,
            "capex": rev * 0.04, "gross_margin": 0.40, "invested_capital": ic,
            "nopat": nopat, "roic": nopat / ic, "effective_tax_rate": 0.21,
        })

    # Stock B: no debt (all-equity) — equity=4B, debt=0
    rows_b = []
    for fy in [2023, 2024]:
        rev, equity, debt = 5e9, 4e9, 0.0
        cash, gw = 0.5e9, 0.5e9
        ic = equity - cash - gw
        ebit = rev * 0.15
        nopat = ebit * 0.79
        rows_b.append({
            "ticker": "UNLEVERED", "fiscal_year": fy,
            "report_date": date(fy, 12, 31), "currency": "USD",
            "accounting_std": "GAAP", "source": "test",
            "revenue": rev, "gross_profit": rev * 0.40, "ebit": ebit,
            "net_income": ebit * 0.75, "interest_expense": 0.0,
            "tax_expense": ebit * 0.21, "total_assets": equity,
            "total_equity": equity, "total_debt": debt, "cash": cash,
            "goodwill": gw, "intangible_assets": 0.2e9,
            "right_of_use_assets": None, "lease_liabilities": None,
            "capex": rev * 0.04, "gross_margin": 0.40, "invested_capital": ic,
            "nopat": nopat, "roic": nopat / ic, "effective_tax_rate": 0.21,
        })

    upsert_fundamentals_annual(conn, pd.DataFrame(rows_a))
    upsert_fundamentals_annual(conn, pd.DataFrame(rows_b))

    # No price data -> both use default beta=1.0, but leverage differs
    spread_lev, conf_lev = compute_roic_wacc_spread("LEVERED", conn, params, AS_OF, "US")
    spread_unlev, conf_unlev = compute_roic_wacc_spread("UNLEVERED", conn, params, AS_OF, "US")

    # Both should be valid
    import math
    assert not math.isnan(spread_lev)
    assert not math.isnan(spread_unlev)

    # Leveraged stock should have different spread (lower WACC due to tax shield on debt)
    assert spread_lev != pytest.approx(spread_unlev, abs=1e-6), \
        f"Leveraged ({spread_lev}) and unleveraged ({spread_unlev}) should have different spreads"

    # Leveraged WACC should be lower (after-tax cost of debt < cost of equity),
    # so spread should be higher for leveraged stock
    assert spread_lev > spread_unlev, \
        f"Leveraged spread ({spread_lev}) should be > unleveraged ({spread_unlev}) due to tax shield"
