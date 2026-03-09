"""
Shared pytest fixtures.
All DB fixtures use in-memory DuckDB (:memory:) — no disk I/O, no cleanup needed.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
from src.data.db import initialize_schema


# ---------------------------------------------------------------------------
# Core DB fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def conn():
    """In-memory DuckDB connection with full schema initialized."""
    c = duckdb.connect(":memory:")
    initialize_schema(c)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Sample universe
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_universe():
    """Small universe DataFrame for tests — no YAML I/O."""
    return pd.DataFrame([
        {
            "ticker": "ETN",
            "eodhd_ticker": None,
            "name": "Eaton Corp",
            "exchange": "NYSE",
            "region": "US",
            "currency": "USD",
            "accounting_std": "GAAP",
            "buckets": ["grid", "ai_infra"],
            "primary_bucket": "grid",
            "isin": None,
            "trends_keyword": "electrical grid infrastructure",
        },
        {
            "ticker": "RHM.DE",
            "eodhd_ticker": "RHM.XETRA",
            "name": "Rheinmetall",
            "exchange": "XETRA",
            "region": "EU",
            "currency": "EUR",
            "accounting_std": "IFRS",
            "buckets": ["defense"],
            "primary_bucket": "defense",
            "isin": "DE0007030009",
            "trends_keyword": "Rheinmetall defense",
        },
        {
            "ticker": "CCJ",
            "eodhd_ticker": None,
            "name": "Cameco",
            "exchange": "NYSE",
            "region": "US",
            "currency": "USD",
            "accounting_std": "GAAP",
            "buckets": ["nuclear", "critical_materials"],
            "primary_bucket": "nuclear",
            "isin": None,
            "trends_keyword": "uranium nuclear",
        },
    ])


# ---------------------------------------------------------------------------
# Sample price data
# ---------------------------------------------------------------------------

def _make_prices(ticker: str, n_days: int = 400,
                 start_price: float = 100.0,
                 currency: str = "USD",
                 vol: float = 0.02) -> pd.DataFrame:
    """Generate synthetic OHLCV price rows for a single ticker."""
    rng = np.random.default_rng(seed=hash(ticker) % (2**32))
    dates = [date(2022, 1, 3) + timedelta(days=i) for i in range(n_days)]
    prices = [start_price]
    for _ in range(n_days - 1):
        prices.append(prices[-1] * (1 + rng.normal(0.0003, vol)))

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


@pytest.fixture
def prices_etn():
    return _make_prices("ETN", n_days=400, start_price=250.0)


@pytest.fixture
def prices_rhm():
    return _make_prices("RHM.DE", n_days=400, start_price=300.0, currency="EUR")


@pytest.fixture
def prices_ccj():
    return _make_prices("CCJ", n_days=400, start_price=50.0)


@pytest.fixture
def prices_spy():
    """Broad market index for relative-strength tests."""
    return _make_prices("SPY", n_days=400, start_price=450.0, vol=0.01)


@pytest.fixture
def all_prices(prices_etn, prices_rhm, prices_ccj, prices_spy):
    return pd.concat([prices_etn, prices_rhm, prices_ccj, prices_spy], ignore_index=True)


# ---------------------------------------------------------------------------
# Sample fundamental data
# ---------------------------------------------------------------------------

def _make_fundamentals(ticker: str, accounting_std: str,
                        currency: str, n_years: int = 5) -> pd.DataFrame:
    """Generate synthetic annual fundamental rows."""
    base_year = 2024
    rows = []
    for i in range(n_years):
        fy = base_year - i
        revenue = 5e9 * (1.08 ** (n_years - i - 1))  # 8% revenue CAGR
        gp      = revenue * 0.40
        ebit    = revenue * 0.15
        equity  = 3e9
        debt    = 1e9
        cash    = 0.5e9
        gw      = 0.5e9
        ic      = equity + debt - cash - gw
        nopat   = ebit * (1 - 0.21)
        row = {
            "ticker":              ticker,
            "fiscal_year":         fy,
            "report_date":         date(fy, 12, 31),
            "currency":            currency,
            "accounting_std":      accounting_std,
            "source":              "test",
            "revenue":             revenue,
            "gross_profit":        gp,
            "ebit":                ebit,
            "net_income":          ebit * 0.75,
            "interest_expense":    debt * 0.04,
            "tax_expense":         ebit * 0.21,
            "total_assets":        equity + debt,
            "total_equity":        equity,
            "total_debt":          debt,
            "cash":                cash,
            "goodwill":            gw,
            "intangible_assets":   0.2e9,
            "right_of_use_assets": 0.1e9 if accounting_std == "IFRS" else None,
            "lease_liabilities":   0.1e9 if accounting_std == "IFRS" else None,
            "capex":               revenue * 0.04,
            "gross_margin":        gp / revenue,
            "invested_capital":    ic,
            "nopat":               nopat,
            "roic":                nopat / ic,
            "effective_tax_rate":  0.21,
        }
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def fundamentals_etn():
    return _make_fundamentals("ETN", "GAAP", "USD")


@pytest.fixture
def fundamentals_rhm():
    return _make_fundamentals("RHM.DE", "IFRS", "EUR")


# ---------------------------------------------------------------------------
# Sample macro data
# ---------------------------------------------------------------------------

@pytest.fixture
def macro_us_10y(conn):
    """Insert a US 10Y yield series into the DB and return the conn."""
    from src.data.db import upsert_macro
    rows = []
    for i in range(730):
        d = date(2023, 1, 1) + timedelta(days=i)
        rows.append({"date": d.isoformat(), "value": 4.3 + 0.001 * i, "source": "fred"})
    df = pd.DataFrame(rows)
    upsert_macro(conn, "US_10Y", df)
    upsert_macro(conn, "EU_10Y_DE", df.assign(value=2.8))
    return conn


# ---------------------------------------------------------------------------
# Params fixture (mirrors config/params.yaml structure)
# ---------------------------------------------------------------------------

@pytest.fixture
def params():
    """Mirrors config/params.yaml — keep in sync when params change."""
    return {
        "kelly": {
            "fraction":       0.25,
            "max_position":   0.08,
            "max_bucket":     0.35,
            "min_position":   0.02,
            "cash_reserve":   0.10,
            "aum_eur":        50_000_000,
            "impact_scaling": 1.0,
        },
        "signals": {
            "entry_threshold":           0.30,
            "crowding_entry_max":        0.40,
            "crowding_exit_threshold":   0.75,
            "quality_exit_threshold":    0.25,
            "composite_decay_pct":       0.20,
            "min_composite_confidence":  0.40,
            "crowding": {
                "autocorr_window":      30,
                "autocorr_baseline":    120,
                "absorption_window":    60,
                "csd_omega1":           0.5,
                "csd_omega2":           0.5,
                "csd_omega3":           0.20,
                "csd_omega_etf":        0.15,
                "csd_omega_short":      0.10,
                "trends_window_recent": 30,
                "trends_window_baseline": 90,
                "etf_corr_window":      60,
                "etf_corr_baseline":    120,
                "etf_corr_max_delta":   0.30,
                "short_interest_window": 30,
                "short_interest_high":  0.25,
                "short_interest_low":   0.05,
            },
            "quality": {
                "roic_wacc_spread_min": 0.0,
                "roic_wacc_spread_max": 0.20,
                "margin_snr_min":       2.0,
                "margin_snr_max":       10.0,
                "gamma":                2.0,
                "convexity_max":        0.50,
                "lookback_years":       5,
            },
        },
        "physical": {
            "eroei_kappa":               10.0,
            "ecs_crit_percentile":       0.70,
            "ecs_lookback_months":       60,
            "us_energy_ppi_series":      "US_PPIENG",
            "bucket_fallback_confidence": 0.40,
        },
        "return_estimation": {
            "equity_risk_premium":  0.05,
            "theta_risk_premium":   0.30,
        },
        "data": {
            "price_staleness_days":       1,
            "fundamental_staleness_days": 90,
            "macro_staleness_days":       7,
            "trends_staleness_days":      7,
            "lookback_prices_days":       756,
            "price_history_years":        5,
        },
        "macro": {
            "eu_risk_free_series": "EU_10Y_DE",
            "us_risk_free_series": "US_10Y",
            "ca_risk_free_series": "CA_10Y",
            "eu_cpi_series":       "EU_HICP_YOY",
            "eu_ppi_series":       "EU_PPI_YOY",
            "us_cpi_series":       "US_CPI_YOY",
            "us_ppi_series":       "US_PPI_YOY",
        },
        "sector_etfs": {
            "US|grid":    "XLI",
            "EU|defense": "NATO.L",
            "US|nuclear": "URA",
        },
        "market_indices": {"US": "SPY", "EU": "EXW1.DE", "CA": "SPY"},
        "reporting": {
            "output_dir":              "reports",
            "top_candidates_count":    10,
            "watch_crowding_threshold": 0.55,
        },
    }
