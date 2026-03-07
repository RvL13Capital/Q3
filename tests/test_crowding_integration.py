"""
Integration tests for the crowding signal (Signal 3).

Uses in-memory DuckDB with synthetic price data; no network calls.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.db import upsert_prices
from src.signals.crowding import (
    compute_etf_correlation,
    compute_relative_strength,
    compute_trends_score,
    compute_short_interest_score,
    compute_crowding_score,
    batch_crowding_scores,
)

AS_OF = "2023-01-13"   # last date inside our 400-day synthetic price window


# ---------------------------------------------------------------------------
# ETF correlation
# ---------------------------------------------------------------------------

def test_etf_correlation_returns_valid(conn, prices_etn, prices_spy, params):
    """With sufficient price history the score must be in [0, 1]."""
    upsert_prices(conn, prices_etn)
    upsert_prices(conn, prices_spy)
    score, conf = compute_etf_correlation("ETN", "SPY", conn, params, AS_OF)
    assert 0.0 <= score <= 1.0
    assert conf > 0.5, "400 days of data should give high confidence"


def test_etf_correlation_high_for_identical_returns(conn, prices_etn, params):
    """Two series with identical returns should yield correlation ≈ 1 → score ≈ 1."""
    # Insert ETN twice under different tickers → returns are identical
    etf = prices_etn.copy()
    etf["ticker"] = "ETN_ETF"
    upsert_prices(conn, prices_etn)
    upsert_prices(conn, etf)
    score, conf = compute_etf_correlation("ETN", "ETN_ETF", conn, params, AS_OF)
    assert score == pytest.approx(1.0, abs=0.01)


def test_etf_correlation_missing_stock(conn, prices_etn, params):
    """Missing stock data → neutral (0.5) with zero confidence."""
    upsert_prices(conn, prices_etn)
    score, conf = compute_etf_correlation("MISSING", "ETN", conn, params, AS_OF)
    assert score == 0.5
    assert conf == 0.0


def test_etf_correlation_missing_etf(conn, prices_etn, params):
    upsert_prices(conn, prices_etn)
    score, conf = compute_etf_correlation("ETN", "MISSING_ETF", conn, params, AS_OF)
    assert score == 0.5
    assert conf == 0.0


def test_etf_correlation_no_sector_etf(conn, prices_etn, params):
    """If sector_etf is None (no mapping), caller handles it; signal = 0.5."""
    # batch_crowding_scores passes None when no ETF is mapped; tested via
    # compute_crowding_score with sector_etf=None
    upsert_prices(conn, prices_etn)
    result = compute_crowding_score(
        "ETN", None, None, "SPY", conn, params, AS_OF, "US"
    )
    # With no ETF and no trends and no short data, only RS can contribute
    assert 0.0 <= result["crowding_score"] <= 1.0


def test_etf_correlation_low_data(conn, prices_etn, params):
    """< 20 days of data → confidence below 1."""
    small = prices_etn.head(15)
    etf = small.copy()
    etf["ticker"] = "ETN_ETF"
    upsert_prices(conn, small)
    upsert_prices(conn, etf)
    as_of = str(small["date"].max()) if not hasattr(small["date"].iloc[0], "isoformat") \
        else small["date"].max().isoformat()
    score, conf = compute_etf_correlation("ETN", "ETN_ETF", conn, params, as_of)
    assert conf < 1.0


# ---------------------------------------------------------------------------
# Relative strength
# ---------------------------------------------------------------------------

def test_relative_strength_same_asset(conn, prices_etn, params):
    """Stock returns equal to market → RS = 1.0 → score in neutral zone."""
    # Use ETN prices for both ticker and index (different DB tickers, same data)
    idx = prices_etn.copy()
    idx["ticker"] = "SPY"
    upsert_prices(conn, prices_etn)
    upsert_prices(conn, idx)
    score, conf = compute_relative_strength("ETN", "SPY", conn, params, AS_OF)
    # RS = 1.0 → (1.0 - 0.5) / 2.5 = 0.2
    assert score == pytest.approx(0.2, abs=0.05)
    assert conf > 0


def test_relative_strength_outperformance(conn, prices_etn, prices_spy, params):
    """
    If the stock strongly outperforms SPY, RS > 1 and score > neutral.
    Our 2% vol stock vs 1% vol SPY won't reliably outperform, so just check bounds.
    """
    upsert_prices(conn, prices_etn)
    upsert_prices(conn, prices_spy)
    score, conf = compute_relative_strength("ETN", "SPY", conn, params, AS_OF)
    assert 0.0 <= score <= 1.0
    assert conf > 0


def test_relative_strength_missing_ticker(conn, prices_spy, params):
    upsert_prices(conn, prices_spy)
    score, conf = compute_relative_strength("MISSING", "SPY", conn, params, AS_OF)
    assert score == 0.5
    assert conf == 0.0


def test_relative_strength_missing_index(conn, prices_etn, params):
    upsert_prices(conn, prices_etn)
    score, conf = compute_relative_strength("ETN", "MISSING_IDX", conn, params, AS_OF)
    assert score == 0.5
    assert conf == 0.0


# ---------------------------------------------------------------------------
# Google Trends score
# ---------------------------------------------------------------------------

def _insert_trends(conn, keyword: str, value: float = 50.0, n_weeks: int = 60):
    """Insert synthetic trends data directly via SQL."""
    from datetime import date as dt, timedelta
    from src.data.db import upsert_trends
    dates = [dt(2022, 1, 1) + timedelta(weeks=i) for i in range(n_weeks)]
    df = pd.DataFrame({
        "keyword": keyword,
        "date":    [d.isoformat() for d in dates],
        "score":   [value] * n_weeks,
        "geo":     "US",
    })
    upsert_trends(conn, df)


def test_trends_score_no_data(conn):
    """No trends data → neutral 0.5, confidence 0."""
    score, conf = compute_trends_score("nonexistent keyword", conn, AS_OF)
    assert score == 0.5
    assert conf == 0.0


def test_trends_score_flat_series(conn):
    """Perfectly flat series → all values at same percentile → neutral."""
    _insert_trends(conn, "test keyword", value=50.0, n_weeks=60)
    score, conf = compute_trends_score("test keyword", conn, AS_OF)
    # p20 = p90 = 50 → degenerate case → 0.5
    assert score == pytest.approx(0.5, abs=0.01)
    assert conf > 0


def test_trends_score_high_interest(conn):
    """Last value is max → score near 1.0 (highly hyped)."""
    from datetime import date as dt, timedelta
    from src.data.db import upsert_trends
    dates = [dt(2022, 1, 1) + timedelta(weeks=i) for i in range(55)]
    scores = list(range(1, 56))  # linearly increasing; last value is max
    df = pd.DataFrame({
        "keyword": "hot keyword",
        "date":    [d.isoformat() for d in dates],
        "score":   scores,
        "geo":     "US",
    })
    upsert_trends(conn, df)
    score, conf = compute_trends_score("hot keyword", conn, AS_OF)
    # Current (55) is above p90 → score = 1.0
    assert score == pytest.approx(1.0)


def test_trends_score_low_interest(conn):
    """Last value is min → score near 0 (under the radar)."""
    from datetime import date as dt, timedelta
    from src.data.db import upsert_trends
    dates = [dt(2022, 1, 1) + timedelta(weeks=i) for i in range(55)]
    scores = list(range(55, 0, -1))  # linearly decreasing; last value is min
    df = pd.DataFrame({
        "keyword": "cold keyword",
        "date":    [d.isoformat() for d in dates],
        "score":   scores,
        "geo":     "US",
    })
    upsert_trends(conn, df)
    score, conf = compute_trends_score("cold keyword", conn, AS_OF)
    assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Short interest
# ---------------------------------------------------------------------------

def test_short_interest_eu_neutral(conn):
    """EU stocks → always neutral, zero confidence."""
    score, conf = compute_short_interest_score("RHM.DE", conn, "EU")
    assert score == 0.5
    assert conf == 0.0


def test_short_interest_us_no_yfinance(conn, monkeypatch):
    """yfinance import fails → neutral, zero confidence."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "yfinance":
            raise ImportError("mocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    score, conf = compute_short_interest_score("ETN", conn, "US")
    assert score == 0.5
    assert conf == 0.0


def test_short_interest_us_no_data(conn, monkeypatch):
    """yfinance returns None for shortPercentOfFloat → neutral."""
    import types

    class FakeTicker:
        @property
        def info(self):
            return {"shortPercentOfFloat": None}

    fake_yf = types.ModuleType("yfinance")
    fake_yf.Ticker = lambda t: FakeTicker()
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

    score, conf = compute_short_interest_score("ETN", conn, "US")
    assert score == 0.5
    assert conf == 0.0


def test_short_interest_us_low_short(conn, monkeypatch):
    """Very low short interest → score near 0 (everyone is long → crowded)."""
    import types

    class FakeTicker:
        @property
        def info(self):
            return {"shortPercentOfFloat": 0.005}  # 0.5%

    fake_yf = types.ModuleType("yfinance")
    fake_yf.Ticker = lambda t: FakeTicker()
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

    score, conf = compute_short_interest_score("ETN", conn, "US")
    assert score < 0.1


def test_short_interest_us_high_short(conn, monkeypatch):
    """High short interest → score near 1 (contrarian setup → anti-crowded)."""
    import types

    class FakeTicker:
        @property
        def info(self):
            return {"shortPercentOfFloat": 0.20}  # 20%

    fake_yf = types.ModuleType("yfinance")
    fake_yf.Ticker = lambda t: FakeTicker()
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

    score, conf = compute_short_interest_score("ETN", conn, "US")
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Composite crowding score
# ---------------------------------------------------------------------------

def test_crowding_score_bounds(conn, prices_etn, prices_spy, params):
    upsert_prices(conn, prices_etn)
    upsert_prices(conn, prices_spy)
    result = compute_crowding_score(
        "ETN",
        None,    # no trends keyword
        "SPY",   # sector ETF = SPY (different from ticker)
        "SPY",
        conn, params, AS_OF, "US",
    )
    assert 0.0 <= result["crowding_score"] <= 1.0
    assert 0.0 <= result["crowding_confidence"] <= 1.0


def test_crowding_score_keys(conn, prices_etn, prices_spy, params):
    upsert_prices(conn, prices_etn)
    upsert_prices(conn, prices_spy)
    result = compute_crowding_score(
        "ETN", None, None, "SPY", conn, params, AS_OF, "US"
    )
    for key in ["ticker", "crowding_score", "crowding_confidence",
                "etf_correlation", "trends_norm", "short_pct"]:
        assert key in result


def test_crowding_score_no_price_data(conn, params, monkeypatch):
    """No price data → all sub-scores neutral → crowding = 0.5."""
    import types
    fake_yf = types.ModuleType("yfinance")

    class _Ticker:
        @property
        def info(self):
            return {}

    fake_yf.Ticker = lambda t: _Ticker()
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

    result = compute_crowding_score(
        "ETN", None, None, "SPY", conn, params, AS_OF, "US"
    )
    assert result["crowding_score"] == 0.5
    assert result["crowding_confidence"] == 0.0


# ---------------------------------------------------------------------------
# Batch crowding scores
# ---------------------------------------------------------------------------

def test_batch_crowding_scores(conn, sample_universe, all_prices, params):
    upsert_prices(conn, all_prices)
    result = batch_crowding_scores(sample_universe, conn, params, AS_OF)
    assert len(result) == len(sample_universe)
    assert (result["crowding_score"].between(0, 1)).all()


def test_batch_crowding_eu_stock_neutral_short(conn, sample_universe, all_prices, params):
    """EU stocks should have short_pct = 0.5 (no data → neutral)."""
    upsert_prices(conn, all_prices)
    result = batch_crowding_scores(sample_universe, conn, params, AS_OF)
    eu_row = result[result["ticker"] == "RHM.DE"]
    assert eu_row["short_pct"].iloc[0] == pytest.approx(0.5)
