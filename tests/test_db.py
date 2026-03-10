"""
DB layer roundtrip tests: schema init, upserts, query helpers, staleness checks.
All use the in-memory DuckDB `conn` fixture from conftest.py.
"""
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.db import (
    get_prices,
    get_latest_fundamentals,
    get_macro_value,
    get_macro_series,
    get_latest_signal_scores,
    get_latest_portfolio,
    get_position_entry_date,
    is_stale,
    log_fetch,
    upsert_prices,
    upsert_fundamentals_annual,
    upsert_macro,
    upsert_signal_scores,
    upsert_portfolio_snapshot,
)


# ────────────────────────────────────────────────────────────────────────────
# Schema / init
# ────────────────────────────────────────────────────────────────────────────

def test_schema_tables_exist(conn):
    """All expected tables must exist after initialize_schema."""
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).df()["table_name"].tolist()
    expected = [
        "prices", "fundamentals_annual", "fundamentals_quarterly",
        "macro_series", "google_trends", "short_interest",
        "signal_scores", "portfolio_snapshots", "exit_log", "fetch_log",
    ]
    for t in expected:
        assert t in tables, f"Table '{t}' missing from schema"


def test_schema_idempotent(conn):
    """Calling initialize_schema twice on the same connection must not error."""
    from src.data.db import initialize_schema
    initialize_schema(conn)  # second call
    tables = conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_schema='main'"
    ).fetchone()[0]
    assert tables >= 10


# ────────────────────────────────────────────────────────────────────────────
# Price upsert / query roundtrips
# ────────────────────────────────────────────────────────────────────────────

def test_upsert_prices_basic(conn, prices_etn):
    rows = upsert_prices(conn, prices_etn)
    assert rows == len(prices_etn)


def test_upsert_prices_idempotent(conn, prices_etn):
    """Upserting the same rows twice must not create duplicates."""
    upsert_prices(conn, prices_etn)
    upsert_prices(conn, prices_etn)
    count = conn.execute("SELECT count(*) FROM prices WHERE ticker='ETN'").fetchone()[0]
    assert count == len(prices_etn)


def test_upsert_prices_update(conn, prices_etn):
    """Later upsert with changed adj_close must overwrite the old value."""
    upsert_prices(conn, prices_etn)
    modified = prices_etn.copy()
    modified["adj_close"] = 999.0
    upsert_prices(conn, modified)
    result = conn.execute(
        "SELECT adj_close FROM prices WHERE ticker='ETN' LIMIT 1"
    ).fetchone()
    assert result[0] == pytest.approx(999.0)


def test_get_prices_wide_format(conn, prices_etn, prices_ccj):
    upsert_prices(conn, prices_etn)
    upsert_prices(conn, prices_ccj)
    df = get_prices(conn, ["ETN", "CCJ"], "2022-01-01", "2023-12-31")
    assert not df.empty
    assert "ETN" in df.columns
    assert "CCJ" in df.columns
    assert df.index.name == "date"


def test_get_prices_adjusted_vs_close(conn, prices_etn):
    upsert_prices(conn, prices_etn)
    df_adj   = get_prices(conn, ["ETN"], "2022-01-01", "2023-12-31", adjusted=True)
    df_close = get_prices(conn, ["ETN"], "2022-01-01", "2023-12-31", adjusted=False)
    # For our synthetic data adj_close == close, but both should be non-empty
    assert not df_adj.empty
    assert not df_close.empty


def test_get_prices_date_filter(conn, prices_etn):
    upsert_prices(conn, prices_etn)
    df = get_prices(conn, ["ETN"], "2022-01-03", "2022-01-10")
    # Only a small window
    assert len(df) <= 10


def test_get_prices_missing_ticker(conn, prices_etn):
    upsert_prices(conn, prices_etn)
    df = get_prices(conn, ["NONEXISTENT"], "2022-01-01", "2023-12-31")
    assert df.empty or "NONEXISTENT" not in df.columns


# ────────────────────────────────────────────────────────────────────────────
# Fundamental upsert / query
# ────────────────────────────────────────────────────────────────────────────

def test_upsert_fundamentals_annual_basic(conn, fundamentals_etn):
    rows = upsert_fundamentals_annual(conn, fundamentals_etn)
    assert rows == len(fundamentals_etn)


def test_upsert_fundamentals_idempotent(conn, fundamentals_etn):
    upsert_fundamentals_annual(conn, fundamentals_etn)
    upsert_fundamentals_annual(conn, fundamentals_etn)
    count = conn.execute(
        "SELECT count(*) FROM fundamentals_annual WHERE ticker='ETN'"
    ).fetchone()[0]
    assert count == len(fundamentals_etn)


def test_get_latest_fundamentals_returns_newest_first(conn, fundamentals_etn):
    upsert_fundamentals_annual(conn, fundamentals_etn)
    df = get_latest_fundamentals(conn, "ETN", n_years=5)
    assert not df.empty
    years = df["fiscal_year"].tolist()
    assert years == sorted(years, reverse=True)  # newest first


def test_get_latest_fundamentals_n_years_limit(conn, fundamentals_etn):
    upsert_fundamentals_annual(conn, fundamentals_etn)
    df = get_latest_fundamentals(conn, "ETN", n_years=2)
    assert len(df) == 2


def test_get_latest_fundamentals_roic_positive(conn, fundamentals_etn):
    """Synthetic data should have positive ROIC (profitable company)."""
    upsert_fundamentals_annual(conn, fundamentals_etn)
    df = get_latest_fundamentals(conn, "ETN", n_years=1)
    assert df["roic"].iloc[0] > 0


def test_fundamentals_ifrs_right_of_use(conn, fundamentals_rhm):
    """IFRS rows should carry non-null right_of_use_assets."""
    upsert_fundamentals_annual(conn, fundamentals_rhm)
    df = get_latest_fundamentals(conn, "RHM.DE", n_years=1)
    assert df["right_of_use_assets"].iloc[0] is not None
    assert float(df["right_of_use_assets"].iloc[0]) > 0


# ────────────────────────────────────────────────────────────────────────────
# Macro upsert / query
# ────────────────────────────────────────────────────────────────────────────

def _make_macro_df(value: float = 4.3) -> pd.DataFrame:
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(365)]
    return pd.DataFrame({
        "date":   [d.isoformat() for d in dates],
        "value":  [value] * 365,
        "source": "fred",
    })


def test_upsert_macro_basic(conn):
    df = _make_macro_df()
    rows = upsert_macro(conn, "US_10Y", df)
    assert rows == 365


def test_upsert_macro_idempotent(conn):
    df = _make_macro_df()
    upsert_macro(conn, "US_10Y", df)
    upsert_macro(conn, "US_10Y", df)
    count = conn.execute(
        "SELECT count(*) FROM macro_series WHERE series_id='US_10Y'"
    ).fetchone()[0]
    assert count == 365


def test_get_macro_value_returns_latest_on_or_before(conn):
    df = _make_macro_df(4.3)
    upsert_macro(conn, "US_10Y", df)
    val = get_macro_value(conn, "US_10Y", "2024-06-15")
    assert val == pytest.approx(4.3)


def test_get_macro_value_none_if_no_data(conn):
    val = get_macro_value(conn, "NONEXISTENT", "2024-01-01")
    assert val is None


def test_get_macro_series_returns_series(conn):
    df = _make_macro_df(4.5)
    upsert_macro(conn, "US_10Y", df)
    series = get_macro_series(conn, "US_10Y", "2024-01-01", "2024-03-31")
    assert not series.empty
    assert abs(series - 4.5).max() < 1e-6


# ────────────────────────────────────────────────────────────────────────────
# Staleness checks
# ────────────────────────────────────────────────────────────────────────────

def test_is_stale_no_data(conn):
    """No data → always stale."""
    assert is_stale(conn, "prices", "NOTHERE", staleness_days=1) is True


def test_is_stale_fresh_data(conn, prices_etn):
    """Just-inserted data should not be stale with a 1-day window."""
    upsert_prices(conn, prices_etn)
    assert is_stale(conn, "prices", "ETN", staleness_days=1) is False


def test_is_stale_old_data(conn, prices_etn):
    """Data with fetched_at backdated by 2 days should be stale with 1-day window."""
    upsert_prices(conn, prices_etn)
    # Manually backdate fetched_at
    conn.execute("""
        UPDATE prices SET fetched_at = current_timestamp - INTERVAL '2 days'
        WHERE ticker = 'ETN'
    """)
    assert is_stale(conn, "prices", "ETN", staleness_days=1) is True


# ────────────────────────────────────────────────────────────────────────────
# Signal scores
# ────────────────────────────────────────────────────────────────────────────

def _make_signal_row(ticker: str, score_date: str, composite: float = 0.65) -> dict:
    return {
        "ticker":               ticker,
        "score_date":           score_date,
        "physical_raw":         2.0,
        "physical_norm":        0.667,
        "physical_confidence":  1.0,
        "quality_score":        0.70,
        "quality_confidence":   0.9,
        "roic_wacc_spread":     0.057,
        "margin_snr":           5.2,
        "inflation_convexity":  0.06,
        "crowding_score":       0.30,
        "crowding_confidence":  0.8,
        "etf_correlation":      0.35,
        "trends_norm":          0.25,
        "short_pct":            0.40,
        "composite_score":      composite,
        "composite_confidence": 0.90,
        "mu_estimate":          0.14,
        "sigma_estimate":       0.28,
        "kelly_fraction":       0.18,
        "kelly_25pct":          0.18,
        "entry_signal":         True,
        "exit_signal":          False,
    }


def test_upsert_signal_scores(conn):
    df = pd.DataFrame([
        _make_signal_row("ETN", "2025-01-05"),
        _make_signal_row("CCJ", "2025-01-05", composite=0.70),
    ])
    rows = upsert_signal_scores(conn, df)
    assert rows == 2


def test_get_latest_signal_scores(conn):
    df1 = pd.DataFrame([_make_signal_row("ETN", "2025-01-05")])
    df2 = pd.DataFrame([_make_signal_row("ETN", "2025-01-12", composite=0.72)])
    upsert_signal_scores(conn, df1)
    upsert_signal_scores(conn, df2)
    result = get_latest_signal_scores(conn, ["ETN"])
    assert len(result) == 1
    assert result.iloc[0]["composite_score"] == pytest.approx(0.72)


def test_get_latest_signal_scores_all_tickers(conn):
    df = pd.DataFrame([
        _make_signal_row("ETN", "2025-01-05"),
        _make_signal_row("CCJ", "2025-01-05", 0.60),
    ])
    upsert_signal_scores(conn, df)
    result = get_latest_signal_scores(conn)  # no filter
    assert set(result["ticker"].tolist()) == {"ETN", "CCJ"}


# ────────────────────────────────────────────────────────────────────────────
# Portfolio snapshots
# ────────────────────────────────────────────────────────────────────────────

def _make_portfolio_df(snapshot_date: str) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "snapshot_date":  snapshot_date,
            "ticker":         "ETN",
            "weight":         0.08,
            "composite_score": 0.65,
            "kelly_25pct":    0.18,
            "bucket_id":      "grid",
            "is_new_position": True,
            "rationale":      None,
        },
        {
            "snapshot_date":  snapshot_date,
            "ticker":         "CCJ",
            "weight":         0.07,
            "composite_score": 0.70,
            "kelly_25pct":    0.20,
            "bucket_id":      "nuclear",
            "is_new_position": True,
            "rationale":      None,
        },
    ])


def test_upsert_portfolio_snapshot(conn):
    df = _make_portfolio_df("2025-01-05")
    rows = upsert_portfolio_snapshot(conn, df)
    assert rows == 2


def test_get_latest_portfolio(conn):
    upsert_portfolio_snapshot(conn, _make_portfolio_df("2025-01-05"))
    upsert_portfolio_snapshot(conn, _make_portfolio_df("2025-01-12"))
    result = get_latest_portfolio(conn)
    # Should return the 2025-01-12 snapshot
    assert set(result["snapshot_date"].astype(str).unique()) == {"2025-01-12"}
    assert len(result) == 2


def test_get_latest_portfolio_empty(conn):
    result = get_latest_portfolio(conn)
    assert result.empty


def test_get_position_entry_date(conn):
    upsert_portfolio_snapshot(conn, _make_portfolio_df("2025-01-05"))
    upsert_portfolio_snapshot(conn, _make_portfolio_df("2025-01-12"))
    entry = get_position_entry_date(conn, "ETN")
    assert str(entry) == "2025-01-05"


# ────────────────────────────────────────────────────────────────────────────
# Fetch log
# ────────────────────────────────────────────────────────────────────────────

def test_log_fetch(conn):
    log_fetch(conn, "prices", "ETN", "success", rows_written=250)
    count = conn.execute(
        "SELECT count(*) FROM fetch_log WHERE module='prices' AND ticker='ETN'"
    ).fetchone()[0]
    assert count == 1


def test_log_fetch_multiple(conn):
    for i in range(5):
        log_fetch(conn, "macro", None, "success", rows_written=i)
    count = conn.execute(
        "SELECT count(*) FROM fetch_log WHERE module='macro'"
    ).fetchone()[0]
    assert count == 5


# ────────────────────────────────────────────────────────────────────────────
# Bitemporal t_k tests
# ────────────────────────────────────────────────────────────────────────────

from src.data.db import get_margin_history


class TestBitemporalSchema:
    """Verify bitemporal t_k column and query filtering."""

    def test_t_k_column_exists_after_schema_init(self, conn):
        """Schema migration adds t_k to all bitemporal tables."""
        for table in ["prices", "fundamentals_annual", "fundamentals_quarterly", "macro_series"]:
            cols = conn.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'").df()
            assert "t_k" in cols["column_name"].values, f"t_k missing from {table}"

    def test_macro_upsert_stores_explicit_t_k(self, conn):
        """When DataFrame includes t_k, it is stored in the DB."""
        df = pd.DataFrame({
            "date": ["2024-01-15", "2024-02-15"],
            "value": [4.0, 4.1],
            "source": "fred",
            "t_k": ["2024-01-20 09:00:00", "2024-02-20 09:00:00"],
        })
        upsert_macro(conn, "TEST_TK", df)
        result = conn.execute(
            "SELECT t_k FROM macro_series WHERE series_id = 'TEST_TK' ORDER BY date"
        ).df()
        assert result["t_k"].notna().all()
        assert len(result) == 2

    def test_get_macro_value_bitemporal_excludes_future_knowledge(self, conn):
        """With bitemporal=True, data with t_k after as_of_date is invisible."""
        df = pd.DataFrame({
            "date": ["2024-01-15", "2024-02-15"],
            "value": [4.0, 4.5],
            "source": "fred",
            "t_k": ["2024-01-20", "2024-02-20"],
        })
        upsert_macro(conn, "BT_RF", df)

        # As of Jan 25: should see Jan data (t_k=Jan 20) but not Feb (t_k=Feb 20)
        val = get_macro_value(conn, "BT_RF", "2024-01-25", bitemporal=True)
        assert val == pytest.approx(4.0)

        # As of Mar 1: both visible
        val_full = get_macro_value(conn, "BT_RF", "2024-03-01", bitemporal=True)
        assert val_full == pytest.approx(4.5)

    def test_get_macro_value_bitemporal_false_ignores_t_k(self, conn):
        """Default bitemporal=False preserves backward-compatible behavior."""
        df = pd.DataFrame({
            "date": ["2024-01-15"],
            "value": [3.9],
            "source": "fred",
            "t_k": ["2026-01-01"],  # far future t_k
        })
        upsert_macro(conn, "BT_COMPAT", df)
        val = get_macro_value(conn, "BT_COMPAT", "2024-06-01", bitemporal=False)
        assert val == pytest.approx(3.9)

    def test_get_macro_series_as_of_tk_filters(self, conn):
        """as_of_tk parameter filters out not-yet-published observations."""
        df = pd.DataFrame({
            "date": ["2024-01-15", "2024-02-15", "2024-03-15"],
            "value": [1.0, 2.0, 3.0],
            "source": "fred",
            "t_k": ["2024-01-20", "2024-02-20", "2024-03-20"],
        })
        upsert_macro(conn, "BT_SERIES", df)

        # Without as_of_tk: all 3 visible
        full = get_macro_series(conn, "BT_SERIES", "2024-01-01", "2024-12-31")
        assert len(full) == 3

        # With as_of_tk=Feb 25: only first 2 visible
        partial = get_macro_series(conn, "BT_SERIES", "2024-01-01", "2024-12-31",
                                   as_of_tk="2024-02-25")
        assert len(partial) == 2
        assert partial.iloc[-1] == pytest.approx(2.0)

    def test_get_latest_fundamentals_as_of_date_filters(self, conn):
        """Fundamentals with future report_date are excluded when as_of_date set."""
        rows = []
        for fy, rd in [(2022, "2023-03-01"), (2023, "2024-03-01"), (2024, "2025-03-01")]:
            rows.append({
                "ticker": "BTX",
                "fiscal_year": fy,
                "report_date": rd,
                "revenue": 1e9,
                "gross_profit": 4e8,
                "ebit": 1.5e8,
                "net_income": 1e8,
                "total_equity": 2e9,
                "total_debt": 5e8,
                "cash": 2e8,
                "goodwill": 1e8,
                "gross_margin": 0.40,
                "invested_capital": 2.2e9,
                "nopat": 1.2e8,
                "roic": 0.055,
                "effective_tax_rate": 0.21,
                "currency": "USD",
                "accounting_std": "GAAP",
                "source": "test",
            })
        df = pd.DataFrame(rows)
        upsert_fundamentals_annual(conn, df)

        # As of mid-2024: FY2022 + FY2023 visible, FY2024 not yet filed
        result = get_latest_fundamentals(conn, "BTX", n_years=5, as_of_date="2024-06-15")
        assert len(result) == 2
        assert set(result["fiscal_year"].tolist()) == {2022, 2023}

        # Without as_of_date: all 3 visible (backward compat)
        result_all = get_latest_fundamentals(conn, "BTX", n_years=5)
        assert len(result_all) == 3

    def test_get_margin_history_as_of_date_filters(self, conn):
        """Margin history respects bitemporal as_of_date."""
        rows = []
        for fy, rd in [(2021, "2022-03-15"), (2022, "2023-03-15"), (2023, "2024-03-15")]:
            rows.append({
                "ticker": "MHIST",
                "fiscal_year": fy,
                "report_date": rd,
                "revenue": 1e9,
                "gross_profit": fy * 1e6,
                "gross_margin": 0.35 + (fy - 2021) * 0.01,
                "ebit": 1e8,
                "net_income": 7e7,
                "total_equity": 1e9,
                "total_debt": 3e8,
                "cash": 1e8,
                "goodwill": 5e7,
                "invested_capital": 1.15e9,
                "nopat": 8e7,
                "roic": 0.07,
                "effective_tax_rate": 0.21,
                "currency": "USD",
                "accounting_std": "GAAP",
                "source": "test",
            })
        df = pd.DataFrame(rows)
        upsert_fundamentals_annual(conn, df)

        # As of Feb 2024: FY2023 not yet filed (report_date=2024-03-15)
        result = get_margin_history(conn, "MHIST", n_years=5, as_of_date="2024-02-01")
        assert len(result) == 2
        assert 2023 not in result["fiscal_year"].tolist()

    def test_ghost_state_macro_revision(self, conn):
        """Macro revision: same date, different t_k → latest-known-at-query-time wins."""
        # Initial release: CPI = 3.1% published on Jan 20
        df1 = pd.DataFrame({
            "date": ["2024-01-15"],
            "value": [3.1],
            "source": "fred",
            "t_k": ["2024-01-20"],
        })
        upsert_macro(conn, "CPI_REV", df1)

        # Revision: same date, CPI revised to 3.3% published on Feb 15
        df2 = pd.DataFrame({
            "date": ["2024-01-15"],
            "value": [3.3],
            "source": "fred",
            "t_k": ["2024-02-15"],
        })
        upsert_macro(conn, "CPI_REV", df2)

        # After revision: shows revised value (INSERT OR REPLACE overwrites)
        val = get_macro_value(conn, "CPI_REV", "2024-03-01", bitemporal=True)
        assert val == pytest.approx(3.3)

        # Before revision published: t_k filter correctly excludes revised value.
        # With current INSERT OR REPLACE, the original row (t_k=Jan 20) is
        # overwritten by the revision (t_k=Feb 15). So querying before Feb 15
        # returns None — the original value is lost.
        #
        # Phase 2.5 TODO: migrate to append-only ghost states (PK includes t_k)
        # so both versions coexist. Then this query should return 3.1 (original).
        val_pre = get_macro_value(conn, "CPI_REV", "2024-02-01", bitemporal=True)
        assert val_pre is None  # expected: original destroyed by overwrite
