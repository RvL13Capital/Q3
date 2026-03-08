"""
DuckDB database layer: schema creation, upserts, and query helpers.
All other modules import from here — never open raw DuckDB connections elsewhere.
"""
import os
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional

import duckdb
import pandas as pd

DB_PATH = Path(__file__).parent.parent.parent / "data" / "q3.duckdb"


def get_connection(db_path: str | Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection. Creates DB + schema on first call."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    initialize_schema(conn)
    return conn


def initialize_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all tables if they don't exist. Idempotent."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker      VARCHAR NOT NULL,
            date        DATE NOT NULL,
            open        DOUBLE,
            high        DOUBLE,
            low         DOUBLE,
            close       DOUBLE,
            adj_close   DOUBLE NOT NULL,
            volume      BIGINT,
            currency    VARCHAR NOT NULL,
            source      VARCHAR NOT NULL,
            fetched_at  TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (ticker, date)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals_annual (
            ticker              VARCHAR NOT NULL,
            fiscal_year         INTEGER NOT NULL,
            report_date         DATE,
            revenue             DOUBLE,
            gross_profit        DOUBLE,
            ebit                DOUBLE,
            net_income          DOUBLE,
            interest_expense    DOUBLE,
            tax_expense         DOUBLE,
            total_assets        DOUBLE,
            total_equity        DOUBLE,
            total_debt          DOUBLE,
            cash                DOUBLE,
            goodwill            DOUBLE,
            intangible_assets   DOUBLE,
            right_of_use_assets DOUBLE,
            lease_liabilities   DOUBLE,
            capex               DOUBLE,
            gross_margin        DOUBLE,
            invested_capital    DOUBLE,
            nopat               DOUBLE,
            roic                DOUBLE,
            effective_tax_rate  DOUBLE,
            currency            VARCHAR NOT NULL,
            accounting_std      VARCHAR NOT NULL,
            source              VARCHAR NOT NULL,
            fetched_at          TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (ticker, fiscal_year)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals_quarterly (
            ticker          VARCHAR NOT NULL,
            fiscal_year     INTEGER NOT NULL,
            fiscal_quarter  INTEGER NOT NULL,
            report_date     DATE,
            revenue         DOUBLE,
            gross_profit    DOUBLE,
            gross_margin    DOUBLE,
            ebit            DOUBLE,
            net_income      DOUBLE,
            currency        VARCHAR NOT NULL,
            accounting_std  VARCHAR NOT NULL,
            source          VARCHAR NOT NULL,
            fetched_at      TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (ticker, fiscal_year, fiscal_quarter)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS macro_series (
            series_id   VARCHAR NOT NULL,
            date        DATE NOT NULL,
            value       DOUBLE NOT NULL,
            unit        VARCHAR,
            source      VARCHAR NOT NULL,
            fetched_at  TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (series_id, date)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS google_trends (
            keyword     VARCHAR NOT NULL,
            date        DATE NOT NULL,
            score       INTEGER,
            geo         VARCHAR DEFAULT 'US',
            fetched_at  TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (keyword, date, geo)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS short_interest (
            ticker          VARCHAR NOT NULL,
            date            DATE NOT NULL,
            short_ratio     DOUBLE,
            short_pct_float DOUBLE,
            source          VARCHAR NOT NULL,
            fetched_at      TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (ticker, date)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS signal_scores (
            ticker              VARCHAR NOT NULL,
            score_date          DATE NOT NULL,
            physical_raw        DOUBLE,
            physical_norm       DOUBLE,
            physical_confidence DOUBLE,
            quality_score       DOUBLE,
            quality_confidence  DOUBLE,
            roic_wacc_spread    DOUBLE,
            margin_snr          DOUBLE,
            inflation_convexity DOUBLE,
            crowding_score      DOUBLE,
            crowding_confidence DOUBLE,
            autocorr_delta      DOUBLE,
            absorption_delta    DOUBLE,
            composite_score     DOUBLE,
            composite_confidence DOUBLE,
            mu_estimate         DOUBLE,
            sigma_estimate      DOUBLE,
            kelly_fraction      DOUBLE,
            kelly_25pct         DOUBLE,
            entry_signal        BOOLEAN,
            exit_signal         BOOLEAN,
            PRIMARY KEY (ticker, score_date)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            snapshot_date   DATE NOT NULL,
            ticker          VARCHAR NOT NULL,
            weight          DOUBLE NOT NULL,
            composite_score DOUBLE,
            kelly_25pct     DOUBLE,
            bucket_id       VARCHAR,
            is_new_position BOOLEAN DEFAULT FALSE,
            rationale       VARCHAR,
            PRIMARY KEY (snapshot_date, ticker)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS exit_log (
            ticker          VARCHAR NOT NULL,
            exit_date       DATE NOT NULL,
            exit_reason     VARCHAR NOT NULL,
            crowding_at_exit DOUBLE,
            composite_at_exit DOUBLE,
            notes           VARCHAR,
            PRIMARY KEY (ticker, exit_date)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            id          INTEGER,
            fetch_date  TIMESTAMP DEFAULT current_timestamp,
            module      VARCHAR NOT NULL,
            ticker      VARCHAR,
            status      VARCHAR NOT NULL,
            message     VARCHAR,
            rows_written INTEGER DEFAULT 0
        )
    """)

    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS fetch_log_seq START 1
    """)


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------

def upsert_prices(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """
    Bulk upsert price records. df must have columns:
    ticker, date, adj_close, currency, source.
    Optional: open, high, low, close, volume.
    Returns number of rows written.
    """
    if df.empty:
        return 0
    required = {"ticker", "date", "adj_close", "currency", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"upsert_prices: missing columns {missing}")

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = None

    df = df[["ticker", "date", "open", "high", "low", "close",
             "adj_close", "volume", "currency", "source"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    conn.execute("""
        INSERT OR REPLACE INTO prices
            (ticker, date, open, high, low, close, adj_close, volume, currency, source)
        SELECT ticker, date, open, high, low, close, adj_close, volume, currency, source
        FROM df
    """)
    return len(df)


def get_prices(
    conn: duckdb.DuckDBPyConnection,
    tickers: list[str],
    start_date: str,
    end_date: str,
    adjusted: bool = True,
) -> pd.DataFrame:
    """
    Return wide-format price DataFrame: index=date, columns=tickers.
    Uses adj_close if adjusted=True, else close.
    """
    price_col = "adj_close" if adjusted else "close"
    tickers_str = ", ".join(f"'{t}'" for t in tickers)
    df = conn.execute(f"""
        SELECT ticker, date, {price_col} as price
        FROM prices
        WHERE ticker IN ({tickers_str})
          AND date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
    """).df()
    if df.empty:
        return pd.DataFrame()
    return df.pivot(index="date", columns="ticker", values="price")


def get_latest_price_date(conn: duckdb.DuckDBPyConnection, ticker: str) -> Optional[date]:
    """Return most recent date with price data for ticker."""
    result = conn.execute(
        "SELECT MAX(date) FROM prices WHERE ticker = ?", [ticker]
    ).fetchone()
    return result[0] if result else None


# ---------------------------------------------------------------------------
# Fundamental helpers
# ---------------------------------------------------------------------------

_ANNUAL_SCHEMA_COLS = [
    "ticker", "fiscal_year", "report_date", "revenue", "gross_profit", "ebit",
    "net_income", "interest_expense", "tax_expense", "total_assets", "total_equity",
    "total_debt", "cash", "goodwill", "intangible_assets", "right_of_use_assets",
    "lease_liabilities", "capex", "gross_margin", "invested_capital", "nopat",
    "roic", "effective_tax_rate", "currency", "accounting_std", "source",
]


def upsert_fundamentals_annual(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    # Only include columns that exist in the schema (exclude auto-default fetched_at)
    present = [c for c in _ANNUAL_SCHEMA_COLS if c in df.columns]
    df_insert = df[present].copy()
    cols_sql = ", ".join(present)
    conn.execute(f"""
        INSERT OR REPLACE INTO fundamentals_annual ({cols_sql})
        SELECT {cols_sql} FROM df_insert
    """)
    return len(df_insert)


def upsert_fundamentals_quarterly(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    conn.execute("""
        INSERT OR REPLACE INTO fundamentals_quarterly
        SELECT * FROM df
    """)
    return len(df)


def get_latest_fundamentals(
    conn: duckdb.DuckDBPyConnection,
    ticker: str,
    n_years: int = 5,
) -> pd.DataFrame:
    """Return up to n_years of annual fundamental rows for ticker, newest first."""
    return conn.execute("""
        SELECT *
        FROM fundamentals_annual
        WHERE ticker = ?
        ORDER BY fiscal_year DESC
        LIMIT ?
    """, [ticker, n_years]).df()


# ---------------------------------------------------------------------------
# Macro helpers
# ---------------------------------------------------------------------------

def upsert_macro(conn: duckdb.DuckDBPyConnection, series_id: str, df: pd.DataFrame) -> int:
    """df must have columns: date, value, source. Optional: unit."""
    if df.empty:
        return 0
    df = df.copy()
    df["series_id"] = series_id
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if "unit" not in df.columns:
        df["unit"] = None
    df = df[["series_id", "date", "value", "unit", "source"]]
    conn.execute("""
        INSERT OR REPLACE INTO macro_series (series_id, date, value, unit, source)
        SELECT series_id, date, value, unit, source FROM df
    """)
    return len(df)


def get_macro_value(
    conn: duckdb.DuckDBPyConnection,
    series_id: str,
    as_of_date: str,
) -> Optional[float]:
    """Return the most recent macro value on or before as_of_date."""
    result = conn.execute("""
        SELECT value FROM macro_series
        WHERE series_id = ? AND date <= ?
        ORDER BY date DESC LIMIT 1
    """, [series_id, as_of_date]).fetchone()
    return result[0] if result else None


def get_macro_series(
    conn: duckdb.DuckDBPyConnection,
    series_id: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Return macro time series as pd.Series indexed by date."""
    df = conn.execute("""
        SELECT date, value FROM macro_series
        WHERE series_id = ? AND date BETWEEN ? AND ?
        ORDER BY date
    """, [series_id, start_date, end_date]).df()
    if df.empty:
        return pd.Series(dtype=float)
    return df.set_index("date")["value"]


# ---------------------------------------------------------------------------
# Trends helpers
# ---------------------------------------------------------------------------

def upsert_trends(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if "geo" not in df.columns:
        df["geo"] = "US"
    conn.execute("""
        INSERT OR REPLACE INTO google_trends (keyword, date, score, geo)
        SELECT keyword, date, score, geo FROM df
    """)
    return len(df)


def get_trends(
    conn: duckdb.DuckDBPyConnection,
    keyword: str,
    start_date: str,
    end_date: str,
    geo: str = "US",
) -> pd.Series:
    df = conn.execute("""
        SELECT date, score FROM google_trends
        WHERE keyword = ? AND geo = ? AND date BETWEEN ? AND ?
        ORDER BY date
    """, [keyword, geo, start_date, end_date]).df()
    if df.empty:
        return pd.Series(dtype=float)
    return df.set_index("date")["score"].astype(float)


# ---------------------------------------------------------------------------
# Signal score helpers
# ---------------------------------------------------------------------------

_SIGNAL_SCORE_COLS = [
    "ticker", "score_date", "physical_raw", "physical_norm", "physical_confidence",
    "quality_score", "quality_confidence", "roic_wacc_spread", "margin_snr",
    "inflation_convexity", "crowding_score", "crowding_confidence",
    "autocorr_delta", "absorption_delta",
    "composite_score", "composite_confidence",
    "mu_estimate", "sigma_estimate", "kelly_fraction", "kelly_25pct",
    "entry_signal", "exit_signal",
]


def upsert_signal_scores(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    df = df.copy()
    df["score_date"] = pd.to_datetime(df["score_date"]).dt.date
    # Ensure columns match schema order (avoids SELECT * positional mismatch)
    present = [c for c in _SIGNAL_SCORE_COLS if c in df.columns]
    df_insert = df[present].copy()
    cols_sql = ", ".join(present)
    conn.execute(f"""
        INSERT OR REPLACE INTO signal_scores ({cols_sql})
        SELECT {cols_sql} FROM df_insert
    """)
    return len(df_insert)


def get_latest_signal_scores(
    conn: duckdb.DuckDBPyConnection,
    tickers: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Return the most recent signal score per ticker."""
    if tickers:
        tickers_str = ", ".join(f"'{t}'" for t in tickers)
        where = f"WHERE ticker IN ({tickers_str})"
    else:
        where = ""
    return conn.execute(f"""
        SELECT s.*
        FROM signal_scores s
        INNER JOIN (
            SELECT ticker, MAX(score_date) AS max_date
            FROM signal_scores {where}
            GROUP BY ticker
        ) latest ON s.ticker = latest.ticker AND s.score_date = latest.max_date
    """).df()


# ---------------------------------------------------------------------------
# Portfolio helpers
# ---------------------------------------------------------------------------

_PORTFOLIO_SNAPSHOT_COLS = [
    "snapshot_date", "ticker", "weight", "composite_score", "kelly_25pct",
    "bucket_id", "is_new_position", "rationale",
]


def upsert_portfolio_snapshot(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    df = df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
    present = [c for c in _PORTFOLIO_SNAPSHOT_COLS if c in df.columns]
    df_insert = df[present].copy()
    cols_sql = ", ".join(present)
    conn.execute(f"""
        INSERT OR REPLACE INTO portfolio_snapshots ({cols_sql})
        SELECT {cols_sql} FROM df_insert
    """)
    return len(df_insert)


def get_latest_portfolio(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Return the most recent portfolio snapshot."""
    result = conn.execute(
        "SELECT MAX(snapshot_date) FROM portfolio_snapshots"
    ).fetchone()
    if not result or result[0] is None:
        return pd.DataFrame()
    latest_date = result[0]
    return conn.execute("""
        SELECT * FROM portfolio_snapshots WHERE snapshot_date = ?
    """, [latest_date]).df()


def get_position_entry_date(conn: duckdb.DuckDBPyConnection, ticker: str) -> Optional[date]:
    """Return the first date a ticker appeared in portfolio_snapshots."""
    result = conn.execute("""
        SELECT MIN(snapshot_date) FROM portfolio_snapshots WHERE ticker = ?
    """, [ticker]).fetchone()
    return result[0] if result else None


# ---------------------------------------------------------------------------
# Staleness check
# ---------------------------------------------------------------------------

def is_stale(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    ticker: Optional[str],
    staleness_days: int,
) -> bool:
    """
    Return True if the most recent fetched_at for ticker in table
    is older than staleness_days (or if no data exists).
    """
    try:
        if ticker:
            result = conn.execute(f"""
                SELECT MAX(fetched_at) FROM {table} WHERE ticker = ?
            """, [ticker]).fetchone()
        else:
            result = conn.execute(f"""
                SELECT MAX(fetched_at) FROM {table}
            """).fetchone()

        if not result or result[0] is None:
            return True

        last_fetch = result[0]
        if isinstance(last_fetch, str):
            last_fetch = datetime.fromisoformat(last_fetch)
        cutoff = datetime.now() - timedelta(days=staleness_days)
        return last_fetch < cutoff
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_fetch(
    conn: duckdb.DuckDBPyConnection,
    module: str,
    ticker: Optional[str],
    status: str,
    message: str = "",
    rows_written: int = 0,
) -> None:
    conn.execute("""
        INSERT INTO fetch_log (id, module, ticker, status, message, rows_written)
        VALUES (nextval('fetch_log_seq'), ?, ?, ?, ?, ?)
    """, [module, ticker, status, message, rows_written])
