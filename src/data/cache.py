"""
CSV-based data cache committed to the repository.

Architecture:
  data/cache/macro/{series_id}.csv          — one file per FRED/ECB series
  data/cache/fundamentals/annual/{ticker}.csv
  data/cache/fundamentals/quarterly/{ticker}.csv

Purpose: bootstrap local DuckDB without live API keys.
GitHub Actions fetches fresh data weekly, exports to these CSVs, and commits
them back to the repo.  Local users `git pull` → get pre-populated cache →
`sync_cache_to_db()` at startup loads into DuckDB → no API keys needed.

CSV format (macro):
  date,value
  2020-01-01,1.234
  ...
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
CACHE_DIR    = _REPO_ROOT / "data" / "cache"
MACRO_DIR    = CACHE_DIR / "macro"
FUND_ANN_DIR = CACHE_DIR / "fundamentals" / "annual"
FUND_QTR_DIR = CACHE_DIR / "fundamentals" / "quarterly"

# ---------------------------------------------------------------------------
# Macro cache
# ---------------------------------------------------------------------------

def _macro_path(series_id: str) -> Path:
    return MACRO_DIR / f"{series_id}.csv"


def load_macro_cache(series_id: str) -> pd.Series:
    """
    Load a cached macro series from CSV.

    Returns a pd.Series indexed by date (datetime.date objects), values as float.
    Returns an empty Series if the file does not exist.
    """
    p = _macro_path(series_id)
    if not p.exists():
        return pd.Series(dtype=float, name=series_id)
    df = pd.read_csv(p, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    df = df.dropna(subset=["value"]).drop_duplicates("date").sort_values("date")
    series = pd.Series(df["value"].values, index=df["date"].values, name=series_id)
    return series


def write_macro_cache(series_id: str, series: pd.Series) -> None:
    """
    Write/update a macro series CSV.

    Merges new data with any existing file (new data wins on date conflicts).
    Creates parent directories if needed.
    """
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    p = _macro_path(series_id)

    new_df = pd.DataFrame({"date": series.index, "value": series.values})
    new_df["date"] = pd.to_datetime(new_df["date"]).dt.date

    if p.exists():
        old_df = pd.read_csv(p, parse_dates=["date"])
        old_df["date"] = old_df["date"].dt.date
        # New data wins on conflicts
        combined = (
            pd.concat([old_df, new_df], ignore_index=True)
            .drop_duplicates("date", keep="last")
            .sort_values("date")
        )
    else:
        combined = new_df.drop_duplicates("date").sort_values("date")

    combined.to_csv(p, index=False)
    log.debug("cache: wrote %d rows → %s", len(combined), p.name)


def macro_cache_latest_date(series_id: str) -> Optional[date]:
    """Return the most recent date in the cached series, or None if absent."""
    p = _macro_path(series_id)
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"])
    if df.empty:
        return None
    return df["date"].max().date()


# ---------------------------------------------------------------------------
# Fundamentals cache
# ---------------------------------------------------------------------------

def _fund_path(ticker: str, period: str) -> Path:
    """period: 'annual' | 'quarterly'"""
    base = FUND_ANN_DIR if period == "annual" else FUND_QTR_DIR
    safe = ticker.replace("/", "_").replace(".", "_")
    return base / f"{safe}.csv"


def load_fundamentals_cache(ticker: str, period: str = "annual") -> pd.DataFrame:
    """
    Load cached fundamentals for a ticker.

    Returns an empty DataFrame if the file does not exist.
    """
    p = _fund_path(ticker, period)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["report_date"])
    if "report_date" in df.columns:
        df["report_date"] = df["report_date"].dt.date
    return df


def write_fundamentals_cache(
    ticker: str, df: pd.DataFrame, period: str = "annual"
) -> None:
    """
    Write/update fundamentals CSV for a ticker.

    Merges with existing file; new data wins on primary-key conflicts.
    Primary key: (fiscal_year) for annual, (fiscal_year, fiscal_quarter) for quarterly.
    """
    if df.empty:
        return
    base = FUND_ANN_DIR if period == "annual" else FUND_QTR_DIR
    base.mkdir(parents=True, exist_ok=True)
    p = _fund_path(ticker, period)

    pk_cols = ["fiscal_year"] if period == "annual" else ["fiscal_year", "fiscal_quarter"]
    pk_cols = [c for c in pk_cols if c in df.columns]

    if p.exists() and pk_cols:
        old_df = pd.read_csv(p)
        combined = (
            pd.concat([old_df, df], ignore_index=True)
            .drop_duplicates(pk_cols, keep="last")
            .sort_values(pk_cols)
        )
    else:
        combined = df.copy()

    combined.to_csv(p, index=False)
    log.debug("cache: wrote %s %s → %s rows", ticker, period, len(combined))


# ---------------------------------------------------------------------------
# Sync: CSV cache → DuckDB
# ---------------------------------------------------------------------------

def sync_cache_to_db(conn) -> dict:
    """
    Load all cached CSVs into DuckDB.

    Called once at pipeline startup — populates the DB from the committed
    CSV cache so that the pipeline can run without live API keys.

    Only inserts rows that do not already exist (INSERT OR IGNORE semantics):
    if the DB already has fresh data from a real fetch, that data is preserved.

    Returns a summary dict:
      {"macro_series": int, "macro_rows": int,
       "fundamentals_annual": int, "fundamentals_annual_rows": int,
       "fundamentals_quarterly": int, "fundamentals_quarterly_rows": int}
    """
    summary = {
        "macro_series": 0, "macro_rows": 0,
        "fundamentals_annual": 0, "fundamentals_annual_rows": 0,
        "fundamentals_quarterly": 0, "fundamentals_quarterly_rows": 0,
    }

    # --- macro ---
    if MACRO_DIR.exists():
        for csv_path in sorted(MACRO_DIR.glob("*.csv")):
            series_id = csv_path.stem
            df = pd.read_csv(csv_path, parse_dates=["date"])
            if df.empty:
                continue
            df["date"] = df["date"].dt.date.astype(str)
            df["series_id"] = series_id
            df["source"] = "cache"

            rows_before = conn.execute(
                "SELECT COUNT(*) FROM macro_series WHERE series_id = ?", [series_id]
            ).fetchone()[0]

            conn.executemany(
                """
                INSERT OR IGNORE INTO macro_series (series_id, date, value, source)
                VALUES (?, ?, ?, ?)
                """,
                df[["series_id", "date", "value", "source"]].values.tolist(),
            )

            rows_after = conn.execute(
                "SELECT COUNT(*) FROM macro_series WHERE series_id = ?", [series_id]
            ).fetchone()[0]

            inserted = rows_after - rows_before
            if inserted > 0:
                log.info("cache→db: %s  +%d rows", series_id, inserted)
                summary["macro_series"] += 1
                summary["macro_rows"] += inserted

    # --- fundamentals annual ---
    if FUND_ANN_DIR.exists():
        for csv_path in sorted(FUND_ANN_DIR.glob("*.csv")):
            ticker = csv_path.stem.replace("_", ".")  # reverse safe-name
            df = pd.read_csv(csv_path)
            if df.empty or "fiscal_year" not in df.columns:
                continue

            # Fill required NOT NULL columns
            for col, default in [
                ("currency", "USD"), ("reporting_std", "GAAP"), ("source", "cache")
            ]:
                if col not in df.columns:
                    df[col] = default
                else:
                    df[col] = df[col].fillna(default)

            # Build insert list for columns that exist in the table
            cols = [
                "ticker", "fiscal_year", "report_date", "revenue", "gross_profit",
                "ebit", "total_equity", "total_debt", "cash", "capex",
                "gross_margin", "roic", "invested_capital", "tax_rate",
                "currency", "reporting_std", "source",
            ]
            if "ticker" not in df.columns:
                df["ticker"] = ticker
            present = [c for c in cols if c in df.columns]
            placeholders = ", ".join(["?"] * len(present))
            col_list = ", ".join(present)

            rows_before = conn.execute(
                "SELECT COUNT(*) FROM fundamentals_annual WHERE ticker = ?", [ticker]
            ).fetchone()[0]

            try:
                conn.executemany(
                    f"INSERT OR IGNORE INTO fundamentals_annual ({col_list}) VALUES ({placeholders})",
                    df[present].where(pd.notnull(df[present]), None).values.tolist(),
                )
            except Exception as exc:
                log.warning("cache→db annual %s: %s", ticker, exc)
                continue

            rows_after = conn.execute(
                "SELECT COUNT(*) FROM fundamentals_annual WHERE ticker = ?", [ticker]
            ).fetchone()[0]

            inserted = rows_after - rows_before
            if inserted > 0:
                summary["fundamentals_annual"] += 1
                summary["fundamentals_annual_rows"] += inserted

    # --- fundamentals quarterly ---
    if FUND_QTR_DIR.exists():
        for csv_path in sorted(FUND_QTR_DIR.glob("*.csv")):
            ticker = csv_path.stem.replace("_", ".")
            df = pd.read_csv(csv_path)
            if df.empty or "fiscal_year" not in df.columns:
                continue

            for col, default in [
                ("currency", "USD"), ("reporting_std", "GAAP"), ("source", "cache")
            ]:
                if col not in df.columns:
                    df[col] = default
                else:
                    df[col] = df[col].fillna(default)

            cols = [
                "ticker", "fiscal_year", "fiscal_quarter", "report_date",
                "revenue", "gross_profit", "gross_margin",
                "currency", "reporting_std", "source",
            ]
            if "ticker" not in df.columns:
                df["ticker"] = ticker
            present = [c for c in cols if c in df.columns]
            placeholders = ", ".join(["?"] * len(present))
            col_list = ", ".join(present)

            rows_before = conn.execute(
                "SELECT COUNT(*) FROM fundamentals_quarterly WHERE ticker = ?", [ticker]
            ).fetchone()[0]

            try:
                conn.executemany(
                    f"INSERT OR IGNORE INTO fundamentals_quarterly ({col_list}) VALUES ({placeholders})",
                    df[present].where(pd.notnull(df[present]), None).values.tolist(),
                )
            except Exception as exc:
                log.warning("cache→db quarterly %s: %s", ticker, exc)
                continue

            rows_after = conn.execute(
                "SELECT COUNT(*) FROM fundamentals_quarterly WHERE ticker = ?", [ticker]
            ).fetchone()[0]

            inserted = rows_after - rows_before
            if inserted > 0:
                summary["fundamentals_quarterly"] += 1
                summary["fundamentals_quarterly_rows"] += inserted

    log.info(
        "sync_cache_to_db complete: macro=%d series (+%d rows), "
        "fund_annual=%d tickers (+%d rows), fund_qtr=%d tickers (+%d rows)",
        summary["macro_series"], summary["macro_rows"],
        summary["fundamentals_annual"], summary["fundamentals_annual_rows"],
        summary["fundamentals_quarterly"], summary["fundamentals_quarterly_rows"],
    )
    return summary
