"""
Refresh the CSV data cache.

Uses an in-memory DuckDB so the main data/q3.duckdb is never touched.
After fetching, writes CSVs to data/cache/ which are committed to the repo.

Usage:
  python scripts/refresh_cache.py                 # macro only
  python scripts/refresh_cache.py --fundamentals  # also export fundamentals
  python scripts/refresh_cache.py --force         # bypass staleness checks

Environment variables (required for live fetch):
  FRED_API_KEY     — https://fred.stlouisfed.org/docs/api/
  EODHD_API_KEY    — https://eodhd.com  (~€30/mo, optional)
  DB_KEY           — DBnomics API key (optional, for higher EU data rate limits)

This script is designed to run in GitHub Actions with secrets injected as env vars.
Output CSVs are committed back to the repo via the workflow's git push step.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("refresh_cache")


def _load_params() -> dict:
    p = Path(__file__).resolve().parent.parent / "config" / "params.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def _init_db(conn) -> None:
    """Bootstrap schema in the in-memory DB."""
    from src.data.db import init_schema
    init_schema(conn)


def _export_macro(conn) -> int:
    """Export all macro_series rows from in-memory DB to CSV cache."""
    from src.data.cache import write_macro_cache

    try:
        df = conn.execute(
            "SELECT series_id, date, value FROM macro_series ORDER BY series_id, date"
        ).df()
    except Exception as exc:
        log.error("Failed to read macro_series: %s", exc)
        return 0

    if df.empty:
        log.warning("macro_series is empty — nothing to export")
        return 0

    total = 0
    for series_id, group in df.groupby("series_id"):
        series = pd.Series(
            group["value"].values,
            index=pd.to_datetime(group["date"]).dt.date,
            name=series_id,
        )
        write_macro_cache(series_id, series)
        log.info("exported %s: %d rows", series_id, len(series))
        total += len(series)
    return total


def _export_fundamentals(conn) -> tuple[int, int]:
    """Export fundamentals_annual and fundamentals_quarterly to CSV cache."""
    from src.data.cache import write_fundamentals_cache

    annual_rows = 0
    quarterly_rows = 0

    # Annual
    try:
        df = conn.execute("SELECT * FROM fundamentals_annual ORDER BY ticker, fiscal_year").df()
        for ticker, group in df.groupby("ticker"):
            write_fundamentals_cache(ticker, group.copy(), period="annual")
            annual_rows += len(group)
    except Exception as exc:
        log.error("fundamentals_annual export: %s", exc)

    # Quarterly
    try:
        df = conn.execute(
            "SELECT * FROM fundamentals_quarterly ORDER BY ticker, fiscal_year, fiscal_quarter"
        ).df()
        for ticker, group in df.groupby("ticker"):
            write_fundamentals_cache(ticker, group.copy(), period="quarterly")
            quarterly_rows += len(group)
    except Exception as exc:
        log.error("fundamentals_quarterly export: %s", exc)

    return annual_rows, quarterly_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh CSV data cache from live APIs")
    parser.add_argument(
        "--fundamentals", action="store_true",
        help="Also fetch and export fundamentals (requires EODHD_API_KEY)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bypass staleness checks; re-fetch everything"
    )
    args = parser.parse_args()

    # ── API keys ──────────────────────────────────────────────────────────────
    fred_api_key     = (os.getenv("FRED_API_KEY")    or "").strip() or None
    eodhd_api_key    = (os.getenv("EODHD_API_KEY")   or "").strip() or None
    eodhd_api_key_2  = (os.getenv("EODHD_API_KEY_2") or "").strip() or None
    tiingo_api_key   = (os.getenv("TIINGO_API")      or "").strip() or None
    dbnomics_api_key = (os.getenv("DB_KEY")          or "").strip() or None
    eodhd_keys = [k for k in [eodhd_api_key, eodhd_api_key_2] if k]

    if not fred_api_key:
        log.warning("FRED_API_KEY not set — US/CA macro will not be fetched")
    if args.fundamentals and not eodhd_keys:
        log.warning("EODHD_API_KEY not set — fundamentals will rely on yfinance only")

    params = _load_params()

    # ── In-memory DuckDB — never touches q3.duckdb ───────────────────────────
    conn = duckdb.connect(":memory:")
    _init_db(conn)

    # ── Fetch macro ───────────────────────────────────────────────────────────
    log.info("=== Fetching macro series ===")
    from src.data.macro import update_macro
    update_macro(
        conn, params,
        fred_api_key=fred_api_key,
        dbnomics_api_key=dbnomics_api_key,
        force_refresh=args.force,
    )

    macro_rows = _export_macro(conn)
    log.info("Macro export complete: %d total rows written to CSVs", macro_rows)

    # ── Fetch fundamentals (optional) ─────────────────────────────────────────
    if args.fundamentals:
        log.info("=== Fetching fundamentals ===")
        from src.data.universe import load_universe
        from src.data.fundamentals import update_fundamentals

        universe = load_universe()
        update_fundamentals(
            conn, universe, params,
            eodhd_api_key=eodhd_keys[0] if eodhd_keys else None,
            eodhd_api_keys=eodhd_keys,
            force_refresh=args.force,
        )
        ann_rows, qtr_rows = _export_fundamentals(conn)
        log.info(
            "Fundamentals export: %d annual rows, %d quarterly rows", ann_rows, qtr_rows
        )

    conn.close()
    log.info("=== refresh_cache.py done ===")


if __name__ == "__main__":
    main()
