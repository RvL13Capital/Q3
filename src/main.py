"""
EARKE Quant 3.0 — Weekly Pipeline Orchestrator

Usage:
  python src/main.py                     # run with today's date
  python src/main.py --date 2025-03-01   # run for a specific date
  python src/main.py --force-refresh     # bypass staleness checks
  python src/main.py --skip-fetch        # use only cached data (signals only)
  python src/main.py --dry-run           # compute but don't write portfolio snapshot
  python src/main.py --trends            # also fetch Google Trends (slow, 5s/batch)

Dashboard:
  streamlit run src/reporting/dashboard.py
"""
import argparse
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("q3.main")


def load_params(config_path: str = "config/params.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def most_recent_trading_day() -> str:
    """Return today's date (pipeline checks for data availability at runtime)."""
    return date.today().isoformat()


def run_weekly_pipeline(
    as_of_date: str | None = None,
    force_refresh: bool = False,
    skip_fetch: bool = False,
    dry_run: bool = False,
    fetch_trends: bool = False,
) -> None:
    """
    Full weekly pipeline. Execution order:
      1. Setup: load params, open DB, load universe
      2. Data refresh (price, fundamentals, macro, trends)
      3. Scoring (physical, quality, crowding, composite)
      4. Portfolio construction + event detection
      5. Report generation
    """
    import time
    start_time = time.time()

    # ── 1. Setup ────────────────────────────────────────────────────────────
    params = load_params()
    if not as_of_date:
        as_of_date = most_recent_trading_day()

    logger.info(f"=== EARKE Quant 3.0 | as_of_date={as_of_date} ===")

    from src.data.db       import get_connection
    from src.data.universe import load_universe, all_trend_keywords, all_sector_etfs

    conn     = get_connection()
    universe = load_universe()
    logger.info(f"Universe loaded: {len(universe)} stocks")

    # API keys from environment
    fred_api_key  = os.getenv("FRED_API_KEY")
    eodhd_api_key = os.getenv("EODHD_API_KEY")
    eodhd_api_key_2 = os.getenv("EODHD_API_KEY_2")
    eodhd_keys = [k for k in [eodhd_api_key, eodhd_api_key_2] if k]

    if not fred_api_key:
        logger.warning("FRED_API_KEY not set — US/CA macro data unavailable")
    if not eodhd_keys:
        logger.warning("EODHD_API_KEY not set — using yfinance for all data")
    elif len(eodhd_keys) > 1:
        logger.info(f"EODHD: {len(eodhd_keys)} API keys loaded (key rotation enabled)")

    # ── 2. Data Refresh ──────────────────────────────────────────────────────
    if not skip_fetch:
        logger.info("── Fetching prices ──")
        from src.data.prices import update_prices

        extra_tickers = all_sector_etfs(universe, params)
        price_results = update_prices(
            conn, universe, params,
            extra_tickers=extra_tickers,
            eodhd_api_key=eodhd_keys[0] if eodhd_keys else None,
            eodhd_api_keys=eodhd_keys,
            force_refresh=force_refresh,
        )
        n_updated = sum(1 for v in price_results.values() if v == "updated")
        n_cached  = sum(1 for v in price_results.values() if v == "cached")
        n_failed  = sum(1 for v in price_results.values() if v == "failed")
        logger.info(f"Prices: {n_updated} updated, {n_cached} cached, {n_failed} failed")

        logger.info("── Fetching fundamentals ──")
        from src.data.fundamentals import update_fundamentals

        fund_results = update_fundamentals(
            conn, universe, params,
            eodhd_api_key=eodhd_keys[0] if eodhd_keys else None,
            eodhd_api_keys=eodhd_keys,
            force_refresh=force_refresh,
        )
        n_fund_updated = sum(1 for v in fund_results.values() if v == "updated")
        logger.info(f"Fundamentals: {n_fund_updated} updated")

        logger.info("── Fetching macro data ──")
        from src.data.macro import update_macro

        update_macro(conn, params, fred_api_key=fred_api_key, force_refresh=force_refresh)

        if fetch_trends:
            logger.info("── Fetching Google Trends ──")
            from src.signals.crowding import fetch_google_trends_batch

            keywords = all_trend_keywords(universe)
            fetch_google_trends_batch(keywords, conn)
            logger.info(f"Trends: fetched {len(keywords)} keywords")
    else:
        logger.info("Skipping data fetch (--skip-fetch mode)")

    # ── 3. Scoring ───────────────────────────────────────────────────────────
    logger.info("── Running signal scoring ──")
    from src.signals.composite import run_weekly_scoring

    scored_df = run_weekly_scoring(conn, universe, params, as_of_date)
    n_entry = scored_df["entry_signal"].sum() if not scored_df.empty else 0
    n_exit  = scored_df["exit_signal"].sum() if not scored_df.empty else 0
    logger.info(f"Scoring complete: {len(scored_df)} stocks, {n_entry} entries, {n_exit} exits")

    # ── 4. Portfolio + Events ────────────────────────────────────────────────
    logger.info("── Running portfolio construction & event detection ──")
    from src.portfolio.monitor import generate_weekly_actions

    actions = generate_weekly_actions(conn, scored_df, universe, params, as_of_date)

    # Also run sigma estimation and update signal_scores with kelly weights
    from src.portfolio.kelly   import estimate_sigma
    from src.data.db           import upsert_signal_scores

    if not scored_df.empty:
        # Estimate sigma for entry candidates and write back into scored_df
        sigma_map: dict[str, float] = {}
        for _, row in scored_df[scored_df["entry_signal"] == True].iterrows():
            sigma, _ = estimate_sigma(row["ticker"], conn, as_of_date)
            sigma_map[row["ticker"]] = round(sigma, 4)
        if sigma_map:
            scored_df["sigma_estimate"] = scored_df["ticker"].map(sigma_map)

    # ── 5. Reporting ─────────────────────────────────────────────────────────
    logger.info("── Generating weekly report ──")
    from src.reporting.weekly_report import generate_weekly_report

    report_path = generate_weekly_report(
        conn, actions, scored_df, params, as_of_date,
        output_dir=params["reporting"]["output_dir"],
    )
    logger.info(f"Report saved: {report_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info(f"=== Pipeline complete in {elapsed:.1f}s ===")

    new_portfolio = actions.get("new_portfolio")
    if new_portfolio is not None and not new_portfolio.empty:
        invested = new_portfolio["weight"].sum()
        logger.info(
            f"Portfolio: {len(new_portfolio)} positions, "
            f"invested={invested:.1%}, cash={1-invested:.1%}"
        )

    if actions.get("any_action"):
        print("\n⚡ ACTION REQUIRED — check the report for details")
        exits = actions.get("exits")
        if exits is not None and not exits.empty:
            triggered = exits[exits["exit_triggered"]]
            if not triggered.empty:
                print(f"  Exits: {', '.join(triggered['ticker'].tolist())}")
        entries = actions.get("entries")
        if entries is not None and not entries.empty:
            print(f"  Entries: {', '.join(entries['ticker'].head(5).tolist())}")
    else:
        print("\n✅ No action required this week.")

    print(f"\nReport: {report_path}")
    print("Dashboard: streamlit run src/reporting/dashboard.py")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="EARKE Quant 3.0 Weekly Pipeline")
    parser.add_argument("--date",          type=str,  help="As-of date (YYYY-MM-DD)")
    parser.add_argument("--force-refresh", action="store_true", help="Bypass staleness checks")
    parser.add_argument("--skip-fetch",    action="store_true", help="Use only cached data")
    parser.add_argument("--dry-run",       action="store_true", help="Don't write portfolio snapshot")
    parser.add_argument("--trends",        action="store_true", help="Also fetch Google Trends")
    args = parser.parse_args()

    run_weekly_pipeline(
        as_of_date=args.date,
        force_refresh=args.force_refresh,
        skip_fetch=args.skip_fetch,
        dry_run=args.dry_run,
        fetch_trends=args.trends,
    )


if __name__ == "__main__":
    main()
