"""
scripts/seed_demo.py — Seed DuckDB with static demo data for offline development.

Populates prices (3 years daily), fundamentals (5 years annual), and macro series
so that `python src/main.py --skip-fetch` produces real signals and a PDF report
without needing any API keys or network access.

Usage:
    python scripts/seed_demo.py          # seed then run pipeline
    python scripts/seed_demo.py --seed-only   # seed only, no pipeline run
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────────────────────────────────────
# Demo universe — 20 stocks, all six buckets, three regions
# ─────────────────────────────────────────────────────────────────────────────
DEMO_STOCKS = [
    # ticker         region  currency  bucket              roic    gm    sigma  price0  wacc_est
    ("ENR.DE",       "EU",   "EUR",    "grid",              0.14,  0.22,  0.28,   25.0,  0.075),
    ("ABB.ST",       "EU",   "EUR",    "grid",              0.18,  0.30,  0.22,   40.0,  0.075),
    ("VRT",          "US",   "USD",    "grid",              0.21,  0.35,  0.30,   80.0,  0.080),
    ("NEE",          "US",   "USD",    "grid",              0.10,  0.46,  0.18,  190.0,  0.065),
    ("CCJ",          "CA",   "USD",    "nuclear",           0.09,  0.40,  0.32,   55.0,  0.075),
    ("CEG",          "US",   "USD",    "nuclear",           0.16,  0.28,  0.25,  260.0,  0.070),
    ("RHM.DE",       "EU",   "EUR",    "defense",           0.22,  0.18,  0.26,  550.0,  0.075),
    ("LMT",          "US",   "USD",    "defense",           0.30,  0.13,  0.16,  480.0,  0.075),
    ("BAESY",        "EU",   "EUR",    "defense",           0.20,  0.15,  0.20,   35.0,  0.070),
    ("AWK",          "US",   "USD",    "water",             0.09,  0.50,  0.17,  130.0,  0.065),
    ("XYL",          "US",   "USD",    "water",             0.13,  0.38,  0.22,  115.0,  0.075),
    ("GEVSGY",       "EU",   "EUR",    "water",             0.11,  0.34,  0.20,   25.0,  0.070),
    ("MP",           "US",   "USD",    "critical_materials", 0.06, 0.25,  0.38,   18.0,  0.085),
    ("GLEN.L",       "EU",   "USD",    "critical_materials", 0.12, 0.16,  0.30,    5.5,  0.085),
    ("FCX",          "US",   "USD",    "critical_materials", 0.14,  0.32,  0.35,   42.0,  0.085),
    ("NVDA",         "US",   "USD",    "ai_infra",           0.55,  0.75,  0.40,  130.0,  0.090),
    ("MSFT",         "US",   "USD",    "ai_infra",           0.28,  0.70,  0.22,  420.0,  0.080),
    ("ANET",         "US",   "USD",    "ai_infra",           0.32,  0.64,  0.28,  325.0,  0.085),
    ("ASML",         "EU",   "EUR",    "ai_infra",           0.35,  0.52,  0.30,  760.0,  0.080),
    ("KLAC",         "US",   "USD",    "ai_infra",           0.38,  0.60,  0.26,  800.0,  0.085),
]

# ─────────────────────────────────────────────────────────────────────────────
# Macro seed values
# ─────────────────────────────────────────────────────────────────────────────
MACRO_SEEDS = {
    # Risk-free rates: rose 2023, plateaued 2024, easing 2026
    "US_10Y":       {"start": 3.8,  "end": 4.25, "unit": "percent", "source": "demo"},
    "EU_10Y_DE":    {"start": 2.1,  "end": 2.78, "unit": "percent", "source": "demo"},
    "CA_10Y":       {"start": 3.3,  "end": 3.85, "unit": "percent", "source": "demo"},
    # Inflation: surged 2022–23, normalised by 2026
    "EU_HICP_YOY":  {"start": 6.2,  "end": 2.40, "unit": "percent", "source": "demo"},
    "EU_PPI_YOY":   {"start": 8.0,  "end": 2.10, "unit": "percent", "source": "demo"},
    "US_CPI_YOY":   {"start": 6.5,  "end": 2.80, "unit": "percent", "source": "demo"},
    "CA_CPI_YOY":   {"start": 6.8,  "end": 2.60, "unit": "percent", "source": "demo"},
    "EU_PPI_MFG":   {"start": 7.5,  "end": 1.80, "unit": "percent", "source": "demo"},
    # PPIENG: peaked at energy crisis highs, declining to normalised levels.
    # Recent values near the bottom of the 5-year window → low ECS percentile
    # → X_E logistic damper ≈ 0.95 (EROEI-favourable regime).
    "US_PPIENG":    {"start": 235,  "end": 175,  "unit": "index",   "source": "demo"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _trading_days(start: date, end: date) -> list[date]:
    out = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri
            out.append(d)
        d += timedelta(days=1)
    return out


def _monthly_dates(start: date, end: date) -> list[date]:
    out = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        last = (date(y, m + 1, 1) - timedelta(days=1)) if m < 12 else date(y, 12, 31)
        out.append(last)
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return [d for d in out if d <= end]


def _gen_prices(ticker: str, price0: float, sigma: float,
                dates: list[date], rng: np.random.Generator,
                currency: str) -> pd.DataFrame:
    """Geometric Brownian Motion log-returns, small mean drift."""
    mu_daily = 0.07 / 252  # 7% annual drift
    n = len(dates)
    log_rets = mu_daily + sigma / np.sqrt(252) * rng.standard_normal(n)
    prices = price0 * np.exp(np.cumsum(log_rets))
    # Ensure positive prices
    prices = np.maximum(prices, price0 * 0.1)
    vol = prices * sigma / np.sqrt(252) * rng.uniform(1, 3, n)  # ~ daily volume
    return pd.DataFrame({
        "ticker":    ticker,
        "date":      dates,
        "open":      prices * (1 + rng.uniform(-0.005, 0.005, n)),
        "high":      prices * (1 + np.abs(rng.uniform(0, 0.015, n))),
        "low":       prices * (1 - np.abs(rng.uniform(0, 0.015, n))),
        "close":     prices,
        "adj_close": prices,
        "volume":    (vol * 1_000_000 / prices).astype(int),
        "currency":  currency,
        "source":    "demo",
    })


def _gen_fundamentals(ticker: str, roic: float, gm: float,
                      currency: str, as_of_year: int) -> pd.DataFrame:
    """5 years of synthetic annual fundamentals with realistic progression."""
    rng = np.random.default_rng(abs(hash(ticker)) % (2**31))
    rows = []
    revenue_base = rng.uniform(2e9, 80e9)  # €2B–€80B revenue
    for i, yr in enumerate(range(as_of_year - 4, as_of_year + 1)):
        growth = 1 + rng.uniform(0.03, 0.12)
        rev = revenue_base * (growth ** i)
        gp  = rev * gm * (1 + rng.uniform(-0.02, 0.02))
        ebit = rev * gm * 0.6 * (1 + rng.uniform(-0.05, 0.05))
        net  = ebit * 0.75
        ta   = rev * rng.uniform(0.8, 2.0)
        te   = ta  * rng.uniform(0.35, 0.60)
        debt = ta  * rng.uniform(0.15, 0.35)
        ic   = te  + debt
        nopat = ic * roic * (1 + rng.uniform(-0.02, 0.02))
        rows.append({
            "ticker":             ticker,
            "fiscal_year":        yr,
            "report_date":        date(yr, 12, 31).isoformat(),
            "revenue":            round(rev),
            "gross_profit":       round(gp),
            "gross_margin":       round(gm + rng.uniform(-0.02, 0.02), 4),
            "ebit":               round(ebit),
            "net_income":         round(net),
            "total_assets":       round(ta),
            "total_equity":       round(te),
            "total_debt":         round(debt),
            "cash":               round(ta * rng.uniform(0.05, 0.15)),
            "capex":              round(rev * rng.uniform(0.03, 0.08)),
            "invested_capital":   round(ic),
            "nopat":              round(nopat),
            "roic":               round(roic + rng.uniform(-0.02, 0.02), 4),
            "effective_tax_rate": round(rng.uniform(0.18, 0.25), 4),
            "currency":           currency,
            "accounting_std":     "GAAP" if currency == "USD" else "IFRS",
            "source":             "demo",
        })
    return pd.DataFrame(rows)


def _gen_macro_series(series_id: str, seed: dict,
                      dates: list[date]) -> pd.DataFrame:
    """Smooth linear interpolation with small noise."""
    n = len(dates)
    base = np.linspace(seed["start"], seed["end"], n)
    rng  = np.random.default_rng(abs(hash(series_id)) % (2**31))
    noise = rng.normal(0, abs(seed["end"] - seed["start"]) * 0.02, n)
    values = base + noise
    return pd.DataFrame({
        "series_id": series_id,
        "date":      [d.isoformat() for d in dates],
        "value":     np.round(values, 4),
        "unit":      seed["unit"],
        "source":    seed["source"],
    })


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def seed(as_of: date | None = None, verbose: bool = True) -> None:
    from src.data.db import get_connection, initialize_schema, upsert_prices, \
        upsert_fundamentals_annual, upsert_macro

    conn = get_connection()
    initialize_schema(conn)

    if as_of is None:
        as_of = date.today()

    start_prices = as_of - timedelta(days=3 * 365 + 30)  # ~3 years history
    price_dates  = _trading_days(start_prices, as_of)
    macro_dates  = _monthly_dates(start_prices, as_of)

    rng = np.random.default_rng(42)  # reproducible

    # ── Prices ──────────────────────────────────────────────────────────────
    price_frames = []
    for (ticker, region, currency, bucket, roic, gm, sigma, price0, _) in DEMO_STOCKS:
        price_frames.append(
            _gen_prices(ticker, price0, sigma, price_dates, rng, currency)
        )
    prices_df = pd.concat(price_frames, ignore_index=True)
    n_prices = upsert_prices(conn, prices_df)
    if verbose:
        print(f"  Prices:       {n_prices:>6,} rows  ({len(DEMO_STOCKS)} tickers × {len(price_dates)} days)")

    # Seed all remaining universe tickers with price stubs (needed for
    # absorption ratio confidence — it requires ≥50% of universe tickers
    # to have ≥60 days of price history).
    from src.data.universe import load_universe
    universe_all = load_universe()
    seeded_tickers = {r[0] for r in DEMO_STOCKS}
    stub_frames = []
    for _, row in universe_all.iterrows():
        t = row["ticker"]
        if t in seeded_tickers:
            continue
        # Use a realistic price stub (200-day history, modest drift, moderate vol)
        price0 = rng.uniform(10.0, 500.0)
        stub_frames.append(_gen_prices(t, price0, 0.25, price_dates[-220:], rng,
                                       "EUR" if row["region"] == "EU" else "USD"))
    if stub_frames:
        stub_df = pd.concat(stub_frames, ignore_index=True)
        n_stub = upsert_prices(conn, stub_df)
        if verbose:
            print(f"  Universe stubs:{n_stub:>6,} rows  ({len(stub_frames)} tickers × 220 days)")

    # Also seed sector ETF prices (needed by crowding ETF correlation)
    etf_seeds = [
        ("SPY",     220.0, 0.16, "USD"),
        ("EWC",      35.0, 0.17, "USD"),   # iShares MSCI Canada ETF (TSX proxy)
        ("EXW1.DE",  80.0, 0.18, "EUR"),
        ("GRID",     25.0, 0.22, "USD"),
        ("URA",      25.0, 0.30, "USD"),
        ("IQQH.DE",  12.0, 0.22, "EUR"),
        ("NATO.L",   20.0, 0.25, "GBP"),
        ("ITA",     120.0, 0.20, "USD"),
        ("SOXX",    200.0, 0.30, "USD"),
        ("REMX",     20.0, 0.35, "USD"),
    ]
    etf_frames = [_gen_prices(t, p0, s, price_dates, rng, c) for t, p0, s, c in etf_seeds]
    n_etf = upsert_prices(conn, pd.concat(etf_frames, ignore_index=True))
    if verbose:
        print(f"  ETF prices:   {n_etf:>6,} rows  ({len(etf_seeds)} ETFs)")

    # ── Fundamentals ─────────────────────────────────────────────────────────
    fund_frames = []
    for (ticker, region, currency, bucket, roic, gm, sigma, price0, _) in DEMO_STOCKS:
        fund_frames.append(_gen_fundamentals(ticker, roic, gm, currency, as_of.year))
    funds_df = pd.concat(fund_frames, ignore_index=True)
    n_funds = upsert_fundamentals_annual(conn, funds_df)
    if verbose:
        print(f"  Fundamentals: {n_funds:>6,} rows  ({len(DEMO_STOCKS)} tickers × 5 years)")

    # ── Macro ─────────────────────────────────────────────────────────────────
    n_macro = 0
    for series_id, seed_vals in MACRO_SEEDS.items():
        df = _gen_macro_series(series_id, seed_vals, macro_dates)
        n_macro += upsert_macro(conn, series_id, df)
    if verbose:
        print(f"  Macro series: {n_macro:>6,} rows  ({len(MACRO_SEEDS)} series × {len(macro_dates)} months)")

    conn.close()
    if verbose:
        print(f"\nDB seeded: {n_prices + n_etf + n_funds + n_macro:,} total rows → {as_of}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed DuckDB with demo data for offline dev")
    parser.add_argument("--seed-only",  action="store_true", help="Seed only, skip pipeline run")
    parser.add_argument("--date",       type=str,  default=None, help="As-of date YYYY-MM-DD")
    args = parser.parse_args()

    as_of = date.fromisoformat(args.date) if args.date else date.today()

    print(f"Seeding demo data (as_of={as_of})…")
    seed(as_of=as_of)

    if not args.seed_only:
        import subprocess
        print("\nRunning pipeline (--skip-fetch --dry-run)…\n")
        subprocess.run(
            [sys.executable, "src/main.py", "--skip-fetch", "--dry-run",
             "--date", as_of.isoformat()],
            check=True,
        )
