"""
Price data fetcher: yfinance (primary) with EODHD fallback for EU stocks.
Writes to DuckDB prices table. Respects staleness window from params.
"""
import os
import time
import logging
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# yfinance fetch
# ---------------------------------------------------------------------------

def fetch_prices_yfinance(
    tickers: list[str],
    start_date: str,
    end_date: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch OHLCV via yfinance for a list of tickers.
    Returns long-format DataFrame with columns:
      ticker, date, open, high, low, close, adj_close, volume, currency, source
    Empty DataFrame on failure.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed")
        return pd.DataFrame()

    rows = []
    # Fetch in batches of 20 to avoid timeouts
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    batch,
                    start=start_date,
                    end=end_date,
                    auto_adjust=False,
                    progress=False,
                    threads=True,
                )
                if data.empty:
                    break

                # Handle single ticker (no multi-level columns)
                if len(batch) == 1:
                    ticker = batch[0]
                    df = data.copy()
                    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                    df["ticker"] = ticker
                    df = df.reset_index().rename(columns={"Date": "date", "index": "date"})
                    df["adj_close"] = df.get("adj close", df.get("close"))
                    df["source"] = "yfinance"
                    df["currency"] = "USD"  # will be overridden per-stock later
                    rows.append(df)
                else:
                    # Multi-level columns: (field, ticker)
                    for ticker in batch:
                        try:
                            sub = data.xs(ticker, level=1, axis=1).copy() if ticker in data.columns.get_level_values(1) else pd.DataFrame()
                            if sub.empty:
                                continue
                            sub.columns = [c.lower().replace(" ", "_") for c in sub.columns]
                            sub = sub.reset_index().rename(columns={"Date": "date"})
                            sub["ticker"] = ticker
                            sub["adj_close"] = sub.get("adj close", sub.get("close"))
                            sub["source"] = "yfinance"
                            sub["currency"] = "USD"
                            rows.append(sub)
                        except Exception:
                            continue
                break
            except Exception as e:
                logger.warning(f"yfinance batch attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    result = result.dropna(subset=["adj_close"])

    # Normalize column names
    rename = {"adj close": "adj_close", "open": "open", "high": "high",
               "low": "low", "close": "close", "volume": "volume"}
    for old, new in rename.items():
        if old in result.columns and new not in result.columns:
            result[new] = result[old]

    keep = ["ticker", "date", "open", "high", "low", "close", "adj_close",
            "volume", "currency", "source"]
    for col in keep:
        if col not in result.columns:
            result[col] = None

    result["date"] = pd.to_datetime(result["date"]).dt.date
    return result[keep].dropna(subset=["adj_close"])


def fetch_prices_eodhd(
    tickers_map: dict[str, str],  # {yfinance_ticker: eodhd_ticker}
    start_date: str,
    end_date: str,
    api_key: str,
) -> pd.DataFrame:
    """
    Fetch from EODHD /eod endpoint.
    tickers_map: {original_ticker: eodhd_format_ticker}
    Returns long-format DataFrame matching prices table schema.
    """
    import requests

    rows = []
    base_url = "https://eodhd.com/api/eod"

    for orig_ticker, eodhd_ticker in tickers_map.items():
        try:
            url = f"{base_url}/{eodhd_ticker}"
            params = {
                "api_token": api_key,
                "fmt": "json",
                "from": start_date,
                "to": end_date,
                "period": "d",
            }
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                continue

            df = pd.DataFrame(data)
            df = df.rename(columns={
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "adjusted_close": "adj_close",
                "volume": "volume",
            })
            df["ticker"] = orig_ticker
            df["source"] = "eodhd"
            df["currency"] = "USD"  # will be overridden
            rows.append(df)
        except Exception as e:
            logger.warning(f"EODHD fetch failed for {eodhd_ticker}: {e}")

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"]).dt.date
    keep = ["ticker", "date", "open", "high", "low", "close", "adj_close",
            "volume", "currency", "source"]
    for col in keep:
        if col not in result.columns:
            result[col] = None
    return result[keep].dropna(subset=["adj_close"])


# ---------------------------------------------------------------------------
# Main update function
# ---------------------------------------------------------------------------

def update_prices(
    conn,
    universe_df: pd.DataFrame,
    params: dict,
    extra_tickers: Optional[list[str]] = None,  # e.g. sector ETFs, market indices
    eodhd_api_key: Optional[str] = None,
    force_refresh: bool = False,
) -> dict[str, str]:
    """
    Update prices for all universe stocks + extra_tickers.

    For each ticker:
      1. Check staleness. Skip if fresh.
      2. Determine fetch window (from last available date or full history).
      3. Try yfinance. For EU stocks without EODHD key, yfinance is fine for major
         exchanges (XETRA tickers like RHM.DE work with yfinance).
      4. If yfinance returns empty AND eodhd_api_key available: try EODHD.
      5. Assign currency from universe_df.
      6. Upsert to DB.

    Returns dict: ticker -> status ('updated'|'cached'|'failed')
    """
    from src.data.db import upsert_prices, is_stale, log_fetch

    staleness_days = params["data"]["price_staleness_days"]
    history_years = params["data"].get("price_history_years", 5)

    # Build full ticker list
    all_tickers = universe_df["ticker"].tolist()
    if extra_tickers:
        all_tickers = list(set(all_tickers + extra_tickers))

    # Currency map from universe
    currency_map = dict(zip(universe_df["ticker"], universe_df["currency"]))
    # Default to USD for ETFs/indices not in universe
    eodhd_map = {}
    if "eodhd_ticker" in universe_df.columns:
        eodhd_map = dict(
            zip(universe_df["ticker"],
                universe_df["eodhd_ticker"].fillna(universe_df["ticker"]))
        )

    results = {}
    tickers_to_fetch = []

    for ticker in all_tickers:
        if not force_refresh and not is_stale(conn, "prices", ticker, staleness_days):
            results[ticker] = "cached"
        else:
            tickers_to_fetch.append(ticker)

    if not tickers_to_fetch:
        logger.info("All prices are fresh, nothing to fetch")
        return results

    # Determine date range
    today = date.today().isoformat()
    start_history = (date.today() - timedelta(days=history_years * 365)).isoformat()

    # Get per-ticker last available date to avoid re-fetching all history
    ticker_starts = {}
    for ticker in tickers_to_fetch:
        from src.data.db import get_latest_price_date
        last_date = get_latest_price_date(conn, ticker)
        if last_date:
            # Fetch from day after last available date
            next_date = (last_date + timedelta(days=1)).isoformat()
            ticker_starts[ticker] = next_date
        else:
            ticker_starts[ticker] = start_history

    # Group by start date for batch efficiency
    start_groups: dict[str, list[str]] = {}
    for ticker, start in ticker_starts.items():
        start_groups.setdefault(start, []).append(ticker)

    for start, group_tickers in start_groups.items():
        logger.info(f"Fetching {len(group_tickers)} tickers from {start}")
        df = fetch_prices_yfinance(group_tickers, start, today)

        if df.empty and eodhd_api_key:
            # Fallback: only EU tickers with eodhd mapping
            eu_tickers = [t for t in group_tickers
                          if universe_df[universe_df["ticker"] == t]["region"].values[0:1] == ["EU"]
                          and t in eodhd_map]
            if eu_tickers:
                eodhd_input = {t: eodhd_map[t] for t in eu_tickers}
                df = fetch_prices_eodhd(eodhd_input, start, today, eodhd_api_key)

        if df.empty:
            for t in group_tickers:
                results[t] = "failed"
                log_fetch(conn, "prices", t, "error", "empty response")
            continue

        # Assign correct currency per ticker
        df["currency"] = df["ticker"].map(currency_map).fillna("USD")

        rows_written = upsert_prices(conn, df)
        for t in group_tickers:
            t_rows = len(df[df["ticker"] == t])
            if t_rows > 0:
                results[t] = "updated"
                log_fetch(conn, "prices", t, "success", rows_written=t_rows)
            else:
                results[t] = "failed"
                log_fetch(conn, "prices", t, "error", "no rows in response")

    return results


# ---------------------------------------------------------------------------
# Return computation helpers
# ---------------------------------------------------------------------------

def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from wide-format price DataFrame."""
    return np.log(price_df / price_df.shift(1)).dropna(how="all")


def compute_simple_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily simple returns."""
    return price_df.pct_change().dropna(how="all")


def compute_rolling_volatility(
    returns_df: pd.DataFrame,
    window: int = 63,
    annualize: bool = True,
) -> pd.DataFrame:
    """Rolling annualized volatility."""
    vol = returns_df.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def compute_annualized_volatility(
    returns_series: pd.Series,
    window: int = 252,
) -> float:
    """
    Annualized vol from trailing window of daily log returns.
    Raises ValueError if fewer than 30 data points available.
    """
    data = returns_series.dropna().tail(window)
    if len(data) < 30:
        raise ValueError(f"Insufficient price data: {len(data)} points")
    return float(data.std() * np.sqrt(252))
