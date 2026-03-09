"""
FX rate fetcher: daily spot rates for major currency pairs.
Primary source: yfinance.  Fallback: Tiingo FX (free tier).

Pairs stored in yfinance convention:
  EURUSD=X   EUR per USD  (rate > 1.0 means 1 EUR buys 1.xx USD)
  GBPUSD=X   GBP per USD
  USDCHF=X   USD per CHF  (rising = USD stronger vs CHF)
  USDCAD=X   USD per CAD  (rising = USD stronger vs CAD)
  USDJPY=X   USD per JPY  (rising = USD stronger; JPY weakness)

For USD-strength analysis the sign convention is handled in fx_flows.py.
"""
import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# yfinance ticker → (base_currency, quote_currency)
FX_PAIRS: dict[str, tuple[str, str]] = {
    "EURUSD=X": ("EUR", "USD"),
    "GBPUSD=X": ("GBP", "USD"),
    "USDCHF=X": ("USD", "CHF"),
    "USDCAD=X": ("USD", "CAD"),
    "USDJPY=X": ("USD", "JPY"),
}

# Portfolio-relevant mapping: which pair converts which region's stock to EUR
# EU stocks are native EUR/CHF; US stocks need EUR/USD; CA stocks need EUR/CAD
REGION_FX_PAIR: dict[str, Optional[str]] = {
    "EU": None,       # EUR-base — no conversion needed for EUR stocks
    "US": "EURUSD=X", # US stock in USD → divide by EURUSD to get EUR value
    "CA": "USDCAD=X", # CA stock in CAD → (USDCAD)^-1 * EURUSD for EUR value
}

# Tiingo FX pair codes (lowercase, no =X suffix)
_TIINGO_FX_MAP: dict[str, str] = {
    "EURUSD=X": "eurusd",
    "GBPUSD=X": "gbpusd",
    "USDCHF=X": "usdchf",
    "USDCAD=X": "usdcad",
    "USDJPY=X": "usdjpy",
}

TIINGO_FX_BASE = "https://api.tiingo.com/tiingo/fx"


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def fetch_fx_yfinance(
    pairs: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch FX rates via yfinance.  Returns long DataFrame: pair, date, rate, source.
    """
    import yfinance as yf

    try:
        raw = yf.download(
            pairs,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        logger.warning(f"yfinance FX download failed: {e}")
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    # yfinance returns MultiIndex when multiple tickers; single column otherwise
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, 0]
    else:
        close = raw[["Close"]] if "Close" in raw.columns else raw
        if len(pairs) == 1:
            close.columns = pairs

    rows = []
    for pair in pairs:
        if pair not in close.columns:
            continue
        s = close[pair].dropna()
        if s.empty:
            continue
        tmp = s.reset_index()
        tmp.columns = ["date", "rate"]
        tmp["pair"] = pair
        tmp["source"] = "yfinance"
        rows.append(tmp)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[["pair", "date", "rate", "source"]]


def fetch_fx_tiingo(
    pairs: list[str],
    start_date: str,
    end_date: str,
    api_key: str,
) -> pd.DataFrame:
    """
    Fetch FX rates via Tiingo forex endpoint.  Free tier covers all 5 pairs.
    Returns long DataFrame: pair, date, rate, source.
    """
    import requests

    rows = []
    headers = {"Content-Type": "application/json", "Authorization": f"Token {api_key}"}

    for pair in pairs:
        tiingo_code = _TIINGO_FX_MAP.get(pair)
        if not tiingo_code:
            continue
        url = f"{TIINGO_FX_BASE}/{tiingo_code}/prices"
        params = {
            "startDate": start_date,
            "endDate": end_date,
            "resampleFreq": "daily",
            "token": api_key,
        }
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20)
            if resp.status_code == 404:
                logger.debug(f"Tiingo FX: {tiingo_code} not found")
                continue
            resp.raise_for_status()
            data = resp.json()
            if not data:
                continue

            df = pd.DataFrame(data)
            # Tiingo FX returns 'close' or 'midPrice'
            rate_col = "close" if "close" in df.columns else "midPrice"
            if rate_col not in df.columns:
                continue
            df = df.rename(columns={"date": "date", rate_col: "rate"})
            df["pair"] = pair
            df["source"] = "tiingo"
            df["date"] = pd.to_datetime(df["date"]).dt.date
            rows.append(df[["pair", "date", "rate", "source"]])
        except Exception as e:
            logger.warning(f"Tiingo FX fetch failed for {pair}: {e}")

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Main update function
# ---------------------------------------------------------------------------

def update_fx_rates(
    conn,
    params: dict,
    tiingo_api_key: Optional[str] = None,
    force_refresh: bool = False,
) -> None:
    """
    Fetch and store FX rates for all configured pairs.
    Primary: yfinance.  Fallback: Tiingo (if key available).
    """
    from src.data.db import upsert_fx_rates, log_fetch, is_stale

    staleness = params["data"].get("fx_staleness_days", 1)
    pairs = list(FX_PAIRS.keys())

    if not force_refresh and not is_stale(conn, "fx_rates", None, staleness):
        logger.info("FX rates are fresh, skipping")
        return

    history_years = params["data"].get("price_history_years", 5)
    start_date = (date.today() - timedelta(days=365 * history_years)).isoformat()
    end_date   = date.today().isoformat()

    # Primary: yfinance
    df = fetch_fx_yfinance(pairs, start_date, end_date)
    fetched = set(df["pair"].unique()) if not df.empty else set()

    # Fallback: Tiingo for any missing pairs
    missing = [p for p in pairs if p not in fetched]
    if missing and tiingo_api_key:
        tiingo_df = fetch_fx_tiingo(missing, start_date, end_date, tiingo_api_key)
        if not tiingo_df.empty:
            df = pd.concat([df, tiingo_df], ignore_index=True) if not df.empty else tiingo_df

    if df.empty:
        log_fetch(conn, "fx", "all_pairs", "error", "No FX data retrieved")
        return

    rows = upsert_fx_rates(conn, df)
    log_fetch(conn, "fx", "all_pairs", "success", rows_written=rows)
    n_pairs = df["pair"].nunique()
    logger.info(f"FX rates: {rows} rows written ({n_pairs} pairs)")
