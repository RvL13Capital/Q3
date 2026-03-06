"""
Universe management: load from universe.yaml, validate, sync to DB.
"""
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "universe.yaml"


def load_universe(config_path: str | Path = CONFIG_PATH) -> pd.DataFrame:
    """
    Load stock universe from YAML. Returns DataFrame with one row per stock.
    The 'buckets' column contains a list of bucket strings.
    """
    with open(config_path) as f:
        data = yaml.safe_load(f)

    rows = []
    seen_tickers = set()
    for stock in data.get("stocks", []):
        ticker = stock["ticker"]
        if ticker in seen_tickers:
            continue  # skip duplicate tickers (e.g. CEG appears twice in universe)
        seen_tickers.add(ticker)
        rows.append({
            "ticker":          ticker,
            "eodhd_ticker":    stock.get("eodhd_ticker"),
            "name":            stock["name"],
            "exchange":        stock["exchange"],
            "region":          stock["region"],
            "currency":        stock["currency"],
            "accounting_std":  stock["accounting_std"],
            "buckets":         stock.get("buckets", []),
            "primary_bucket":  stock.get("primary_bucket"),
            "isin":            stock.get("isin"),
            "trends_keyword":  stock.get("trends_keyword"),
        })

    return pd.DataFrame(rows)


def get_sector_etf(
    primary_bucket: str,
    region: str,
    params: dict,
) -> Optional[str]:
    """Look up the sector ETF for a given bucket + region combination."""
    key = f"{region}|{primary_bucket}"
    return params.get("sector_etfs", {}).get(key)


def get_market_index(region: str, params: dict) -> str:
    """Return the broad market index ticker for a region."""
    return params.get("market_indices", {}).get(region, "SPY")


def get_bucket_tickers(universe_df: pd.DataFrame, bucket: str) -> list[str]:
    """Return all tickers that belong to the given bucket."""
    mask = universe_df["buckets"].apply(lambda b: bucket in b)
    return universe_df.loc[mask, "ticker"].tolist()


def get_universe_by_region(universe_df: pd.DataFrame, region: str) -> pd.DataFrame:
    return universe_df[universe_df["region"] == region].copy()


def get_stock(universe_df: pd.DataFrame, ticker: str) -> Optional[dict]:
    """Return a single stock row as a dict, or None if not found."""
    row = universe_df[universe_df["ticker"] == ticker]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def all_trend_keywords(universe_df: pd.DataFrame) -> list[str]:
    """Return deduplicated list of all trends_keyword values."""
    kws = universe_df["trends_keyword"].dropna().unique().tolist()
    return kws


def all_sector_etfs(universe_df: pd.DataFrame, params: dict) -> list[str]:
    """Return all unique sector ETF tickers across the universe."""
    etfs = set()
    for _, row in universe_df.iterrows():
        etf = get_sector_etf(row["primary_bucket"], row["region"], params)
        if etf:
            etfs.add(etf)
    # add broad market indices
    for region in ["US", "EU", "CA"]:
        etfs.add(get_market_index(region, params))
    return sorted(etfs)
