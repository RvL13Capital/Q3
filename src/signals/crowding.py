"""
Signal 3: Crowding Score (X_C)

Four sub-components:
  1. Sector ETF correlation  (60-day rolling: high correlation = herd is in)
  2. Relative strength       (6-month vs broad market: outperformance = crowded)
  3. Google Trends           (retail attention peak = late-cycle crowding)
  4. Short interest          (US/CA only; high short = actually anti-crowding)

High crowding_score = crowded = BAD for new entries.
The composite formula inverts this: (1 - crowding) contributes to the final score.
"""
import logging
import time
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sub-score 1: Sector ETF correlation
# ---------------------------------------------------------------------------

def compute_etf_correlation(
    ticker: str,
    sector_etf: str,
    conn,
    params: dict,
    as_of_date: str,
) -> tuple[float, float]:
    """
    Returns (correlation, confidence).
    60-day rolling Pearson correlation of daily log returns.
    """
    from src.data.db import get_prices
    from src.data.prices import compute_log_returns

    window = params["signals"]["crowding"]["etf_corr_window"]
    start = (datetime.strptime(as_of_date, "%Y-%m-%d")
             - timedelta(days=window * 2)).strftime("%Y-%m-%d")

    price_df = get_prices(conn, [ticker, sector_etf], start, as_of_date)
    if price_df.empty or ticker not in price_df.columns or sector_etf not in price_df.columns:
        return 0.5, 0.0

    returns = compute_log_returns(price_df[[ticker, sector_etf]]).dropna()
    n = len(returns)
    if n < 20:
        return 0.5, n / 20.0

    corr = returns[ticker].corr(returns[sector_etf])
    if pd.isna(corr):
        return 0.5, 0.0

    # Rescale: correlation in [-1, 1] → [0, 1] (clamp negatives to 0)
    score = max(0.0, min(1.0, (corr + 1.0) / 2.0))
    confidence = min(1.0, n / window)
    return round(score, 4), round(confidence, 4)


# ---------------------------------------------------------------------------
# Sub-score 2: Relative strength vs broad market
# ---------------------------------------------------------------------------

def compute_relative_strength(
    ticker: str,
    market_index: str,
    conn,
    params: dict,
    as_of_date: str,
) -> tuple[float, float]:
    """
    Returns (rs_score, confidence).
    RS = cumulative return of stock / cumulative return of market index over 126 days.
    High RS (≥3x market) → crowding_score = 1.0
    Low RS (≤0.5x market) → crowding_score = 0.0
    """
    from src.data.db import get_prices

    window = params["signals"]["crowding"].get("rel_strength_window", 126)
    start = (datetime.strptime(as_of_date, "%Y-%m-%d")
             - timedelta(days=window * 2)).strftime("%Y-%m-%d")

    price_df = get_prices(conn, [ticker, market_index], start, as_of_date)
    if price_df.empty:
        return 0.5, 0.0

    if ticker not in price_df.columns or market_index not in price_df.columns:
        return 0.5, 0.0

    prices = price_df[[ticker, market_index]].dropna()
    n = len(prices)
    if n < 30:
        return 0.5, n / window

    # Use last `window` rows
    prices = prices.tail(window)
    ret_stock  = prices[ticker].iloc[-1] / prices[ticker].iloc[0]
    ret_market = prices[market_index].iloc[-1] / prices[market_index].iloc[0]

    if ret_market <= 0:
        return 0.5, 0.0

    rs = ret_stock / ret_market  # e.g. 2.0 = stock gained 2x what market gained
    # Normalize: RS ≤ 0.5 → 0.0, RS ≥ 3.0 → 1.0, linear between
    score = max(0.0, min(1.0, (rs - 0.5) / (3.0 - 0.5)))
    confidence = min(1.0, n / window)
    return round(score, 4), round(confidence, 4)


# ---------------------------------------------------------------------------
# Sub-score 3: Google Trends
# ---------------------------------------------------------------------------

def compute_trends_score(
    trends_keyword: str,
    conn,
    as_of_date: str,
    window_weeks: int = 52,
) -> tuple[float, float]:
    """
    Returns (trends_score, confidence).
    Current interest / max interest over trailing 52 weeks.
    Current ≥ 90th percentile → score 1.0 (highly hyped).
    Current ≤ 20th percentile → score 0.0 (under the radar).
    """
    from src.data.db import get_trends

    start = (datetime.strptime(as_of_date, "%Y-%m-%d")
             - timedelta(weeks=window_weeks + 4)).strftime("%Y-%m-%d")

    series = get_trends(conn, trends_keyword, start, as_of_date)
    if series.empty:
        return 0.5, 0.0  # neutral with zero confidence

    current = series.iloc[-1]
    p20 = series.quantile(0.20)
    p90 = series.quantile(0.90)

    if p90 <= p20:
        return 0.5, 0.5

    score = max(0.0, min(1.0, (current - p20) / (p90 - p20)))
    confidence = min(1.0, len(series) / window_weeks)
    return round(score, 4), round(confidence, 4)


def fetch_google_trends_batch(
    keywords: list[str],
    conn,
    geo: str = "US",
    timeframe: str = "today 12-m",
    delay_secs: float = 5.0,
) -> pd.DataFrame:
    """
    Fetch Google Trends for all keywords (max 5 per pytrends request).
    Stores results in google_trends table.
    Returns combined DataFrame with columns: keyword, date, score.
    """
    from src.data.db import upsert_trends

    try:
        from pytrends.request import TrendReq
    except ImportError:
        logger.warning("pytrends not installed; Google Trends unavailable")
        return pd.DataFrame()

    all_rows = []
    batch_size = 5

    for i in range(0, len(keywords), batch_size):
        batch = keywords[i: i + batch_size]
        try:
            pt = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
            pt.build_payload(batch, timeframe=timeframe, geo=geo)
            df = pt.interest_over_time()
            if df.empty:
                continue
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])

            for kw in batch:
                if kw not in df.columns:
                    continue
                kw_df = df[[kw]].reset_index().rename(
                    columns={"date": "date", kw: "score"}
                )
                kw_df["keyword"] = kw
                kw_df["geo"] = geo
                all_rows.append(kw_df)

            upsert_trends(conn, pd.concat(
                [r[["keyword", "date", "score", "geo"]] for r in all_rows[-len(batch):]]
            ) if all_rows else pd.DataFrame())

        except Exception as e:
            logger.warning(f"Google Trends batch {batch} failed: {e}")

        time.sleep(delay_secs)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Sub-score 4: Short interest
# ---------------------------------------------------------------------------

def compute_short_interest_score(
    ticker: str,
    conn,
    region: str,
) -> tuple[float, float]:
    """
    Returns (score, confidence).
    EU stocks: return (0.5, 0.0) — short data unavailable, neutral.
    US/CA: low short % → score 0.1 (long consensus = possibly crowded)
           high short % → score 0.9 (contrarian signal; shorts = anti-crowding)
    """
    if region == "EU":
        return 0.5, 0.0

    # Try fetching from yfinance info
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        short_pct = info.get("shortPercentOfFloat")
        if short_pct is None:
            return 0.5, 0.0
        # short_pct is a fraction (0.05 = 5%)
        # Low short (<2%) = everyone is long = crowded → score near 0
        # High short (>15%) = heavy short = contrarian setup → score near 1
        score = max(0.0, min(1.0, (short_pct - 0.01) / (0.15 - 0.01)))
        return round(score, 4), 0.8  # lower confidence (data is bi-monthly)
    except Exception:
        return 0.5, 0.0


# ---------------------------------------------------------------------------
# Composite crowding score
# ---------------------------------------------------------------------------

def compute_crowding_score(
    ticker: str,
    trends_keyword: Optional[str],
    sector_etf: Optional[str],
    market_index: str,
    conn,
    params: dict,
    as_of_date: str,
    region: str,
) -> dict:
    """
    Combine four sub-scores into composite crowding_score.
    Weights from params, confidence-weighted.
    High crowding_score = crowded = BAD (will be inverted in composite.py).
    """
    w = params["signals"]["crowding"]

    etf_score, etf_conf = (
        compute_etf_correlation(ticker, sector_etf, conn, params, as_of_date)
        if sector_etf else (0.5, 0.0)
    )
    rs_score, rs_conf = compute_relative_strength(
        ticker, market_index, conn, params, as_of_date
    )
    trend_score, trend_conf = (
        compute_trends_score(trends_keyword, conn, as_of_date)
        if trends_keyword else (0.5, 0.0)
    )
    short_score, short_conf = compute_short_interest_score(ticker, conn, region)

    # Confidence-weighted average
    weights = {
        "etf":   w["etf_corr_weight"],
        "rs":    w["rel_strength_weight"],
        "trend": w["trends_weight"],
        "short": w["short_interest_weight"],
    }
    scores = {"etf": etf_score,   "rs": rs_score,
              "trend": trend_score, "short": short_score}
    confs  = {"etf": etf_conf,    "rs": rs_conf,
              "trend": trend_conf,  "short": short_conf}

    total_w_conf = sum(weights[k] * confs[k] for k in weights)
    if total_w_conf == 0:
        crowding_score = 0.5
        crowding_conf  = 0.0
    else:
        crowding_score = sum(scores[k] * weights[k] * confs[k] for k in weights) / total_w_conf
        crowding_conf  = sum(confs[k] for k in confs) / 4.0

    return {
        "ticker":           ticker,
        "crowding_score":   round(crowding_score, 4),
        "crowding_confidence": round(crowding_conf, 4),
        "etf_correlation":  etf_score,
        "trends_norm":      trend_score,
        "short_pct":        short_score,
    }


def batch_crowding_scores(
    universe_df: pd.DataFrame,
    conn,
    params: dict,
    as_of_date: str,
) -> pd.DataFrame:
    """Compute crowding scores for all stocks."""
    from src.data.universe import get_sector_etf, get_market_index

    rows = []
    for _, stock in universe_df.iterrows():
        sector_etf = get_sector_etf(stock["primary_bucket"], stock["region"], params)
        market_idx = get_market_index(stock["region"], params)
        result = compute_crowding_score(
            stock["ticker"],
            stock.get("trends_keyword"),
            sector_etf,
            market_idx,
            conn,
            params,
            as_of_date,
            stock["region"],
        )
        rows.append(result)
    return pd.DataFrame(rows)
