"""
Swing-trade momentum patterns — timing layer on top of the fundamental filter.

Three complementary patterns detect "huge upside" setups in 5–30 day swing timeframes:

  1. RS Score (Relative Strength Rank)
     IBD-style percentile of a stock's composite return vs all universe peers.
     Institutional accumulation shows up as rising RS *before* price breaks out.
     Formula: percentile_rank(0.60 × 12M_return + 0.40 × 3M_return)
     Most recent rs_skip_days are excluded to reduce short-term noise.

  2. Breakout Score
     O'Neil/Minervini pivot-breakout: price at or just above its N-day high,
     confirmed by elevated volume (institutional conviction).
     price_score  = clamp((close / high_N − 0.90) / 0.15, 0, 1)
       → 0 below 90% of N-day high, 1 at 105%+ (clear breakout)
     volume_factor = clamp(recent_vol / (avg_vol × volume_ratio_thr), 0, 1)
     breakout_score = price_score × volume_factor

  3. VCP Score (Volatility Contraction Pattern)
     Minervini-style ATR compression: tight base = coiling energy.
     atr_ratio = ATR_recent(15d) / ATR_baseline(60d)
     vcp_score  = clamp(1 − atr_ratio / 1.5, 0, 1)
       → 1 on maximum compression, 0 when ATR expands ≥50% above baseline

Composite:
  swing_score = w_rs × rs_score + w_breakout × breakout_score + w_vcp × vcp_score

Integration with composite.py:
  When swing_score < swing_entry_threshold the entry_signal is suppressed.
  The fundamental layer (X_E × X_P × (1−X_C)) acts as the structural filter;
  the momentum layer acts as the timing trigger. A stock must pass BOTH to enter.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-score 1: Relative Strength Rank
# ---------------------------------------------------------------------------

def compute_rs_score(
    ticker: str,
    universe_tickers: list[str],
    conn,
    params: dict,
    as_of_date: str,
) -> tuple[float, float]:
    """
    Returns (rs_score, confidence) both in [0, 1].

    rs_score is the universe percentile rank (0 = weakest, 1 = strongest)
    of the stock's composite return: 60% × 12M + 40% × 3M,
    both skipping the most recent rs_skip_days.

    Stocks with rs_score > 0.80 are in the leadership zone and are being
    accumulated institutionally — these tend to make the biggest moves.
    """
    from src.data.db import get_prices

    mom          = params.get("momentum", {})
    long_window  = int(mom.get("rs_lookback_long",  252))
    short_window = int(mom.get("rs_lookback_short",  63))
    skip_days    = int(mom.get("rs_skip_days",        5))

    total_needed = long_window + skip_days + 10
    start = (
        datetime.strptime(as_of_date, "%Y-%m-%d")
        - timedelta(days=int(total_needed * 1.5))
    ).strftime("%Y-%m-%d")

    price_df = get_prices(conn, universe_tickers, start, as_of_date)
    if price_df.empty or ticker not in price_df.columns:
        return 0.5, 0.0

    price_df = price_df.sort_index()

    def _return(series: pd.Series, lookback: int, skip: int) -> float:
        """(P_end / P_start) − 1, skipping the most recent `skip` bars."""
        s = series.dropna()
        if len(s) < lookback + skip + 1:
            return float("nan")
        end_price   = s.iloc[-(skip + 1)] if skip > 0 else s.iloc[-1]
        start_price = s.iloc[-(lookback + skip + 1)]
        if start_price <= 0:
            return float("nan")
        return end_price / start_price - 1.0

    # Composite RS return for every universe ticker
    rs_map: dict[str, float] = {}
    for t in universe_tickers:
        if t not in price_df.columns:
            continue
        r_long  = _return(price_df[t], long_window,  skip_days)
        r_short = _return(price_df[t], short_window, skip_days)
        if pd.isna(r_long) and pd.isna(r_short):
            continue
        rl = r_long  if not pd.isna(r_long)  else 0.0
        rs = r_short if not pd.isna(r_short) else rl
        rs_map[t] = 0.60 * rl + 0.40 * rs

    if ticker not in rs_map or len(rs_map) < 3:
        return 0.5, 0.0

    values     = np.array(list(rs_map.values()))
    ticker_val = rs_map[ticker]
    rank       = float(np.sum(values < ticker_val) / len(values))

    n_points   = len(price_df[ticker].dropna())
    confidence = min(1.0, n_points / (long_window + skip_days))
    return round(rank, 4), round(confidence, 4)


# ---------------------------------------------------------------------------
# Sub-score 2: Breakout Score
# ---------------------------------------------------------------------------

def compute_breakout_score(
    ticker: str,
    conn,
    params: dict,
    as_of_date: str,
) -> tuple[float, float]:
    """
    Returns (breakout_score, confidence) both in [0, 1].

    Highest score when price is AT or just above its N-day high AND
    recent volume is elevated relative to the 50-day average.

    price_score  = clamp((close / high_N − 0.90) / 0.15, 0, 1)
    volume_factor = clamp(recent_5d_vol / (avg_50d_vol × vol_ratio_thr), 0, 1)
    breakout_score = price_score × volume_factor
    """
    from src.data.db import get_prices

    mom           = params.get("momentum", {})
    high_window   = int(mom.get("breakout_window",         252))
    vol_window    = int(mom.get("breakout_volume_window",   50))
    vol_ratio_thr = float(mom.get("breakout_volume_ratio",  1.5))

    start = (
        datetime.strptime(as_of_date, "%Y-%m-%d")
        - timedelta(days=int(high_window * 1.5))
    ).strftime("%Y-%m-%d")

    price_df = get_prices(conn, [ticker], start, as_of_date)
    if price_df.empty or ticker not in price_df.columns:
        return 0.0, 0.0

    close_series = price_df[ticker].dropna()
    if len(close_series) < 20:
        return 0.0, len(close_series) / 20.0

    # Price component
    high_n        = float(close_series.tail(high_window).max())
    current_price = float(close_series.iloc[-1])
    if high_n <= 0:
        return 0.0, 0.0

    price_ratio = current_price / high_n
    price_score = max(0.0, min(1.0, (price_ratio - 0.90) / 0.15))

    # Volume component — fetch from DB
    vol_score = 0.5  # neutral when volume unavailable
    try:
        vol_df = conn.execute(
            "SELECT date, volume FROM prices WHERE ticker = ? AND date BETWEEN ? AND ? ORDER BY date",
            [ticker, start, as_of_date],
        ).df()
        if not vol_df.empty and "volume" in vol_df.columns:
            vol_df["date"] = pd.to_datetime(vol_df["date"])
            vol_series = vol_df.set_index("date")["volume"].dropna().sort_index()
            if len(vol_series) >= vol_window + 3:
                avg_vol    = float(vol_series.tail(vol_window).mean())
                recent_vol = float(vol_series.tail(5).mean())
                if avg_vol > 0:
                    vol_score = max(0.0, min(1.0, (recent_vol / avg_vol) / vol_ratio_thr))
    except Exception as exc:
        logger.debug("Volume fetch failed for %s: %s", ticker, exc)

    breakout_score = price_score * vol_score
    confidence     = min(1.0, len(close_series) / high_window)
    return round(breakout_score, 4), round(confidence, 4)


# ---------------------------------------------------------------------------
# Sub-score 3: VCP Score (Volatility Contraction Pattern)
# ---------------------------------------------------------------------------

def compute_vcp_score(
    ticker: str,
    conn,
    params: dict,
    as_of_date: str,
) -> tuple[float, float]:
    """
    Returns (vcp_score, confidence) both in [0, 1].

    ATR compression: shrinking daily range signals supply exhaustion and
    base-building — the "coiling spring" before a big move.

    atr_ratio = ATR_recent(15d) / ATR_baseline(60d)
    vcp_score  = clamp(1 − atr_ratio / 1.5, 0, 1)
      → 1.0  at maximum compression (ratio → 0)
      → 0.33 when ATR equals baseline  (ratio = 1)
      → 0.0  when ATR expands ≥50% above baseline (ratio ≥ 1.5)
    """
    from src.data.db import get_prices

    mom              = params.get("momentum", {})
    recent_window    = int(mom.get("vcp_atr_window_recent",   15))
    baseline_window  = int(mom.get("vcp_atr_window_baseline", 60))

    total_needed = baseline_window + recent_window + 10
    start = (
        datetime.strptime(as_of_date, "%Y-%m-%d")
        - timedelta(days=int(total_needed * 1.5))
    ).strftime("%Y-%m-%d")

    price_df = get_prices(conn, [ticker], start, as_of_date)
    if price_df.empty or ticker not in price_df.columns:
        return 0.5, 0.0  # neutral on missing data

    close_series = price_df[ticker].dropna()
    min_needed   = baseline_window + recent_window
    if len(close_series) < min_needed // 2:
        return 0.5, len(close_series) / min_needed

    # ATR proxy: mean absolute daily close change
    daily_range = close_series.diff().abs().dropna()
    if len(daily_range) < recent_window:
        return 0.5, 0.0

    atr_recent   = float(daily_range.tail(recent_window).mean())
    atr_baseline = float(daily_range.tail(baseline_window).mean())

    if atr_baseline <= 0:
        return 0.5, 0.0

    atr_ratio = atr_recent / atr_baseline
    vcp_score = max(0.0, min(1.0, 1.0 - atr_ratio / 1.5))

    confidence = min(1.0, len(daily_range) / baseline_window)
    return round(vcp_score, 4), round(confidence, 4)


# ---------------------------------------------------------------------------
# Composite swing score
# ---------------------------------------------------------------------------

def compute_momentum_score(
    ticker: str,
    universe_tickers: list[str],
    conn,
    params: dict,
    as_of_date: str,
) -> dict:
    """
    Combine RS rank, breakout, and VCP into a composite swing_score [0, 1].

    High swing_score = stock has:
      • Strong RS (institutional accumulation already underway)
      • Price at or near a pivot breakout point with volume
      • Tight base / low ATR (base fully formed, ready to launch)

    This is the Minervini/O'Neil Stage-2 breakout setup.
    Used as a timing gate on top of the fundamental composite signal.
    """
    mom        = params.get("momentum", {})
    w_rs       = float(mom.get("swing_weight_rs",       0.40))
    w_breakout = float(mom.get("swing_weight_breakout", 0.40))
    w_vcp      = float(mom.get("swing_weight_vcp",      0.20))

    rs_score,       rs_conf    = compute_rs_score(ticker, universe_tickers, conn, params, as_of_date)
    breakout_score, break_conf = compute_breakout_score(ticker, conn, params, as_of_date)
    vcp_score,      vcp_conf   = compute_vcp_score(ticker, conn, params, as_of_date)

    w_total = w_rs + w_breakout + w_vcp
    if w_total <= 0:
        w_rs = w_breakout = w_vcp = 1.0 / 3.0
        w_total = 1.0

    swing_score = (
        (w_rs       / w_total) * rs_score
        + (w_breakout / w_total) * breakout_score
        + (w_vcp      / w_total) * vcp_score
    )
    swing_confidence = (
        (w_rs       / w_total) * rs_conf
        + (w_breakout / w_total) * break_conf
        + (w_vcp      / w_total) * vcp_conf
    )

    return {
        "ticker":           ticker,
        "swing_score":      round(swing_score, 4),
        "swing_confidence": round(swing_confidence, 4),
        "rs_rank":          round(rs_score, 4),
        "breakout_score":   round(breakout_score, 4),
        "vcp_score":        round(vcp_score, 4),
    }


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------

def batch_momentum_scores(
    universe_df: pd.DataFrame,
    conn,
    params: dict,
    as_of_date: str,
) -> pd.DataFrame:
    """
    Compute swing momentum scores for all universe stocks.
    RS rank requires the full universe price series for peer comparison —
    computed once per batch run.

    Returns DataFrame:
      ticker, swing_score, swing_confidence, rs_rank, breakout_score, vcp_score
    """
    all_tickers = universe_df["ticker"].tolist()
    if not all_tickers:
        return pd.DataFrame(columns=[
            "ticker", "swing_score", "swing_confidence",
            "rs_rank", "breakout_score", "vcp_score",
        ])

    rows = []
    for _, stock in universe_df.iterrows():
        result = compute_momentum_score(
            stock["ticker"], all_tickers, conn, params, as_of_date
        )
        rows.append(result)

    return pd.DataFrame(rows)
