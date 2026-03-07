"""
Event-driven monitor: scans for entry/exit threshold crossings each week.
Only produces actionable output when a threshold is crossed.
No threshold crossing = "no action required" week.
"""
import logging
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exit signal evaluation
# ---------------------------------------------------------------------------

def check_exit_signals(
    conn,
    current_portfolio: pd.DataFrame,
    params: dict,
    as_of_date: str,
) -> pd.DataFrame:
    """
    For each currently held position, evaluate exit triggers.

    EXIT TRIGGERS (any one is sufficient):
      1. Crowding ≥ params['crowding_exit_threshold']   (herd is fully in)
      2. Quality  ≤ params['quality_exit_threshold']    (moat has eroded)
      3. Composite decay > 20% from entry composite     (thesis weakened)
      4. Drawdown from entry price > 35%               (price stop-loss)
      5. Bucket weight drift > 40% (5pp above construction cap)

    Returns DataFrame: ticker, exit_triggered, exit_reasons
    """
    from src.data.db import get_latest_signal_scores, get_prices, get_position_entry_date

    crowd_thr  = params["signals"]["crowding_exit_threshold"]
    qual_thr   = params["signals"]["quality_exit_threshold"]
    decay_pct  = params["signals"]["composite_decay_pct"]
    max_bucket = params["kelly"]["max_bucket"] + 0.05  # drift trigger = cap + 5pp

    if current_portfolio.empty:
        return pd.DataFrame()

    tickers = current_portfolio["ticker"].tolist()
    scores  = get_latest_signal_scores(conn, tickers)
    scores_map = scores.set_index("ticker").to_dict("index") if not scores.empty else {}

    # Bucket weight check
    bucket_totals = current_portfolio.groupby("primary_bucket")["weight"].sum().to_dict() \
        if "primary_bucket" in current_portfolio.columns else {}

    rows = []
    for _, pos in current_portfolio.iterrows():
        ticker  = pos["ticker"]
        weight  = pos["weight"]
        reasons = []

        s = scores_map.get(ticker, {})
        crowding  = s.get("crowding_score")
        quality   = s.get("quality_score")
        composite = s.get("composite_score")

        # Trigger 1: crowding
        if crowding is not None and crowding >= crowd_thr:
            reasons.append(f"CROWDED ({crowding:.2f} ≥ {crowd_thr})")

        # Trigger 2: quality
        if quality is not None and quality <= qual_thr:
            reasons.append(f"QUALITY_DETERIORATED ({quality:.2f} ≤ {qual_thr})")

        # Trigger 3: composite decay (compare to entry composite stored in snapshot)
        entry_composite = pos.get("composite_score")  # composite at time of entry
        if (composite is not None and entry_composite is not None
                and not pd.isna(composite) and not pd.isna(entry_composite)
                and entry_composite > 0):
            decay = (entry_composite - composite) / entry_composite
            if decay > decay_pct:
                reasons.append(f"COMPOSITE_DECAY ({decay:.1%} from entry)")

        # Trigger 4: price drawdown from entry
        entry_date_val = get_position_entry_date(conn, ticker)
        if entry_date_val:
            entry_date_str = entry_date_val.isoformat() if hasattr(entry_date_val, "isoformat") else str(entry_date_val)
            prices_df = get_prices(conn, [ticker], entry_date_str, as_of_date)
            if not prices_df.empty and ticker in prices_df.columns:
                entry_price = prices_df[ticker].dropna().iloc[0]
                current_price = prices_df[ticker].dropna().iloc[-1]
                if entry_price > 0:
                    drawdown = (entry_price - current_price) / entry_price
                    if drawdown > 0.35:
                        reasons.append(f"DRAWDOWN ({drawdown:.1%} from entry)")

        # Trigger 5: bucket weight drift
        bucket = pos.get("primary_bucket") or pos.get("bucket_id")
        if bucket and bucket in bucket_totals:
            bucket_w = bucket_totals[bucket]
            if bucket_w > max_bucket:
                reasons.append(f"BUCKET_DRIFT ({bucket} at {bucket_w:.1%} > {max_bucket:.1%})")

        rows.append({
            "ticker":          ticker,
            "exit_triggered":  len(reasons) > 0,
            "exit_reasons":    reasons,
            "crowding_score":  crowding,
            "quality_score":   quality,
            "composite_score": composite,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry candidate identification
# ---------------------------------------------------------------------------

def check_entry_signals(
    conn,
    current_portfolio: pd.DataFrame,
    scored_df: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """
    Return entry candidates not currently held, sorted by composite_score desc.
    """
    current_tickers = set(current_portfolio["ticker"].tolist()) if not current_portfolio.empty else set()

    candidates = scored_df[
        (scored_df["entry_signal"] == True)
        & (~scored_df["ticker"].isin(current_tickers))
    ].copy()

    if candidates.empty:
        return pd.DataFrame()

    return candidates.sort_values("composite_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# WATCH list (crowding approaching exit threshold)
# ---------------------------------------------------------------------------

def get_watch_list(
    current_portfolio: pd.DataFrame,
    scored_df: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """
    Positions where crowding is elevated but not yet at exit threshold.
    These need closer monitoring.
    """
    watch_thr  = params["reporting"].get("watch_crowding_threshold", 0.55)
    exit_thr   = params["signals"]["crowding_exit_threshold"]

    current_tickers = set(current_portfolio["ticker"].tolist()) if not current_portfolio.empty else set()

    watch = scored_df[
        scored_df["ticker"].isin(current_tickers)
        & scored_df["crowding_score"].between(watch_thr, exit_thr, inclusive="left")
    ].copy()

    return watch.sort_values("crowding_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main event generator
# ---------------------------------------------------------------------------

def generate_weekly_actions(
    conn,
    scored_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    params: dict,
    as_of_date: str,
) -> dict:
    """
    Top-level monitor function.
    Returns dict with keys: exits, entries, watch_list, new_portfolio, diff.
    """
    from src.data.db import get_latest_portfolio
    from src.portfolio.construction import construct_portfolio, diff_portfolio

    current_portfolio = get_latest_portfolio(conn)

    # Add primary_bucket to current portfolio for drift check
    if not current_portfolio.empty and "primary_bucket" not in current_portfolio.columns:
        if "bucket_id" in current_portfolio.columns:
            current_portfolio = current_portfolio.rename(columns={"bucket_id": "primary_bucket"})

    exits      = check_exit_signals(conn, current_portfolio, params, as_of_date)
    entries    = check_entry_signals(conn, current_portfolio, scored_df, params)
    watch_list = get_watch_list(current_portfolio, scored_df, params)

    # Construct the new target portfolio
    new_portfolio = construct_portfolio(
        conn, scored_df, universe_df, params, as_of_date
    )

    # Diff: what changed vs last week
    diff = diff_portfolio(conn, new_portfolio, as_of_date) if not new_portfolio.empty else pd.DataFrame()

    n_exits   = exits["exit_triggered"].sum() if not exits.empty else 0
    n_entries = len(entries)
    n_watch   = len(watch_list)

    if n_exits == 0 and n_entries == 0 and n_watch == 0:
        logger.info(f"[{as_of_date}] No action required this week.")
    else:
        logger.info(
            f"[{as_of_date}] Actions: {n_exits} exits, {n_entries} new entries, "
            f"{n_watch} on watch list"
        )

    return {
        "exits":         exits,
        "entries":       entries,
        "watch_list":    watch_list,
        "new_portfolio": new_portfolio,
        "diff":          diff,
        "as_of_date":    as_of_date,
        "any_action":    n_exits > 0 or n_entries > 0,
    }
