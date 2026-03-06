"""
Portfolio construction: apply Kelly weights with constraints.

Constraint order (order matters):
  1. Drop stocks below min_position (2%)
  2. Apply per-bucket cap (35% max)
  3. Apply per-stock cap (8% max)
  4. Apply cash floor (≥10% cash → invested ≤ 90%)
  5. Re-check min_position after scaling; drop if needed
  6. Report cash remainder
"""
import logging
from datetime import date as date_type
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def apply_constraints(kelly_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Apply portfolio constraints to raw Kelly weights.

    Args:
        kelly_df: DataFrame with columns: ticker, kelly_25pct, primary_bucket
        params: params.yaml dict

    Returns:
        DataFrame with: ticker, weight, primary_bucket, is_constrained
        Note: weights sum to ≤ (1 - cash_reserve)
    """
    if kelly_df.empty:
        return pd.DataFrame(columns=["ticker", "weight", "primary_bucket", "is_constrained"])

    p = params["kelly"]
    min_pos     = p["min_position"]
    max_pos     = p["max_position"]
    max_bucket  = p["max_bucket"]
    cash_floor  = p["cash_reserve"]
    max_invested = 1.0 - cash_floor

    df = kelly_df.copy()
    df["weight"] = df["kelly_25pct"].clip(lower=0.0)
    df["is_constrained"] = False

    # ── Step 1: Drop stocks below min_position ───────────────────────────────
    df = df[df["weight"] >= min_pos].copy()
    if df.empty:
        return df[["ticker", "weight", "primary_bucket", "is_constrained"]]

    # ── Step 2: Bucket cap enforcement ───────────────────────────────────────
    for bucket, group in df.groupby("primary_bucket"):
        bucket_total = group["weight"].sum()
        if bucket_total > max_bucket:
            scale = max_bucket / bucket_total
            df.loc[df["primary_bucket"] == bucket, "weight"] *= scale
            df.loc[df["primary_bucket"] == bucket, "is_constrained"] = True

    # ── Step 3: Per-stock cap ─────────────────────────────────────────────────
    over_cap = df["weight"] > max_pos
    if over_cap.any():
        df.loc[over_cap, "weight"] = max_pos
        df.loc[over_cap, "is_constrained"] = True

    # ── Step 4: Cash floor ────────────────────────────────────────────────────
    total = df["weight"].sum()
    if total > max_invested:
        scale = max_invested / total
        df["weight"] *= scale
        df["is_constrained"] = True

    # ── Step 5: Re-check min_position ────────────────────────────────────────
    below_min = df["weight"] < min_pos
    if below_min.any():
        df = df[~below_min].copy()
        # Re-apply cash floor
        total = df["weight"].sum()
        if total > max_invested:
            df["weight"] *= max_invested / total

    # ── Step 6: Position count warning ───────────────────────────────────────
    n = len(df)
    if n < 8:
        logger.warning(f"Portfolio has only {n} positions (target: 8–15)")
    elif n > 15:
        logger.warning(f"Portfolio has {n} positions (target: 8–15); consider tightening thresholds")

    return df[["ticker", "weight", "primary_bucket", "is_constrained",
               "composite_score", "kelly_25pct"]].copy()


def construct_portfolio(
    conn,
    scored_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    params: dict,
    as_of_date: str,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Main entry point.
      1. Filter to entry candidates
      2. Compute Kelly weights
      3. Apply constraints
      4. Optionally persist to portfolio_snapshots

    Returns final portfolio DataFrame.
    """
    from src.portfolio.kelly import compute_kelly_weights
    from src.data.db import upsert_portfolio_snapshot, get_latest_portfolio

    kelly_df = compute_kelly_weights(scored_df, conn, params, as_of_date, universe_df)
    if kelly_df.empty:
        logger.info("No entry candidates after Kelly computation")
        return pd.DataFrame()

    portfolio = apply_constraints(kelly_df, params)
    if portfolio.empty:
        logger.info("No positions after constraint application")
        return pd.DataFrame()

    cash_weight = 1.0 - portfolio["weight"].sum()
    logger.info(
        f"Portfolio: {len(portfolio)} positions, "
        f"invested={1-cash_weight:.1%}, cash={cash_weight:.1%}"
    )

    if not dry_run:
        # Determine which positions are new vs existing
        current = get_latest_portfolio(conn)
        current_tickers = set(current["ticker"].tolist()) if not current.empty else set()

        snapshot = portfolio[["ticker", "weight", "primary_bucket",
                               "composite_score", "kelly_25pct"]].copy()
        snapshot["snapshot_date"] = as_of_date
        snapshot["bucket_id"] = snapshot["primary_bucket"]
        snapshot["is_new_position"] = ~snapshot["ticker"].isin(current_tickers)
        upsert_portfolio_snapshot(conn, snapshot)

    return portfolio


def diff_portfolio(
    conn,
    new_portfolio: pd.DataFrame,
    as_of_date: str,
) -> pd.DataFrame:
    """
    Compare new portfolio to most recent prior snapshot.
    Returns DataFrame with action column: 'add'|'remove'|'increase'|'decrease'|'hold'
    """
    from src.data.db import get_latest_portfolio

    prev = get_latest_portfolio(conn)
    if prev.empty:
        diff = new_portfolio[["ticker", "weight"]].copy()
        diff["old_weight"] = 0.0
        diff["new_weight"] = diff["weight"]
        diff["action"] = "add"
        return diff

    prev_map = dict(zip(prev["ticker"], prev["weight"]))
    new_map  = dict(zip(new_portfolio["ticker"], new_portfolio["weight"]))

    all_tickers = set(prev_map) | set(new_map)
    rows = []
    for t in all_tickers:
        old_w = prev_map.get(t, 0.0)
        new_w = new_map.get(t, 0.0)
        if new_w == 0 and old_w > 0:
            action = "remove"
        elif old_w == 0 and new_w > 0:
            action = "add"
        elif new_w > old_w * 1.05:
            action = "increase"
        elif new_w < old_w * 0.95:
            action = "decrease"
        else:
            action = "hold"
        rows.append({"ticker": t, "old_weight": old_w, "new_weight": new_w, "action": action})

    return pd.DataFrame(rows).sort_values("action")
