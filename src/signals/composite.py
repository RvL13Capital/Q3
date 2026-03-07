"""
Composite signal: combines physical, quality, and crowding into a single score.
Uses confidence-weighted averaging so missing data contributes nothing.

Entry:  composite >= 0.55 AND crowding <= 0.40
Exit:   crowding >= 0.75  OR  quality <= 0.25  OR  composite < entry * 0.80
"""
import logging
from datetime import date as date_type
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_COMPOSITE_CONFIDENCE = 0.40  # below this → insufficient data → excluded


# ---------------------------------------------------------------------------
# μ estimate mapping
# ---------------------------------------------------------------------------

def _interpolate_mu(composite_score: float, params: dict, rf: float) -> float:
    """
    Map composite_score → expected excess return above rf via anchor points.
    Returns annualized expected return (decimal).
    """
    anchors = params["return_estimation"]["composite_anchors"]
    anchors = sorted(anchors, key=lambda x: x[0])

    score = composite_score
    # Below lowest anchor
    if score <= anchors[0][0]:
        excess = anchors[0][1]
    # Above highest anchor
    elif score >= anchors[-1][0]:
        excess = anchors[-1][1]
    else:
        # Linear interpolation between brackets
        for i in range(len(anchors) - 1):
            lo_score, lo_excess = anchors[i]
            hi_score, hi_excess = anchors[i + 1]
            if lo_score <= score <= hi_score:
                t = (score - lo_score) / (hi_score - lo_score)
                excess = lo_excess + t * (hi_excess - lo_excess)
                break
        else:
            excess = anchors[-1][1]

    return rf + excess


# ---------------------------------------------------------------------------
# Main composite function
# ---------------------------------------------------------------------------

def compute_composite_score(
    physical: dict,
    quality: dict,
    crowding: dict,
    params: dict,
    rf: float = 0.04,
) -> dict:
    """
    Combine physical, quality, and crowding into composite score.

    physical  dict keys: physical_norm, physical_confidence
    quality   dict keys: quality_score, quality_confidence
    crowding  dict keys: crowding_score, crowding_confidence

    Returns full dict suitable for inserting into signal_scores table.
    """
    ticker = physical.get("ticker", quality.get("ticker", crowding.get("ticker", "")))

    p_norm = physical.get("physical_norm", 0.0)
    p_conf = physical.get("physical_confidence", 0.0)

    q_score = quality.get("quality_score", 0.5)
    q_conf  = quality.get("quality_confidence", 0.0)

    c_score = crowding.get("crowding_score", 0.5)
    c_conf  = crowding.get("crowding_confidence", 0.0)
    c_inv   = 1.0 - c_score  # invert crowding: low crowding = good

    w = params["signals"]["weights"]
    w_p = w["physical"]
    w_q = w["quality"]
    w_c = w["crowding"]

    # Confidence-weighted numerator and denominator
    denom = w_p * p_conf + w_q * q_conf + w_c * c_conf
    if denom == 0:
        composite = float("nan")
        comp_conf = 0.0
    else:
        composite = (w_p * p_norm * p_conf + w_q * q_score * q_conf + w_c * c_inv * c_conf) / denom
        comp_conf = (p_conf + q_conf + c_conf) / 3.0

    # Data gate: if insufficient data, mark as NaN
    if comp_conf < MIN_COMPOSITE_CONFIDENCE:
        composite = float("nan")

    # Entry / exit signals
    entry_threshold = params["signals"]["entry_threshold"]
    crowd_entry_max = params["signals"]["crowding_entry_max"]
    crowd_exit_thr  = params["signals"]["crowding_exit_threshold"]
    qual_exit_thr   = params["signals"]["quality_exit_threshold"]

    if not pd.isna(composite):
        entry_signal = (
            composite >= entry_threshold
            and c_score <= crowd_entry_max
            and q_score >= qual_exit_thr
        )
        exit_signal = (
            c_score >= crowd_exit_thr
            or q_score <= qual_exit_thr
        )
    else:
        entry_signal = False
        exit_signal  = False

    # μ estimate (only for entry candidates)
    if not pd.isna(composite) and composite >= entry_threshold:
        mu_estimate = _interpolate_mu(composite, params, rf)
    else:
        mu_estimate = None

    return {
        "ticker":               ticker,
        "physical_raw":         physical.get("physical_raw"),
        "physical_norm":        physical.get("physical_norm"),
        "physical_confidence":  p_conf,
        "quality_score":        quality.get("quality_score"),
        "quality_confidence":   q_conf,
        "roic_wacc_spread":     quality.get("roic_wacc_spread"),
        "margin_snr":           quality.get("margin_snr"),
        "inflation_convexity":  quality.get("inflation_convexity"),
        "crowding_score":       c_score,
        "crowding_confidence":  c_conf,
        "etf_correlation":      crowding.get("etf_correlation"),
        "trends_norm":          crowding.get("trends_norm"),
        "short_pct":            crowding.get("short_pct"),
        "composite_score":      round(composite, 4) if not pd.isna(composite) else None,
        "composite_confidence": round(comp_conf, 4),
        "mu_estimate":          round(mu_estimate, 4) if mu_estimate else None,
        "sigma_estimate":       None,  # filled in by kelly.py
        "kelly_fraction":       None,
        "kelly_25pct":          None,
        "entry_signal":         entry_signal,
        "exit_signal":          exit_signal,
    }


# ---------------------------------------------------------------------------
# Full weekly scoring run
# ---------------------------------------------------------------------------

def run_weekly_scoring(
    conn,
    universe_df: pd.DataFrame,
    params: dict,
    as_of_date: str,
) -> pd.DataFrame:
    """
    Orchestrates the full scoring pipeline for all universe stocks.
    Stores results in signal_scores table.
    Returns the scored DataFrame.
    """
    from src.signals.physical  import batch_physical_scores
    from src.signals.quality   import batch_quality_scores
    from src.signals.crowding  import batch_crowding_scores
    from src.data.macro        import get_risk_free_rate
    from src.data.db           import upsert_signal_scores

    logger.info(f"Running weekly scoring for {len(universe_df)} stocks ({as_of_date})")

    physical_df = batch_physical_scores(universe_df)
    quality_df  = batch_quality_scores(universe_df, conn, params, as_of_date)
    crowding_df = batch_crowding_scores(universe_df, conn, params, as_of_date)

    rows = []
    for _, stock in universe_df.iterrows():
        ticker = stock["ticker"]
        region = stock["region"]

        p = physical_df[physical_df["ticker"] == ticker].iloc[0].to_dict() if not physical_df[physical_df["ticker"] == ticker].empty else {}
        q = quality_df[quality_df["ticker"] == ticker].iloc[0].to_dict() if not quality_df[quality_df["ticker"] == ticker].empty else {}
        c = crowding_df[crowding_df["ticker"] == ticker].iloc[0].to_dict() if not crowding_df[crowding_df["ticker"] == ticker].empty else {}

        rf = get_risk_free_rate(conn, region, as_of_date, params)
        result = compute_composite_score(p, q, c, params, rf=rf)
        result["score_date"] = as_of_date
        rows.append(result)

    scored_df = pd.DataFrame(rows)
    upsert_signal_scores(conn, scored_df)

    n_entry = int(scored_df["entry_signal"].sum()) if "entry_signal" in scored_df.columns else 0
    n_exit  = int(scored_df["exit_signal"].sum())  if "exit_signal"  in scored_df.columns else 0
    logger.info(f"Scoring complete: {n_entry} entry signals, {n_exit} exit signals")

    return scored_df
