"""
Composite signal — EARKE eq 7 (Basis-Drift Synthese):

  composite = X_E × X_P × (1 − X_C)
  μ_base    = rf + θ × composite

Multiplicative structure: all three tensors must be favourable simultaneously.
X_E = physical/EROEI score (0–1)
X_P = quality/moat score   (0–1, 0 when ROIC ≤ WACC)
X_C = crowding/CSD score   (0–1, inverted to 1−X_C)
θ   = risk-premium scalar (params: return_estimation.theta_risk_premium)

Entry:  composite >= 0.30 AND crowding <= 0.40
        AND (swing_score >= swing_entry_threshold  OR  threshold == 0)
Exit:   crowding >= 0.75  OR  quality <= 0.25  OR  composite < entry * 0.80

Swing timing gate (Module II.5 — price-structure timing):
  When params.momentum.swing_entry_threshold > 0, entry_signal is suppressed
  unless the swing_score (RS rank + breakout + VCP) also clears the threshold.
  This adds a price-structure timing layer on top of the fundamental filter.
  Set swing_entry_threshold: 0.0 to run in pure-fundamental mode.
"""
import logging
from datetime import date as date_type
from typing import Optional



import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Kept as a module-level constant so test code that imports it directly still works.
# Runtime reads params["signals"]["min_composite_confidence"] which overrides this.
MIN_COMPOSITE_CONFIDENCE = 0.40


# ---------------------------------------------------------------------------
# Data Maturity Score (Fix 4)
# ---------------------------------------------------------------------------

def compute_data_maturity(
    n_price_days: int,
    n_fundamental_years: int,
    n_margin_years: int,
    params: dict,
) -> float:
    """
    Data Maturity Score — geometric mean of three maturity dimensions.

    Each dimension: score_i = min(1.0, actual / required).
    Combined: dms = geometric_mean(scores), floored at penalty_floor.

    Geometric mean penalizes stocks weak in any dimension.
    Returns a value in [penalty_floor, 1.0].
    """
    dm_params = params.get("data_maturity", {})
    min_price = dm_params.get("min_price_days", 252)
    min_fund = dm_params.get("min_fundamental_years", 3)
    min_margin = dm_params.get("min_margin_years", 3)
    floor = dm_params.get("penalty_floor", 0.30)

    # Individual dimension scores
    price_score = min(1.0, n_price_days / max(1, min_price))
    fund_score = min(1.0, n_fundamental_years / max(1, min_fund))
    margin_score = min(1.0, n_margin_years / max(1, min_margin))

    # Geometric mean of the three scores
    product = price_score * fund_score * margin_score
    if product <= 0:
        dms = 0.0
    else:
        dms = product ** (1.0 / 3.0)

    return max(floor, dms)


# ---------------------------------------------------------------------------
# Main composite function
# ---------------------------------------------------------------------------

def compute_composite_score(
    physical: dict,
    quality: dict,
    crowding: dict,
    params: dict,
    rf: float = 0.04,
    swing: Optional[dict] = None,
    data_maturity: Optional[float] = None,
) -> dict:
    """
    Combine physical, quality, and crowding into composite score.

    physical  dict keys: physical_norm, physical_confidence
    quality   dict keys: quality_score, quality_confidence
    crowding  dict keys: crowding_score, crowding_confidence

    Returns full dict suitable for inserting into signal_scores table.
    """
    ticker = physical.get("ticker", quality.get("ticker", crowding.get("ticker", "")))

    p_norm = physical.get("physical_norm", 0.0)   # X_E
    p_conf = physical.get("physical_confidence", 0.0)

    q_score = quality.get("quality_score", 0.0)   # X_P
    q_conf  = quality.get("quality_confidence", 0.0)

    c_score = crowding.get("crowding_score", 0.5)  # X_C
    c_conf  = crowding.get("crowding_confidence", 0.0)
    c_inv   = 1.0 - c_score                        # (1 − X_C)

    signal_conf = (p_conf + q_conf + c_conf) / 3.0
    dm_params = params.get("data_maturity", {})
    dm_enabled = dm_params.get("enabled", False)
    dm_blend = dm_params.get("composite_confidence_blend", 0.30)
    if dm_enabled and data_maturity is not None:
        comp_conf = (1.0 - dm_blend) * signal_conf + dm_blend * data_maturity
    else:
        comp_conf = signal_conf

    # Data gate — threshold from params; fall back to module constant for compat.
    min_conf = params.get("signals", {}).get(
        "min_composite_confidence", MIN_COMPOSITE_CONFIDENCE
    )
    if comp_conf < min_conf:
        composite = float("nan")
    else:
        # Multiplicative per eq 7: X_E × X_P × (1 − X_C)
        composite = p_norm * q_score * c_inv

    # μ_base per eq 7: rf + θ × X_P × X_E × (1 − X_C) = rf + θ × composite
    theta = params["return_estimation"].get("theta_risk_premium", 0.30)
    entry_threshold = params["signals"]["entry_threshold"]
    crowd_entry_max = params["signals"]["crowding_entry_max"]
    crowd_exit_thr  = params["signals"]["crowding_exit_threshold"]
    qual_exit_thr   = params["signals"]["quality_exit_threshold"]

    # Swing timing gate values (optional — None when momentum module not run)
    s_score = swing.get("swing_score")      if swing else None
    s_conf  = swing.get("swing_confidence") if swing else None

    if not pd.isna(composite):
        mu_estimate = rf + theta * composite
        entry_signal = (
            composite >= entry_threshold
            and c_score <= crowd_entry_max
            and q_score >= qual_exit_thr  # no separate entry quality gate — don't enter what we'd exit
        )
        # Apply swing timing gate when configured and data is available
        swing_thr = params.get("momentum", {}).get("swing_entry_threshold", 0.0)
        if s_score is not None and swing_thr > 0:
            entry_signal = entry_signal and (s_score >= swing_thr)
        exit_signal = (
            c_score >= crowd_exit_thr
            or q_score <= qual_exit_thr
        )
    else:
        mu_estimate  = None
        entry_signal = False
        exit_signal  = False

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
        "autocorr_delta":       crowding.get("autocorr_delta"),
        "absorption_delta":     crowding.get("absorption_delta"),
        "etf_corr_score":       crowding.get("etf_corr_score"),
        "short_interest_score": crowding.get("short_interest_score"),
        "swing_score":          round(s_score, 4) if s_score is not None else None,
        "swing_confidence":     round(s_conf,  4) if s_conf  is not None else None,
        "rs_rank":              round(swing.get("rs_rank",       0.0), 4) if swing else None,
        "breakout_score":       round(swing.get("breakout_score", 0.0), 4) if swing else None,
        "vcp_score":            round(swing.get("vcp_score",     0.0), 4) if swing else None,
        "composite_score":      round(composite, 4) if not pd.isna(composite) else None,
        "composite_confidence": round(comp_conf, 4),
        "mu_estimate":          round(mu_estimate, 4) if mu_estimate is not None else None,
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

    physical_df  = batch_physical_scores(universe_df, conn=conn, params=params, as_of_date=as_of_date)
    quality_df   = batch_quality_scores(universe_df, conn, params, as_of_date)
    crowding_df  = batch_crowding_scores(universe_df, conn, params, as_of_date)

    # Swing momentum timing layer (optional — gracefully disabled when no price data)
    from src.signals.momentum import batch_momentum_scores
    try:
        momentum_df = batch_momentum_scores(universe_df, conn, params, as_of_date)
    except Exception as e:
        logger.warning(f"Momentum scoring failed (non-fatal): {e}")
        momentum_df = pd.DataFrame()

    # Pre-fetch rf rates once per region for the mu_base calculation.
    regions  = universe_df["region"].unique().tolist()
    rf_cache = {r: get_risk_free_rate(conn, r, as_of_date, params, as_of_tk=as_of_date) for r in regions}

    # Index sub-score DataFrames for O(1) lookup (preserve "ticker" key in each row dict).
    p_idx = {r["ticker"]: r.to_dict() for _, r in physical_df.iterrows()}  if not physical_df.empty  else {}
    q_idx = {r["ticker"]: r.to_dict() for _, r in quality_df.iterrows()}   if not quality_df.empty   else {}
    c_idx = {r["ticker"]: r.to_dict() for _, r in crowding_df.iterrows()}  if not crowding_df.empty  else {}
    m_idx = {r["ticker"]: r.to_dict() for _, r in momentum_df.iterrows()}  if not momentum_df.empty  else {}

    rows = []
    for _, stock in universe_df.iterrows():
        ticker = stock["ticker"]
        region = stock["region"]

        p = p_idx.get(ticker, {})
        q = q_idx.get(ticker, {})
        c = c_idx.get(ticker, {})
        m = m_idx.get(ticker) or None   # None → gate disabled for this stock

        rf = rf_cache.get(region, rf_cache.get("US", 0.04))

        # Data Maturity Score (Fix 4)
        dms_value = None
        dm_params = params.get("data_maturity", {})
        if dm_params.get("enabled", False):
            from src.data.db import get_margin_history
            # Count price days available for this ticker
            price_count = conn.execute(
                "SELECT COUNT(DISTINCT date) FROM prices WHERE ticker = ? AND date <= ?",
                [ticker, as_of_date],
            ).fetchone()[0]
            n_price_days = price_count or 0
            # Count fundamental years
            margin_df = get_margin_history(conn, ticker, n_years=10, as_of_date=as_of_date)
            if not margin_df.empty:
                n_fund_years = margin_df["fiscal_year"].nunique() if "fiscal_year" in margin_df.columns else 0
                n_margin_years = margin_df["gross_margin"].dropna().shape[0]
            else:
                n_fund_years = 0
                n_margin_years = 0
            dms_value = compute_data_maturity(n_price_days, n_fund_years, n_margin_years, params)

        result = compute_composite_score(p, q, c, params, rf=rf, swing=m, data_maturity=dms_value)
        result["score_date"] = as_of_date
        rows.append(result)

    scored_df = pd.DataFrame(rows)
    upsert_signal_scores(conn, scored_df)

    n_entry = int(scored_df["entry_signal"].sum()) if "entry_signal" in scored_df.columns else 0
    n_exit  = int(scored_df["exit_signal"].sum())  if "exit_signal"  in scored_df.columns else 0
    logger.info(f"Scoring complete: {n_entry} entry signals, {n_exit} exit signals")

    return scored_df
