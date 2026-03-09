"""
Signal 1: Physical Necessity Score (X_E) — EARKE eq 4

  X_E,t = 1 / (1 + exp(κ · (ECS_proxy,t − ECS_crit)))

ECS_proxy = percentile rank of current energy PPI (FRED PPIENG) over a rolling
lookback window (default: 60 months). This proxies the Energy Cost Share of GDP:
when energy prices are elevated relative to their recent history, the macro
environment structurally restricts alpha generation.

  ECS_proxy ≈ 0.0  → energy cheap  → X_E ≈ 1.0  (macro brake inactive)
  ECS_proxy = ECS_crit (0.70) → X_E = 0.5  (50% damped)
  ECS_proxy ≈ 1.0  → energy expensive → X_E ≈ 0.0  (alpha structurally capped)

Fallback: if PPIENG data unavailable, uses a stock-level bucket-count score as
an approximation (preserving the original behaviour before this upgrade).

κ = 10.0  (calibrated for 0–1 percentile scale; spec uses 50 for raw ECS %)
ECS_crit  = 0.70  (70th percentile historically = structural headwind)
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Bucket-count fallback (original approximation, kept as graceful degradation)
_BUCKET_SCORE_MAP = {0: 0.0, 1: 1.5, 2: 2.0, 3: 3.0}


# ---------------------------------------------------------------------------
# X_E: EROEI logistic damper (eq 4)
# ---------------------------------------------------------------------------

def compute_ecs_x_e(
    conn,
    params: dict,
    as_of_date: str,
) -> tuple[float, float]:
    """
    Returns (x_e, ecs_proxy).

    x_e       — logistic damper in [0, 1]
    ecs_proxy — current energy PPI percentile over lookback window [0, 1]

    Uses FRED PPIENG (energy producer price index, level) stored in macro_series.
    Percentile-ranks the current value against the trailing lookback window.
    """
    from src.data.db import get_macro_series

    physical_params = params.get("physical", {})
    kappa          = float(physical_params.get("eroei_kappa", 10.0))
    ecs_crit       = float(physical_params.get("ecs_crit_percentile", 0.70))
    lookback_months = int(physical_params.get("ecs_lookback_months", 60))
    series_id      = physical_params.get("us_energy_ppi_series", "US_PPIENG")

    start = (datetime.strptime(as_of_date, "%Y-%m-%d")
             - timedelta(days=lookback_months * 31)).strftime("%Y-%m-%d")

    series = get_macro_series(conn, series_id, start, as_of_date)
    if series.empty or len(series) < 6:
        return float("nan"), float("nan")  # caller will use bucket fallback

    current_val  = float(series.iloc[-1])
    history_vals = series.values.astype(float)

    # Percentile rank of current value in rolling history (0 = cheapest, 1 = most expensive)
    ecs_proxy = float(np.sum(history_vals <= current_val) / len(history_vals))

    x_e = 1.0 / (1.0 + np.exp(kappa * (ecs_proxy - ecs_crit)))
    return round(float(x_e), 4), round(ecs_proxy, 4)


# ---------------------------------------------------------------------------
# Bucket-count fallback (used when PPIENG unavailable)
# ---------------------------------------------------------------------------

def _bucket_x_e(stock: dict) -> float:
    buckets = stock.get("buckets") or []
    if isinstance(buckets, str):
        buckets = [b.strip() for b in buckets.split(",")]
    bucket_count = len(set(buckets))
    raw  = _BUCKET_SCORE_MAP.get(min(bucket_count, 3), 0.0)
    return round(raw / 3.0, 4)


# ---------------------------------------------------------------------------
# Per-stock score assembly
# ---------------------------------------------------------------------------

def compute_physical_score(
    stock: dict | pd.Series,
    x_e_value: Optional[float] = None,
    ecs_proxy: Optional[float] = None,
    params: Optional[dict] = None,
) -> dict:
    """
    Assemble physical score for a single stock.

    x_e_value — pre-computed logistic X_E (passed in from batch to avoid N DB hits)
    ecs_proxy — the ECS percentile that produced x_e_value (for reporting)
    params    — full params dict; used to read bucket_fallback_confidence

    If x_e_value is None or nan, falls back to bucket-count approximation.
    Confidence is 1.0 when the logistic formula succeeds; reads
    params.physical.bucket_fallback_confidence (default 0.40) on fallback,
    because the bucket approximation is materially less precise than PPIENG data.
    """
    if isinstance(stock, pd.Series):
        stock = stock.to_dict()

    fallback_conf = 0.40
    if params is not None:
        fallback_conf = float(
            params.get("physical", {}).get("bucket_fallback_confidence", 0.40)
        )

    if x_e_value is not None and not (isinstance(x_e_value, float) and np.isnan(x_e_value)):
        norm         = x_e_value
        method       = "logistic"
        bucket_count = len(set(stock.get("buckets") or []))
        raw          = _BUCKET_SCORE_MAP.get(min(bucket_count, 3), 0.0)
        confidence   = 1.0
    else:
        bucket_count = len(set(stock.get("buckets") or []))
        raw          = _BUCKET_SCORE_MAP.get(min(bucket_count, 3), 0.0)
        norm         = raw / 3.0
        method       = "bucket_fallback"
        ecs_proxy    = None
        confidence   = fallback_conf

    return {
        "ticker":               stock.get("ticker", ""),
        "bucket_count":         bucket_count,
        "primary_bucket":       stock.get("primary_bucket"),
        "ecs_proxy":            ecs_proxy,
        "physical_raw":         raw,   # exact bucket-step value (0, 1.5, 2.0, 3.0)
        "physical_norm":        norm,
        "physical_confidence":  confidence,
        "physical_method":      method,
    }


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------

def batch_physical_scores(
    universe_df: pd.DataFrame,
    conn=None,
    params: dict = None,
    as_of_date: str = None,
) -> pd.DataFrame:
    """
    Compute physical scores for all stocks.

    When conn + params + as_of_date are provided, uses the logistic X_E formula.
    Falls back to bucket-count if PPIENG data is unavailable or args are missing.
    X_E is computed once (macro-level signal) and shared across all stocks.
    """
    x_e_value, ecs_proxy = float("nan"), float("nan")

    if conn is not None and params is not None and as_of_date is not None:
        x_e_value, ecs_proxy = compute_ecs_x_e(conn, params, as_of_date)
        if not np.isnan(x_e_value):
            logger.info(
                f"X_E logistic: ECS_proxy={ecs_proxy:.2f}  X_E={x_e_value:.3f}"
            )
        else:
            logger.info("X_E: PPIENG data unavailable — using bucket-count fallback")

    rows = [
        compute_physical_score(
            row,
            x_e_value=x_e_value if not np.isnan(x_e_value) else None,
            ecs_proxy=ecs_proxy if not np.isnan(ecs_proxy) else None,
            params=params,
        )
        for _, row in universe_df.iterrows()
    ]
    return pd.DataFrame(rows)
