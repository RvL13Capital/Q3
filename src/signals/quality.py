"""
Signal 2: Quality / Moat Score (X_P)

Three sub-components:
  1. ROIC − WACC spread     (profitability above cost of capital)
  2. Gross margin SNR        (margin stability = moat strength)
  3. Inflation convexity     (pricing power: revenue growth > PPI)

Each sub-score is 0–1 with a confidence weight.
Confidence-weighted average → final quality_score 0–1.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _clamp_normalize(value: float, low: float, high: float) -> float:
    """Linear normalize value to [0,1] between low and high. Clamp at extremes."""
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


# ---------------------------------------------------------------------------
# Sub-score 1: ROIC − WACC spread
# ---------------------------------------------------------------------------

def compute_roic_wacc_spread(
    ticker: str,
    conn,
    params: dict,
    as_of_date: str,
    region: str,
) -> tuple[float, float, float]:
    """
    Returns (spread, normalized_score, confidence).
    spread = roic - wacc  (decimal, e.g. 0.05 = 5pp above cost of capital)
    """
    from src.data.db import get_latest_fundamentals
    from src.data.macro import get_risk_free_rate

    df = get_latest_fundamentals(conn, ticker, n_years=1)
    if df.empty or df["roic"].isna().all():
        return float("nan"), 0.5, 0.0

    roic = df["roic"].dropna().iloc[0]
    if pd.isna(roic):
        return float("nan"), 0.5, 0.0

    rf = get_risk_free_rate(conn, region, as_of_date, params)
    erp = params["return_estimation"]["equity_risk_premium"]
    wacc = rf + erp  # simplified WACC (no beta, no leverage adjustment for MVP)

    spread = roic - wacc

    q_params = params["signals"]["quality"]
    score = _clamp_normalize(spread,
                             q_params["roic_wacc_spread_min"],
                             q_params["roic_wacc_spread_max"])

    # Confidence: 1.0 for annual, 0.7 if only quarterly available
    source = df["source"].iloc[0] if not df.empty else "unknown"
    confidence = 0.7 if source == "yfinance" else 1.0

    return spread, score, confidence


# ---------------------------------------------------------------------------
# Sub-score 2: Gross margin SNR
# ---------------------------------------------------------------------------

def compute_margin_snr(
    ticker: str,
    conn,
    params: dict,
) -> tuple[float, float, float]:
    """
    Returns (snr, normalized_score, confidence).
    SNR = mean(gross_margin) / std(gross_margin) across available years.
    Higher → more stable margins → stronger moat.
    """
    from src.data.db import get_latest_fundamentals

    lookback = params["signals"]["quality"].get("lookback_years", 5)
    df = get_latest_fundamentals(conn, ticker, n_years=lookback)

    if df.empty:
        return float("nan"), 0.5, 0.0

    margins = df["gross_margin"].dropna()
    n = len(margins)
    if n < 2:
        return float("nan"), 0.5, max(0.0, n / 5.0)

    mean_gm = margins.mean()
    std_gm  = margins.std()

    if std_gm == 0 or pd.isna(std_gm):
        snr = 10.0 if mean_gm > 0 else 0.0  # perfectly stable → cap at max
    else:
        snr = abs(mean_gm) / std_gm

    q_params = params["signals"]["quality"]
    score = _clamp_normalize(snr,
                             q_params["margin_snr_min"],
                             q_params["margin_snr_max"])

    confidence = min(1.0, n / 5.0)
    return snr, score, confidence


# ---------------------------------------------------------------------------
# Sub-score 3: Inflation convexity (pricing power)
# ---------------------------------------------------------------------------

def compute_inflation_convexity(
    ticker: str,
    conn,
    params: dict,
    as_of_date: str,
    region: str,
    lookback_years: int = 3,
) -> tuple[float, float, float]:
    """
    Returns (convexity, normalized_score, confidence).
    convexity = revenue_CAGR - avg_PPI_YoY  (both decimal)
    Positive → company is pricing above input cost inflation.
    """
    from src.data.db import get_latest_fundamentals
    from src.data.macro import get_inflation_yoy

    df = get_latest_fundamentals(conn, ticker, n_years=lookback_years + 1)
    if df.empty:
        return float("nan"), 0.5, 0.0

    revenues = df[["fiscal_year", "revenue"]].dropna().sort_values("fiscal_year")
    n_rev = len(revenues)
    if n_rev < 2:
        return float("nan"), 0.5, max(0.0, n_rev / lookback_years)

    # Revenue CAGR (revenues sorted ascending by fiscal_year)
    r_start = revenues["revenue"].iloc[0]    # oldest
    r_end   = revenues["revenue"].iloc[-1]   # newest
    years   = revenues["fiscal_year"].iloc[-1] - revenues["fiscal_year"].iloc[0]
    if years <= 0 or r_start <= 0:
        return float("nan"), 0.5, 0.0

    rev_cagr = (r_end / r_start) ** (1.0 / years) - 1.0

    # Average PPI YoY over same period
    ppi_series = get_inflation_yoy(conn, region, "ppi", as_of_date, params,
                                   lookback_months=lookback_years * 12)
    if ppi_series.empty:
        avg_ppi = 0.04  # fallback: assume 4% PPI if data unavailable
        ppi_confidence = 0.5
    else:
        avg_ppi = ppi_series.mean() / 100.0  # convert from % to decimal
        ppi_confidence = 1.0

    convexity = rev_cagr - avg_ppi

    q_params = params["signals"]["quality"]
    score = _clamp_normalize(convexity,
                             q_params["convexity_min"],
                             q_params["convexity_max"])

    confidence = min(1.0, n_rev / lookback_years) * ppi_confidence
    return convexity, score, confidence


# ---------------------------------------------------------------------------
# Composite quality score
# ---------------------------------------------------------------------------

def compute_quality_score(
    ticker: str,
    conn,
    params: dict,
    as_of_date: str,
    region: str,
) -> dict:
    """
    Combine three sub-scores into composite quality_score.
    Equal weighting (1/3 each). Confidence-weighted.
    """
    spread, roic_score, roic_conf = compute_roic_wacc_spread(
        ticker, conn, params, as_of_date, region
    )
    snr, margin_score, margin_conf = compute_margin_snr(ticker, conn, params)
    convexity, conv_score, conv_conf = compute_inflation_convexity(
        ticker, conn, params, as_of_date, region
    )

    # Confidence-weighted average of sub-scores
    total_conf = roic_conf + margin_conf + conv_conf
    if total_conf == 0:
        quality_score = 0.5
        quality_conf  = 0.0
    else:
        quality_score = (
            roic_score * roic_conf +
            margin_score * margin_conf +
            conv_score * conv_conf
        ) / total_conf
        quality_conf = total_conf / 3.0  # average confidence

    return {
        "ticker":               ticker,
        "quality_score":        round(quality_score, 4),
        "quality_confidence":   round(quality_conf, 4),
        "roic_wacc_spread":     round(spread, 4) if not pd.isna(spread) else None,
        "margin_snr":           round(snr, 4) if not pd.isna(snr) else None,
        "inflation_convexity":  round(convexity, 4) if not pd.isna(convexity) else None,
    }


def batch_quality_scores(
    universe_df: pd.DataFrame,
    conn,
    params: dict,
    as_of_date: str,
) -> pd.DataFrame:
    """Compute quality scores for all stocks. Returns DataFrame."""
    rows = []
    for _, stock in universe_df.iterrows():
        result = compute_quality_score(
            stock["ticker"], conn, params, as_of_date, stock["region"]
        )
        rows.append(result)
    return pd.DataFrame(rows)
