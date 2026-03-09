"""
Signal 2: Quality / Moat Score (X_P) — EARKE eq 5

  X_P,i,t = max(0, E[ROIC − WACC]) × (1 / σ(GM)) × exp(γ · max(0, ∂GM/∂PPI))

Three sub-components multiplied per spec:
  1. ROIC − WACC spread     raw decimal (e.g. 0.10 for 10pp moat).
                            Hard gate: spread ≤ 0 → X_P = 0.
  2. Gross margin SNR        (1/σ(GM) normalised to [0,1] against empirical range)
  3. Inflation convexity     exp(γ · max(0, ∂GM/∂PPI)) — exponential multiplier ≥ 1.
                            γ from params.signals.quality.gamma (default 2.0).

Normalisation: X_P = min(1, spread_raw/spread_max × snr_norm × exp_conv).
Dividing by spread_max (20pp reference ceiling) maintains [0,1] bounds while
keeping the raw spread as the primary scale factor (not independently normalised).
"""
import logging
import math
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
) -> tuple[float, float]:
    """
    Returns (spread, confidence).
    spread = roic - wacc  (decimal, e.g. 0.10 for 10pp moat above cost of capital).
    Spread is returned raw — caller applies max(0, spread) and normalises by spread_max.
    """
    from src.data.db import get_latest_fundamentals
    from src.data.macro import get_risk_free_rate

    df = get_latest_fundamentals(conn, ticker, n_years=1)
    if df.empty or df["roic"].isna().all():
        return float("nan"), 0.0

    roic = df["roic"].dropna().iloc[0]
    if pd.isna(roic):
        return float("nan"), 0.0

    rf = get_risk_free_rate(conn, region, as_of_date, params)
    erp = params["return_estimation"]["equity_risk_premium"]
    wacc = rf + erp  # simplified WACC (no beta, no leverage adjustment for MVP)

    spread = roic - wacc

    # Confidence: 1.0 for annual, 0.7 if only yfinance available
    source = df["source"].iloc[0] if not df.empty else "unknown"
    confidence = 0.7 if source == "yfinance" else 1.0

    return spread, confidence


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
    Uses quarterly data as supplement when fewer than 3 annual rows exist.
    """
    from src.data.db import get_margin_history

    lookback = params["signals"]["quality"].get("lookback_years", 5)
    df = get_margin_history(conn, ticker, n_years=lookback)

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
) -> tuple[float, float]:
    """
    Returns (convexity, confidence).

    convexity = ∂GM/∂PPI: OLS slope of annual gross-margin changes vs annual PPI
    (both in decimal). Caller applies exp(γ · max(0, convexity)) per eq 5.

    Positive → margins expand when input costs rise (anti-fragile pricing power).
    Zero     → margins move independently of input cost inflation.
    Negative → margins are compressed by rising input costs.
    """
    from src.data.db import get_margin_history
    from src.data.macro import get_inflation_yoy

    df = get_margin_history(conn, ticker, n_years=lookback_years + 1)
    if df.empty:
        return float("nan"), 0.0

    margins = df[["fiscal_year", "gross_margin"]].dropna().sort_values("fiscal_year")
    n = len(margins)
    if n < 3:  # need ≥2 diffs for a regression
        return float("nan"), max(0.0, n / (lookback_years + 1))

    # Year-over-year changes in gross margin (decimal, e.g. 0.02 for 2pp expansion)
    delta_gm = margins["gross_margin"].diff().dropna().values
    fiscal_years = margins["fiscal_year"].values[1:]

    # Annual average PPI for each fiscal year (from monthly YoY series)
    ppi_series = get_inflation_yoy(conn, region, "ppi", as_of_date, params,
                                   lookback_months=(lookback_years + 1) * 12)
    if ppi_series.empty:
        return float("nan"), 0.0

    ppi_series.index = pd.to_datetime(ppi_series.index)
    annual_ppi = ppi_series.groupby(ppi_series.index.year).mean() / 100.0  # decimal

    # Align annual_ppi to fiscal years
    aligned_ppi = np.array([
        annual_ppi.get(yr, float("nan")) for yr in fiscal_years
    ])

    mask = ~np.isnan(aligned_ppi) & ~np.isnan(delta_gm)
    if mask.sum() < 2:
        return float("nan"), 0.0

    dgm = delta_gm[mask]
    ppi = aligned_ppi[mask]

    # OLS slope: β = cov(Δgm, ppi) / var(ppi)
    ppi_var = float(np.var(ppi, ddof=0))
    if ppi_var < 1e-10:
        convexity = 0.0
    else:
        convexity = float(np.cov(dgm, ppi, ddof=0)[0, 1] / ppi_var)

    if np.isnan(convexity):
        return float("nan"), 0.0

    confidence = min(1.0, int(mask.sum()) / lookback_years)
    return convexity, confidence


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
    X_P per eq 5: max(0, ROIC−WACC) × (1/σ(GM)) × exp(γ · max(0, ∂GM/∂PPI))

    Implementation:
      spread_factor = max(0, spread) / spread_max     — raw spread, primary scale
      snr_factor    = clamp_normalise(snr, min, max)  — margin stability [0,1]
      conv_factor   = exp(γ · max(0, slope))           — convexity multiplier ≥ 1
      quality_score = min(1, spread_factor × snr_factor × conv_factor)

    Hard gate: spread ≤ 0 → quality_score = 0 (no moat, no trade).
    """
    q_params    = params["signals"]["quality"]
    spread_max  = q_params["roic_wacc_spread_max"]   # 0.20
    gamma       = float(q_params.get("gamma", 2.0))

    spread, roic_conf = compute_roic_wacc_spread(
        ticker, conn, params, as_of_date, region
    )
    snr, margin_score, margin_conf = compute_margin_snr(ticker, conn, params)
    convexity, conv_conf = compute_inflation_convexity(
        ticker, conn, params, as_of_date, region
    )

    if roic_conf == 0.0 or pd.isna(spread):
        quality_score = 0.0
        quality_conf  = 0.0
    elif spread <= 0.0:
        # Hard gate: ROIC ≤ WACC → no moat
        quality_score = 0.0
        quality_conf  = (roic_conf + margin_conf + conv_conf) / 3.0
    else:
        # Sub-component 1: raw spread normalised by ceiling (not clamped to 1 yet)
        spread_factor = spread / spread_max  # e.g. 0.10/0.20 = 0.50

        # Sub-component 2: margin SNR [0, 1]
        snr_factor = margin_score

        # Sub-component 3: exp(γ · max(0, slope)) — convexity amplifier
        safe_slope  = max(0.0, convexity) if not pd.isna(convexity) else 0.0
        conv_factor = math.exp(gamma * safe_slope)

        quality_score = min(1.0, spread_factor * snr_factor * conv_factor)
        quality_conf  = (roic_conf + margin_conf + conv_conf) / 3.0

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
