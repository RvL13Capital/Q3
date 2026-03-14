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
# Beta computation for CAPM-based WACC
# ---------------------------------------------------------------------------

def _compute_beta(
    conn,
    ticker: str,
    market_index: str,
    as_of_date: str,
    min_obs: int = 120,
) -> tuple[float, bool]:
    """
    Compute CAPM beta via OLS regression of stock log returns vs market index log returns.

    Returns (beta, is_valid).
    Falls back to (1.0, False) if insufficient observations.
    """
    from src.data.db import get_prices
    from datetime import datetime, timedelta

    # Compute start_date ~3 years before as_of_date
    end_dt = datetime.strptime(as_of_date, "%Y-%m-%d").date() if isinstance(as_of_date, str) else as_of_date
    start_dt = end_dt - timedelta(days=756)
    start_str = start_dt.isoformat()
    end_str = end_dt.isoformat()

    # get_prices returns wide-format: index=date, columns=tickers
    price_df = get_prices(conn, [ticker, market_index], start_str, end_str)
    if price_df.empty or ticker not in price_df.columns or market_index not in price_df.columns:
        return 1.0, False

    # Drop rows where either is NaN
    merged = price_df[[ticker, market_index]].dropna()
    if len(merged) < min_obs:
        return 1.0, False

    merged = merged.sort_index()
    stock_ret = np.log(merged[ticker] / merged[ticker].shift(1)).iloc[1:]
    market_ret = np.log(merged[market_index] / merged[market_index].shift(1)).iloc[1:]

    # Filter out any NaN/inf
    mask = np.isfinite(stock_ret.values) & np.isfinite(market_ret.values)
    if mask.sum() < min_obs:
        return 1.0, False

    sr = stock_ret.values[mask]
    mr = market_ret.values[mask]

    # OLS: beta = cov(rs, rm) / var(rm)
    var_m = np.var(mr, ddof=1)
    if var_m < 1e-12:
        return 1.0, False

    beta = float(np.cov(sr, mr, ddof=1)[0, 1] / var_m)

    # Sanity bounds: beta outside [-1, 5] is likely data error
    if beta < -1.0 or beta > 5.0:
        logger.warning("Beta for %s = %.2f (out of bounds) — using default", ticker, beta)
        return 1.0, False

    return beta, True


# ---------------------------------------------------------------------------
# Sub-score 1: ROIC − WACC spread
# ---------------------------------------------------------------------------

def compute_roic_wacc_spread(
    ticker: str,
    conn,
    params: dict,
    as_of_date: str,
    region: str,
    rf_cache: dict | None = None,
) -> tuple[float, float]:
    """
    Returns (spread, confidence).
    spread = roic - wacc  (decimal, e.g. 0.10 for 10pp moat above cost of capital).
    Spread is returned raw — caller applies max(0, spread) and normalises by spread_max.

    rf_cache: optional {region: rf_rate} dict to avoid repeated DB lookups in batch mode.
    """
    from src.data.db import get_latest_fundamentals
    from src.data.macro import get_risk_free_rate

    df = get_latest_fundamentals(conn, ticker, n_years=1, as_of_date=as_of_date)
    if df.empty or df["roic"].isna().all():
        return float("nan"), 0.0

    roic = df["roic"].dropna().iloc[0]
    if pd.isna(roic):
        return float("nan"), 0.0

    rf = rf_cache[region] if (rf_cache and region in rf_cache) else \
        get_risk_free_rate(conn, region, as_of_date, params, as_of_tk=as_of_date)
    erp = params["return_estimation"]["equity_risk_premium"]

    q_params = params.get("signals", {}).get("quality", {})
    use_beta = q_params.get("wacc_use_beta", False)
    beta_valid = True  # default: no penalty

    if use_beta:
        market_indices = params.get("market_indices", {})
        market_index = market_indices.get(region, market_indices.get("US", "SPY"))
        beta_default = float(q_params.get("wacc_beta_default", 1.0))
        beta_min_obs = int(q_params.get("wacc_beta_min_obs", 120))

        beta, beta_valid = _compute_beta(conn, ticker, market_index, as_of_date, beta_min_obs)
        if not beta_valid:
            beta = beta_default

        cost_of_equity = rf + beta * erp

        total_debt_val = 0.0
        total_equity_val = 0.0
        if "total_debt" in df.columns and not df["total_debt"].isna().all():
            total_debt_val = float(df["total_debt"].dropna().iloc[0])
        if "total_equity" in df.columns and not df["total_equity"].isna().all():
            total_equity_val = float(df["total_equity"].dropna().iloc[0])

        if total_equity_val > 0 and total_debt_val > 0:
            total_capital = total_debt_val + total_equity_val
            w_e = total_equity_val / total_capital
            w_d = total_debt_val / total_capital
            cost_of_debt = float(q_params.get("wacc_cost_of_debt", 0.05))
            tax_shield = float(q_params.get("wacc_tax_shield", 0.21))
            wacc = w_e * cost_of_equity + w_d * cost_of_debt * (1 - tax_shield)
        else:
            wacc = cost_of_equity
    else:
        wacc = rf + erp  # original simplified WACC

    spread = roic - wacc

    # Confidence: 1.0 for annual, 0.7 if only yfinance available
    source = df["source"].iloc[0] if not df.empty else "unknown"
    confidence = 0.7 if source == "yfinance" else 1.0

    # Confidence penalty when using fallback beta
    if use_beta and not beta_valid:
        confidence = max(0.0, confidence - 0.15)

    return spread, confidence


# ---------------------------------------------------------------------------
# Sub-score 2: Gross margin SNR
# ---------------------------------------------------------------------------

def compute_margin_snr(
    ticker: str,
    conn,
    params: dict,
    as_of_date: str | None = None,
) -> tuple[float, float, float]:
    """
    Returns (snr, normalized_score, confidence).
    SNR = mean(gross_margin) / std(gross_margin) across available years.
    Higher → more stable margins → stronger moat.
    Uses quarterly data as supplement when fewer than 3 annual rows exist.
    """
    from src.data.db import get_margin_history

    lookback = params["signals"]["quality"].get("lookback_years", 5)
    df = get_margin_history(conn, ticker, n_years=lookback, as_of_date=as_of_date)

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

    df = get_margin_history(conn, ticker, n_years=lookback_years + 1, as_of_date=as_of_date)
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
                                   lookback_months=(lookback_years + 1) * 12,
                                   as_of_tk=as_of_date)
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

    min_points = params.get("signals", {}).get("quality", {}).get("convexity_min_points", lookback_years)
    confidence = min(1.0, int(mask.sum()) / max(min_points, lookback_years))
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
    rf_cache: dict | None = None,
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
        ticker, conn, params, as_of_date, region, rf_cache=rf_cache
    )
    snr, margin_score, margin_conf = compute_margin_snr(ticker, conn, params, as_of_date=as_of_date)
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
        # Confidence-based attenuation: noisy regression → less amplification
        conv_factor = 1.0 + (conv_factor - 1.0) * conv_conf
        # Hard cap from params (default: no cap for backward compat)
        convexity_cap = float(q_params.get("convexity_cap", float("inf")))
        conv_factor = min(conv_factor, convexity_cap)

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
    from src.data.macro import get_risk_free_rate

    # Pre-fetch rf rates once per unique region instead of once per stock.
    regions = universe_df["region"].unique().tolist()
    rf_cache = {r: get_risk_free_rate(conn, r, as_of_date, params) for r in regions}

    rows = []
    for _, stock in universe_df.iterrows():
        result = compute_quality_score(
            stock["ticker"], conn, params, as_of_date, stock["region"],
            rf_cache=rf_cache,
        )
        rows.append(result)
    return pd.DataFrame(rows)
