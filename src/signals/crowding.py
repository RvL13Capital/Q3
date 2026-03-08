"""
Signal 3: Crowding Score (X_C) — Critical Slowing Down Index

EARKE eq 6:
  X_C,t = clamp(ω₁·Δρ₁(t) + ω₂·Δ(λ_max / Σλ) [+ ω₃·trends_score])

Two required sub-components:
  1. Δρ₁ — change in lag-1 autocorrelation of stock returns.
     Rising autocorrelation = returns becoming mean-avoiding = CSD signal.
  2. Δ(λ_max/Σλ) — change in absorption ratio from PCA of universe return
     correlation matrix. Rising = stocks moving in sync = systemic crowding.

Optional third component (when Google Trends data is present in DB):
  3. trends_score — normalised ratio of recent-to-baseline search interest.
     Rising public attention above its own 90-day average = crowding proxy.
     Weight ω₃ (params.signals.crowding.csd_omega3, default 0.20) is only
     applied when trends data is found; weights renormalise to 1.0 otherwise.

High X_C = crowded / pre-crash regime = BAD for new entries.
The composite formula inverts this: (1 - X_C) contributes to the final score.

Absorption ratio is computed once per batch run (universe-level signal)
and shared across all per-stock calls for efficiency.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-score 1: Lag-1 autocorrelation delta (per-stock)
# ---------------------------------------------------------------------------

def compute_lag1_autocorr(
    ticker: str,
    conn,
    params: dict,
    as_of_date: str,
) -> tuple[float, float, float, float]:
    """
    Returns (rho_current, rho_baseline, delta, confidence).

    delta = rho_current − rho_baseline.
    Positive delta means autocorrelation is rising (CSD warning signal).
    """
    from src.data.db import get_prices
    from src.data.prices import compute_log_returns

    window   = params["signals"]["crowding"]["autocorr_window"]     # e.g. 30
    baseline = params["signals"]["crowding"]["autocorr_baseline"]   # e.g. 120
    total    = baseline + window + 30

    start = (datetime.strptime(as_of_date, "%Y-%m-%d")
             - timedelta(days=total)).strftime("%Y-%m-%d")

    price_df = get_prices(conn, [ticker], start, as_of_date)
    if price_df.empty or ticker not in price_df.columns:
        return 0.0, 0.0, 0.0, 0.0

    returns = compute_log_returns(price_df[[ticker]]).dropna()
    n = len(returns)
    if n < 20:
        return 0.0, 0.0, 0.0, n / 20.0

    ret = returns[ticker]

    current_slice = ret.tail(window)
    rho_current = float(current_slice.autocorr(lag=1)) if len(current_slice) >= 10 else 0.0
    if pd.isna(rho_current):
        rho_current = 0.0

    baseline_slice = ret.iloc[-(baseline + window):-window]
    rho_baseline = float(baseline_slice.autocorr(lag=1)) if len(baseline_slice) >= 10 else 0.0
    if pd.isna(rho_baseline):
        rho_baseline = 0.0

    delta      = rho_current - rho_baseline
    confidence = min(1.0, n / baseline)
    return rho_current, rho_baseline, round(delta, 4), round(confidence, 4)


# ---------------------------------------------------------------------------
# Sub-score 2: Absorption ratio delta (universe-level, computed once per batch)
# ---------------------------------------------------------------------------

def compute_absorption_ratio(
    universe_tickers: list[str],
    conn,
    params: dict,
    as_of_date: str,
) -> tuple[float, float, float, float]:
    """
    Returns (ratio_current, ratio_baseline, delta, confidence).

    Absorption ratio = λ_max / Σλ  where λ are positive eigenvalues of the
    return correlation matrix.  Rising ratio = stocks co-moving = systemic crowding.
    """
    from src.data.db import get_prices
    from src.data.prices import compute_log_returns

    window = params["signals"]["crowding"]["absorption_window"]   # e.g. 60
    total  = window * 2 + 30

    start = (datetime.strptime(as_of_date, "%Y-%m-%d")
             - timedelta(days=total)).strftime("%Y-%m-%d")

    price_df = get_prices(conn, universe_tickers, start, as_of_date)
    if price_df.empty:
        return 0.5, 0.5, 0.0, 0.0

    returns = compute_log_returns(price_df).dropna(how="all")

    def _absorption(ret_slice: pd.DataFrame) -> Optional[float]:
        clean = ret_slice.dropna(axis=1, how="any")
        if clean.shape[1] < 3 or clean.shape[0] < 10:
            return None
        corr = clean.corr().values
        try:
            eigvals = np.linalg.eigvalsh(corr)
            eigvals = eigvals[eigvals > 0]
            if len(eigvals) == 0:
                return None
            return float(eigvals.max() / eigvals.sum())
        except np.linalg.LinAlgError:
            return None

    ratio_current  = _absorption(returns.tail(window))
    ratio_baseline = _absorption(returns.iloc[-(window * 2):-window])

    if ratio_current is None:
        return 0.5, 0.5, 0.0, 0.0
    if ratio_baseline is None:
        return ratio_current, ratio_current, 0.0, 0.5

    delta      = ratio_current - ratio_baseline
    n_tickers  = returns.tail(window).dropna(axis=1, how="any").shape[1]
    confidence = min(1.0, n_tickers / max(1, len(universe_tickers) * 0.5))
    return (round(ratio_current, 4), round(ratio_baseline, 4),
            round(delta, 4), round(confidence, 4))


# ---------------------------------------------------------------------------
# Google Trends crowding sub-score (optional third component)
# ---------------------------------------------------------------------------

def _get_trends_crowding_score(
    conn,
    keyword: str,
    as_of_date: str,
    params: dict,
) -> tuple[float, float]:
    """
    Compute a crowding proxy from Google Trends search interest.

    Returns (trend_score, confidence) both in [0, 1].

    trend_score > 0.5 means recent search interest is above its own 90-day
    baseline, signalling growing public attention → potential crowding.

    Normalization: maps the ratio (recent_30d / baseline_90d) from [0.5, 2.0]
    linearly onto [0, 1], so a 2× spike yields trend_score = 1.0.
    """
    from src.data.db import get_trends

    w = params["signals"]["crowding"]
    recent_days   = int(w.get("trends_window_recent",   30))
    baseline_days = int(w.get("trends_window_baseline", 90))

    end   = as_of_date
    start = (datetime.strptime(as_of_date, "%Y-%m-%d")
             - timedelta(days=baseline_days)).strftime("%Y-%m-%d")

    try:
        series = get_trends(conn, keyword, start, end)
    except Exception:
        return 0.0, 0.0

    if series.empty or len(series) < 7:
        return 0.0, 0.0

    baseline_score = float(series.mean())
    if baseline_score < 1.0:
        return 0.0, 0.0

    recent_score = float(series.tail(recent_days).mean())
    # ratio in [0.5, 2.0] → trend_score in [0, 1]
    ratio       = recent_score / baseline_score
    trend_score = max(0.0, min(1.0, (ratio - 0.5) / 1.5))
    confidence  = min(1.0, len(series) / baseline_days)
    return round(trend_score, 4), round(confidence, 4)


# ---------------------------------------------------------------------------
# Composite crowding score (CSD formula, eq 6)
# ---------------------------------------------------------------------------

def compute_crowding_score(
    ticker: str,
    conn,
    params: dict,
    as_of_date: str,
    absorption_delta: float = 0.0,
    absorption_conf: float = 0.0,
    trends_keyword: Optional[str] = None,
) -> dict:
    """
    X_C,t = clamp(ω₁·Δρ₁ + ω₂·Δ(λ_max/Σλ) [+ ω₃·trends_score])

    absorption_delta / absorption_conf are pre-computed universe-level values
    passed in from batch_crowding_scores to avoid redundant fetches.

    trends_keyword: if provided and DB has Trends data, adds a third component
    weighted by csd_omega3 (params).  All weights are renormalised to sum 1.0.
    """
    w      = params["signals"]["crowding"]
    omega1 = w.get("csd_omega1", 0.5)
    omega2 = w.get("csd_omega2", 0.5)
    omega3 = w.get("csd_omega3", 0.20)

    _, _, autocorr_delta, autocorr_conf = compute_lag1_autocorr(
        ticker, conn, params, as_of_date
    )

    # Normalize deltas to [0, 1]:
    # Δρ₁ typical range [-0.30, +0.30]: rising autocorr → score near 1.0
    autocorr_score   = max(0.0, min(1.0, (autocorr_delta + 0.30) / 0.60))
    # Δ(λ_max/Σλ) typical range [-0.10, +0.10]: rising absorption → score near 1.0
    absorption_score = max(0.0, min(1.0, (absorption_delta + 0.10) / 0.20))

    # Optional Google Trends third component
    trend_score, trend_conf = 0.0, 0.0
    if trends_keyword and conn is not None:
        try:
            trend_score, trend_conf = _get_trends_crowding_score(
                conn, trends_keyword, as_of_date, params
            )
        except Exception as e:
            logger.debug(f"Trends score unavailable for {ticker}: {e}")

    if trend_conf > 0:
        # Three-component formula: renormalise weights to sum 1.0
        w_sum         = omega1 + omega2 + omega3
        crowding_score = max(0.0, min(1.0, (
            omega1 * autocorr_score
            + omega2 * absorption_score
            + omega3 * trend_score
        ) / w_sum))
        if absorption_conf > 0:
            crowding_conf = (autocorr_conf + absorption_conf + trend_conf) / 3.0
        else:
            crowding_conf = (autocorr_conf * 0.70 + trend_conf) / 2.0
    else:
        # Two-component fallback (original behaviour, omega1+omega2 already = 1.0)
        crowding_score = max(0.0, min(1.0, omega1 * autocorr_score + omega2 * absorption_score))
        if absorption_conf > 0:
            crowding_conf = (autocorr_conf + absorption_conf) / 2.0
        else:
            crowding_conf = autocorr_conf * 0.70  # penalise when no universe-level data

    return {
        "ticker":              ticker,
        "crowding_score":      round(crowding_score, 4),
        "crowding_confidence": round(crowding_conf, 4),
        "autocorr_delta":      round(autocorr_delta, 4),
        "absorption_delta":    round(absorption_delta, 4),
        "trends_score":        round(trend_score, 4),
    }


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------

def batch_crowding_scores(
    universe_df: pd.DataFrame,
    conn,
    params: dict,
    as_of_date: str,
) -> pd.DataFrame:
    """
    Compute CSD crowding scores for all universe stocks.
    Absorption ratio is computed once (universe-level) and shared across stocks.
    """
    all_tickers = universe_df["ticker"].tolist()

    if not all_tickers:
        return pd.DataFrame(columns=[
            "ticker", "crowding_score", "crowding_confidence",
            "autocorr_delta", "absorption_delta", "trends_score",
        ])

    logger.info(f"Computing absorption ratio for {len(all_tickers)} tickers")
    _, _, absorption_delta, absorption_conf = compute_absorption_ratio(
        all_tickers, conn, params, as_of_date
    )
    logger.info(
        f"Absorption ratio delta={absorption_delta:.4f}  conf={absorption_conf:.2f}"
    )

    rows = []
    for _, stock in universe_df.iterrows():
        result = compute_crowding_score(
            stock["ticker"], conn, params, as_of_date,
            absorption_delta=absorption_delta,
            absorption_conf=absorption_conf,
            trends_keyword=stock.get("trends_keyword"),
        )
        rows.append(result)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Google Trends batch fetch (data pipeline — kept for DB population)
# ---------------------------------------------------------------------------

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
    Returns combined DataFrame with columns: keyword, date, score, geo.
    """
    import time
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
