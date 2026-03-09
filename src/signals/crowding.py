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

from src.data.universe import get_sector_etf

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
# Sub-score 3: Sector ETF rolling correlation (optional, per-stock)
# ---------------------------------------------------------------------------

def compute_etf_correlation_score(
    ticker: str,
    etf_ticker: str,
    conn,
    params: dict,
    as_of_date: str,
) -> tuple[float, float]:
    """
    Returns (etf_corr_score, confidence) both in [0, 1].

    Pearson correlation of stock log-returns vs sector ETF log-returns over
    etf_corr_window days, compared to a baseline period.

    Rising co-movement with the sector ETF = herd convergence = crowding signal.
    Score: delta_corr / max_delta, clamped to [0, 1].
    max_delta is the correlation increase considered fully crowded (default 0.30).
    delta <= 0 (correlation falling or stable) → score = 0, no crowding signal.
    """
    from src.data.db import get_prices
    from src.data.prices import compute_log_returns

    w        = params["signals"]["crowding"]
    window   = int(w.get("etf_corr_window", 60))
    baseline = int(w.get("etf_corr_baseline", 120))
    max_delta = float(w.get("etf_corr_max_delta", 0.30))
    total    = baseline + window + 30

    start = (datetime.strptime(as_of_date, "%Y-%m-%d")
             - timedelta(days=total)).strftime("%Y-%m-%d")

    price_df = get_prices(conn, [ticker, etf_ticker], start, as_of_date)
    if (price_df.empty
            or ticker not in price_df.columns
            or etf_ticker not in price_df.columns):
        return 0.0, 0.0

    returns = compute_log_returns(price_df).dropna()
    if len(returns) < 20:
        return 0.0, len(returns) / 20.0

    current_slice  = returns.tail(window)
    baseline_slice = returns.iloc[-(baseline + window):-window]

    def _corr(df: pd.DataFrame) -> float:
        if len(df) < 10:
            return float("nan")
        clean = df[[ticker, etf_ticker]].dropna()
        if len(clean) < 10:
            return float("nan")
        c = clean.corr()
        return float(c.loc[ticker, etf_ticker])

    corr_current  = _corr(current_slice)
    corr_baseline = _corr(baseline_slice)

    if pd.isna(corr_current):
        return 0.0, 0.0

    delta = (corr_current - corr_baseline) if not pd.isna(corr_baseline) else corr_current
    score = max(0.0, min(1.0, delta / max(max_delta, 1e-9)))
    confidence = min(1.0, len(current_slice.dropna()) / window)
    return round(score, 4), round(confidence, 4)


# ---------------------------------------------------------------------------
# Sub-score 4: Short interest (optional, per-stock)
# ---------------------------------------------------------------------------

def compute_short_interest_score(
    ticker: str,
    conn,
    params: dict,
    as_of_date: str,
) -> tuple[float, float]:
    """
    Returns (short_score, confidence) both in [0, 1].

    High short_pct_float signals extreme positioning → crowding risk.
    Score: linear from short_interest_low (score=0) to short_interest_high (score=1).
    """
    w        = params["signals"]["crowding"]
    high_thr = float(w.get("short_interest_high", 0.25))
    low_thr  = float(w.get("short_interest_low", 0.05))
    window   = int(w.get("short_interest_window", 30))

    start = (datetime.strptime(as_of_date, "%Y-%m-%d")
             - timedelta(days=window)).strftime("%Y-%m-%d")

    try:
        df = conn.execute("""
            SELECT short_pct_float FROM short_interest
            WHERE ticker = ? AND date BETWEEN ? AND ?
            ORDER BY date DESC LIMIT 5
        """, [ticker, start, as_of_date]).df()
    except Exception:
        return 0.0, 0.0

    if df.empty or df["short_pct_float"].isna().all():
        return 0.0, 0.0

    pct = float(df["short_pct_float"].dropna().mean())
    if high_thr <= low_thr:
        return 0.0, 0.0

    score = max(0.0, min(1.0, (pct - low_thr) / (high_thr - low_thr)))
    confidence = min(1.0, len(df["short_pct_float"].dropna()) / 3.0)
    return round(score, 4), round(confidence, 4)


# ---------------------------------------------------------------------------
# Google Trends crowding sub-score (optional component)
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
    etf_ticker: Optional[str] = None,
) -> dict:
    """
    X_C,t = clamp(weighted sum of active sub-scores)

    Required (always computed):
      ω₁·Δρ₁            — lag-1 autocorrelation delta
      ω₂·Δ(λ_max/Σλ)   — absorption ratio delta (universe-level, pre-computed)

    Optional (each only applied when data is present; weights renormalised):
      ω₃·trends_score   — Google Trends recent vs baseline
      ω_etf·etf_corr    — rolling Pearson correlation vs sector ETF
      ω_short·short_int — short interest as % of float

    All weights renormalise to sum 1.0 over active components.
    """
    w         = params["signals"]["crowding"]
    omega1    = w.get("csd_omega1", 0.5)
    omega2    = w.get("csd_omega2", 0.5)
    omega3    = w.get("csd_omega3", 0.20)
    omega_etf = w.get("csd_omega_etf", 0.15)
    omega_si  = w.get("csd_omega_short", 0.10)

    _, _, autocorr_delta, autocorr_conf = compute_lag1_autocorr(
        ticker, conn, params, as_of_date
    )

    # Normalise deltas to [0, 1]
    # Δρ₁ typical range [-0.30, +0.30]: rising autocorr → score near 1.0
    autocorr_score   = max(0.0, min(1.0, (autocorr_delta + 0.30) / 0.60))
    # Δ(λ_max/Σλ) typical range [-0.10, +0.10]: rising absorption → score near 1.0
    absorption_score = max(0.0, min(1.0, (absorption_delta + 0.10) / 0.20))

    # Capture optional sub-scores as named variables; add to components only when data present.
    # Avoids fragile weight-value matching when extracting scores from the component list.
    trend_score = 0.0
    etf_score   = 0.0
    short_score = 0.0

    # Build weighted components list — only include optional ones when data exists
    components = [
        (omega1, autocorr_score,   autocorr_conf),
        (omega2, absorption_score, absorption_conf),
    ]

    # Google Trends
    if trends_keyword and conn is not None:
        try:
            ts, tc = _get_trends_crowding_score(conn, trends_keyword, as_of_date, params)
            if tc > 0:
                trend_score = ts
                components.append((omega3, ts, tc))
        except Exception as e:
            logger.debug(f"Trends score unavailable for {ticker}: {e}")

    # Sector ETF correlation
    if etf_ticker and conn is not None:
        try:
            es, ec = compute_etf_correlation_score(ticker, etf_ticker, conn, params, as_of_date)
            if ec > 0:
                etf_score = es
                components.append((omega_etf, es, ec))
        except Exception as e:
            logger.debug(f"ETF correlation unavailable for {ticker}: {e}")

    # Short interest
    if conn is not None:
        try:
            ss, sc = compute_short_interest_score(ticker, conn, params, as_of_date)
            if sc > 0:
                short_score = ss
                components.append((omega_si, ss, sc))
        except Exception as e:
            logger.debug(f"Short interest unavailable for {ticker}: {e}")

    # Renormalise weights over active components
    total_w = sum(c[0] for c in components)
    if total_w <= 0:
        crowding_score = 0.0
        crowding_conf  = 0.0
    else:
        crowding_score = max(0.0, min(1.0,
            sum(c[0] * c[1] for c in components) / total_w
        ))
        crowding_conf = sum(c[0] * c[2] for c in components) / total_w

    return {
        "ticker":              ticker,
        "crowding_score":      round(crowding_score, 4),
        "crowding_confidence": round(crowding_conf, 4),
        "autocorr_delta":      round(autocorr_delta, 4),
        "absorption_delta":    round(absorption_delta, 4),
        "trends_score":        round(trend_score, 4),
        "etf_corr_score":      round(etf_score, 4),
        "short_interest_score": round(short_score, 4),
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
        etf_ticker = get_sector_etf(
            stock.get("primary_bucket", ""),
            stock.get("region", ""),
            params,
        )
        result = compute_crowding_score(
            stock["ticker"], conn, params, as_of_date,
            absorption_delta=absorption_delta,
            absorption_conf=absorption_conf,
            trends_keyword=stock.get("trends_keyword"),
            etf_ticker=etf_ticker,
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
    consecutive_429 = 0
    skipped_batches = 0
    _MAX_CONSECUTIVE_429 = 2  # abort session after this many consecutive rate-limits

    for i in range(0, len(keywords), batch_size):
        batch = keywords[i: i + batch_size]
        try:
            pt = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
            pt.build_payload(batch, timeframe=timeframe, geo=geo)
            df = pt.interest_over_time()
            consecutive_429 = 0  # reset on success
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
            if "429" in str(e):
                consecutive_429 += 1
                skipped_batches += 1
                if consecutive_429 >= _MAX_CONSECUTIVE_429:
                    remaining = (len(keywords) - i - batch_size + batch_size - 1) // batch_size
                    logger.warning(
                        f"Google Trends rate-limited (429) — aborting after "
                        f"{consecutive_429} consecutive failures. "
                        f"{skipped_batches + remaining} batches skipped total."
                    )
                    break
            else:
                logger.warning(f"Google Trends batch {batch} failed: {e}")

        time.sleep(delay_secs)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)
