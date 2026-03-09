"""
FX & Capital Flow Analysis — Module II extension.

Computes three layers of capital flow information:

  1. FX Momentum       — USD strength index vs EUR, GBP, CHF, CAD, JPY.
                         Identifies broad risk-on/risk-off regime and
                         FX headwinds/tailwinds for each portfolio region.

  2. Sector Flows      — Relative performance of sector ETFs vs broad market,
                         computed in both local currency and USD.
                         The gap (FX contribution) measures cross-border capital rotation.

  3. Country Flows     — Regional equity flow scores (EU / US / CA) in EUR-base,
                         derived from broad index relative strength.

  4. Implications      — Narrative flags consumed by the weekly report and
                         potentially by Module V (Kelly scaling) and X_C crowding signal.

All computations are purely analytical — no DB writes from this module.
If price or FX data is unavailable, each sub-section degrades gracefully to empty.

Output of compute_capital_flows():
  {
    "fx_momentum":   pd.DataFrame,   # pair × {return_1m, return_3m, return_12m, usd_signal}
    "usd_index":     float,          # [-1, +1]  positive = USD strengthening
    "usd_trend":     str,            # 'strengthening' | 'weakening' | 'neutral'
    "sector_flows":  pd.DataFrame,   # region/bucket × flow metrics
    "country_flows": pd.DataFrame,   # region × {flow_score, flow_direction, fx_adjusted_ret}
    "implications":  list[str],      # plain-English flags
    "as_of_date":    str,
  }
"""
import logging
from datetime import date, timedelta, datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# USD strength sign convention
# ---------------------------------------------------------------------------
# For each yfinance pair, the sign to apply to its log return so that
# +1 always means "USD is gaining" and -1 means "USD is losing".
#
#   EURUSD=X rising → 1 EUR buys more USD → USD is WEAKENING  → sign = -1
#   GBPUSD=X rising → 1 GBP buys more USD → USD is WEAKENING  → sign = -1
#   USDCHF=X rising → 1 USD buys more CHF → USD is STRENGTHENING → sign = +1
#   USDCAD=X rising → 1 USD buys more CAD → USD is STRENGTHENING → sign = +1
#   USDJPY=X rising → 1 USD buys more JPY → USD is STRENGTHENING → sign = +1
#     (JPY also used as risk-off proxy — see implication logic below)
_USD_STRENGTH_SIGN: dict[str, float] = {
    "EURUSD=X": -1.0,
    "GBPUSD=X": -1.0,
    "USDCHF=X": +1.0,
    "USDCAD=X": +1.0,
    "USDJPY=X": +1.0,
}

# FX pair needed to convert each region's return to EUR
# EU stocks are already EUR — no FX adjustment needed.
# US  return_EUR = return_USD - return_EURUSD  (i.e. USD weakening hurts USD assets in EUR)
# CA  return_EUR = return_CAD + return_CADUSD_in_EUR
#               = return_CAD - return_USDCAD + return_EURUSD... simplified below.
_REGION_TO_EUR_PAIR: dict[str, Optional[str]] = {
    "EU": None,
    "US": "EURUSD=X",   # subtract EURUSD log-return from USD return to get EUR return
    "CA": "USDCAD=X",   # subtract USDCAD log-return and add EURUSD log-return
}

# Safe-haven pairs: when these strengthen (log return negative for USD-strength)
# it signals risk-off / systemic stress — reinforces X_C
_SAFE_HAVEN_PAIRS = {"USDJPY=X", "USDCHF=X"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_returns(prices: pd.Series, periods: int) -> Optional[float]:
    """Log return over trailing `periods` business days. None if insufficient data."""
    prices = prices.dropna()
    if len(prices) < periods + 1:
        return None
    return float(np.log(prices.iloc[-1] / prices.iloc[-1 - periods]))


def _safe_log_ret(s: pd.Series, n: int) -> Optional[float]:
    """Return n-period log return or None."""
    try:
        return _log_returns(s, n)
    except Exception:
        return None


def _annualise(ret: Optional[float], periods: int, trading_days: int = 252) -> Optional[float]:
    if ret is None:
        return None
    return ret * (trading_days / periods)


# ---------------------------------------------------------------------------
# 1 — FX Momentum
# ---------------------------------------------------------------------------

def compute_fx_momentum(
    conn,
    as_of_date: str,
    params: dict,
) -> tuple[pd.DataFrame, float, str]:
    """
    Compute 1M / 3M / 12M log returns for each FX pair and a composite
    USD strength index.

    Returns:
      (fx_momentum_df, usd_index, usd_trend)
      fx_momentum_df columns: pair, base, quote, return_1m, return_3m, return_12m,
                               usd_signal_1m, usd_signal_3m
    """
    from src.data.db import get_fx_rates
    from src.data.fx import FX_PAIRS

    lookback = 400  # calendar days — enough for 12M + buffer
    start = (datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=lookback)).strftime("%Y-%m-%d")

    pairs = list(FX_PAIRS.keys())
    wide = get_fx_rates(conn, pairs, start, as_of_date)

    if wide.empty:
        return pd.DataFrame(), 0.0, "neutral"

    # Business-day counts
    _bd = {1: 21, 3: 63, 12: 252}

    rows = []
    usd_signals_1m: list[float] = []
    usd_signals_3m: list[float] = []

    for pair, (base, quote) in FX_PAIRS.items():
        if pair not in wide.columns:
            continue
        s = wide[pair].dropna()
        r1  = _safe_log_ret(s, _bd[1])
        r3  = _safe_log_ret(s, _bd[3])
        r12 = _safe_log_ret(s, _bd[12])

        sign = _USD_STRENGTH_SIGN.get(pair, 0.0)
        us1  = sign * r1  if r1  is not None else None
        us3  = sign * r3  if r3  is not None else None

        if us1 is not None:
            usd_signals_1m.append(us1)
        if us3 is not None:
            usd_signals_3m.append(us3)

        rows.append({
            "pair":          pair,
            "base":          base,
            "quote":         quote,
            "return_1m":     r1,
            "return_3m":     r3,
            "return_12m":    r12,
            "usd_signal_1m": us1,
            "usd_signal_3m": us3,
        })

    if not rows:
        return pd.DataFrame(), 0.0, "neutral"

    df = pd.DataFrame(rows)

    # USD index: simple average of 1M and 3M signals (both equally weighted)
    all_signals = [s for s in (usd_signals_1m + usd_signals_3m) if s is not None]
    usd_index = float(np.mean(all_signals)) if all_signals else 0.0

    # Clamp to [-1, +1] using a ±5% annualised threshold
    _scale = 0.05 / 252  # 5% annualised per day
    usd_index_norm = float(np.clip(usd_index / (_scale * 21), -1.0, 1.0))

    neutral_band = params.get("fx", {}).get("usd_neutral_band", 0.15)
    if usd_index_norm > neutral_band:
        usd_trend = "strengthening"
    elif usd_index_norm < -neutral_band:
        usd_trend = "weakening"
    else:
        usd_trend = "neutral"

    return df, usd_index_norm, usd_trend


# ---------------------------------------------------------------------------
# 2 — Sector Capital Flows
# ---------------------------------------------------------------------------

def compute_sector_flows(
    conn,
    universe_df: pd.DataFrame,
    params: dict,
    as_of_date: str,
) -> pd.DataFrame:
    """
    For each (region, bucket) ETF pair: compute 4W and 13W returns in
    local currency and USD-equivalent, plus relative strength vs broad market.

    FX contribution = USD return − local return.
    Positive FX contribution → tailwind (foreign capital flows into that region).
    Negative               → headwind (capital leaving that region).

    Returns DataFrame with columns:
      region, bucket, etf_ticker, local_ret_4w, usd_ret_4w, fx_contrib_4w,
      local_ret_13w, usd_ret_13w, fx_contrib_13w,
      vs_market_4w, vs_market_13w, flow_score
    """
    from src.data.db import get_prices, get_fx_rates

    sector_etf_cfg = params.get("sector_etfs", {})
    market_idx_cfg = params.get("market_indices", {})

    lookback_days = 400
    start = (datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    # Collect ETF tickers
    etf_tickers: set[str] = set()
    for etf in sector_etf_cfg.values():
        etf_tickers.add(etf)
    for idx in market_idx_cfg.values():
        etf_tickers.add(idx)

    if not etf_tickers:
        return pd.DataFrame()

    prices_wide = get_prices(conn, list(etf_tickers), start, as_of_date)
    if prices_wide.empty:
        return pd.DataFrame()

    # FX rates for USD↔ other currencies
    fx_pairs = ["EURUSD=X", "USDCAD=X"]
    fx_wide = get_fx_rates(conn, fx_pairs, start, as_of_date)

    _bd4  = 21   # ~4 weeks
    _bd13 = 63   # ~13 weeks

    def _px_ret(ticker: str, n: int) -> Optional[float]:
        if ticker not in prices_wide.columns:
            return None
        return _safe_log_ret(prices_wide[ticker].dropna(), n)

    def _fx_adj(region: str, n: int) -> float:
        """Log-return of FX move that transforms local return to EUR return."""
        if region == "EU" or fx_wide.empty:
            return 0.0
        if region == "US":
            pair = "EURUSD=X"
            if pair not in fx_wide.columns:
                return 0.0
            # EUR/USD rising means EUR strengthens → US assets cheaper in EUR → negative FX effect for EUR investor
            r = _safe_log_ret(fx_wide[pair].dropna(), n)
            return -float(r) if r is not None else 0.0   # subtract EURUSD return
        if region == "CA":
            # CA local return is in CAD; convert CAD→EUR:
            # FX adjustment ≈ -(USDCAD return) - (EURUSD return) … simplified:
            # If USDCAD rises (USD stronger vs CAD), CAD weaker → CA assets cheaper in EUR
            usdcad = "USDCAD=X"
            eurusd = "EURUSD=X"
            r_usdcad = _safe_log_ret(fx_wide[usdcad].dropna(), n) if usdcad in fx_wide.columns else None
            r_eurusd = _safe_log_ret(fx_wide[eurusd].dropna(), n) if eurusd in fx_wide.columns else None
            adj = 0.0
            if r_usdcad is not None:
                adj -= r_usdcad  # CAD weakening hurts
            if r_eurusd is not None:
                adj -= r_eurusd  # EUR strengthening hurts
            return adj
        return 0.0

    rows = []
    for key, etf_ticker in sector_etf_cfg.items():
        region, bucket = key.split("|", 1)
        market_idx = market_idx_cfg.get(region)

        # Local returns
        l4  = _px_ret(etf_ticker, _bd4)
        l13 = _px_ret(etf_ticker, _bd13)

        # FX adjustments (additive log returns)
        fx4  = _fx_adj(region, _bd4)
        fx13 = _fx_adj(region, _bd13)

        # USD/EUR-converted returns
        u4  = (l4  + fx4)  if l4  is not None else None
        u13 = (l13 + fx13) if l13 is not None else None

        # Broad-market relative strength
        m4  = _px_ret(market_idx, _bd4)  if market_idx else None
        m13 = _px_ret(market_idx, _bd13) if market_idx else None

        rel4  = (l4  - m4)  if (l4  is not None and m4  is not None) else None
        rel13 = (l13 - m13) if (l13 is not None and m13 is not None) else None

        # Flow score: average of 4W and 13W relative strength in EUR-equivalent
        # Normalised to [-1, +1] using ±10% annualised threshold
        _scale = 0.10 / 252
        rel_signals = [r for r in [rel4, rel13] if r is not None]
        if rel_signals:
            avg_rel = np.mean(rel_signals)
            flow_score = float(np.clip(avg_rel / (_scale * _bd4), -1.0, 1.0))
        else:
            flow_score = 0.0

        rows.append({
            "region":         region,
            "bucket":         bucket,
            "etf_ticker":     etf_ticker,
            "local_ret_4w":   l4,
            "usd_ret_4w":     u4,
            "fx_contrib_4w":  fx4 if l4 is not None else None,
            "local_ret_13w":  l13,
            "usd_ret_13w":    u13,
            "fx_contrib_13w": fx13 if l13 is not None else None,
            "vs_market_4w":   rel4,
            "vs_market_13w":  rel13,
            "flow_score":     flow_score,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3 — Country Flows
# ---------------------------------------------------------------------------

def compute_country_flows(
    conn,
    params: dict,
    as_of_date: str,
) -> pd.DataFrame:
    """
    Broad index relative strength per region, converted to EUR base.
    Flow score > 0 → net inflow; < 0 → net outflow vs universe average.

    Returns DataFrame: region, broad_index, local_ret_4w, eur_ret_4w,
                        local_ret_13w, eur_ret_13w, flow_score, flow_direction
    """
    from src.data.db import get_prices, get_fx_rates

    market_idx_cfg = params.get("market_indices", {})
    regions = list(market_idx_cfg.keys())

    if not regions:
        return pd.DataFrame()

    lookback = 400
    start = (datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=lookback)).strftime("%Y-%m-%d")

    all_idx = list(set(market_idx_cfg.values()))
    prices_wide = get_prices(conn, all_idx, start, as_of_date)

    fx_pairs = ["EURUSD=X", "USDCAD=X"]
    fx_wide = get_fx_rates(conn, fx_pairs, start, as_of_date)

    _bd4  = 21
    _bd13 = 63

    def _fxadj_to_eur(region: str, n: int) -> float:
        """Same FX adjustment as sector_flows."""
        if region == "EU" or fx_wide.empty:
            return 0.0
        if region == "US":
            pair = "EURUSD=X"
            if pair not in fx_wide.columns:
                return 0.0
            r = _safe_log_ret(fx_wide[pair].dropna(), n)
            return -float(r) if r is not None else 0.0
        if region == "CA":
            usdcad = "USDCAD=X"
            eurusd = "EURUSD=X"
            r_u = _safe_log_ret(fx_wide[usdcad].dropna(), n) if usdcad in fx_wide.columns else None
            r_e = _safe_log_ret(fx_wide[eurusd].dropna(), n) if eurusd in fx_wide.columns else None
            return (-(r_u or 0.0)) + (-(r_e or 0.0))
        return 0.0

    rows = []
    for region in regions:
        idx = market_idx_cfg[region]
        if idx not in prices_wide.columns:
            continue
        s = prices_wide[idx].dropna()
        l4  = _safe_log_ret(s, _bd4)
        l13 = _safe_log_ret(s, _bd13)
        fx4  = _fxadj_to_eur(region, _bd4)
        fx13 = _fxadj_to_eur(region, _bd13)
        e4  = (l4  + fx4)  if l4  is not None else None
        e13 = (l13 + fx13) if l13 is not None else None

        rows.append({
            "region":       region,
            "broad_index":  idx,
            "local_ret_4w":  l4,
            "eur_ret_4w":    e4,
            "local_ret_13w": l13,
            "eur_ret_13w":   e13,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Flow score: rank EUR returns across regions (cross-sectional)
    for col in ["eur_ret_4w", "eur_ret_13w"]:
        valid = df[col].dropna()
        if len(valid) < 2:
            df[f"rank_{col}"] = 0.0
            continue
        mean_r, std_r = valid.mean(), valid.std()
        if std_r == 0:
            df[f"rank_{col}"] = 0.0
        else:
            df[f"rank_{col}"] = ((df[col] - mean_r) / std_r).clip(-2, 2) / 2.0

    df["flow_score"] = df[["rank_eur_ret_4w", "rank_eur_ret_13w"]].mean(axis=1)
    df["flow_direction"] = df["flow_score"].apply(
        lambda x: "inflow" if x > 0.15 else ("outflow" if x < -0.15 else "neutral")
    )
    df = df.drop(columns=["rank_eur_ret_4w", "rank_eur_ret_13w"])
    return df


# ---------------------------------------------------------------------------
# 4 — Implications
# ---------------------------------------------------------------------------

def derive_implications(
    fx_df: pd.DataFrame,
    usd_index: float,
    usd_trend: str,
    sector_flows: pd.DataFrame,
    country_flows: pd.DataFrame,
    params: dict,
) -> list[str]:
    """
    Translate quantitative flow signals into portfolio-relevant flags.

    Each implication is a short string describing the macro condition and
    the directional effect on the portfolio.
    """
    impl: list[str] = []
    thresholds = params.get("fx", {})
    safe_haven_thr  = thresholds.get("safe_haven_threshold", 0.02)   # 2% 1M log return
    flow_strong_thr = thresholds.get("flow_strong_threshold", 0.40)

    # ── USD regime ────────────────────────────────────────────────────────────
    if usd_trend == "strengthening":
        impl.append(
            f"USD_STRENGTHENING (index={usd_index:+.2f}): "
            "EU/CA stocks face FX headwind for USD-base investors; "
            "US-listed holdings gain currency tailwind."
        )
    elif usd_trend == "weakening":
        impl.append(
            f"USD_WEAKENING (index={usd_index:+.2f}): "
            "EU/CA stocks gain FX tailwind; "
            "risk-on environment typically supports cyclical sectors."
        )

    # ── Safe-haven signal (JPY / CHF) ─────────────────────────────────────────
    if not fx_df.empty:
        for pair in ("USDJPY=X", "USDCHF=X"):
            sub = fx_df[fx_df["pair"] == pair]
            if sub.empty:
                continue
            r1m = sub["return_1m"].iloc[0]
            if r1m is None:
                continue
            # For safe-haven pairs: USDJPY/USDCHF falling = JPY/CHF strengthening = risk-off
            if r1m is not None and r1m < -safe_haven_thr:
                ccy = "JPY" if "JPY" in pair else "CHF"
                impl.append(
                    f"SAFE_HAVEN_BID_{ccy} (1M={r1m:.2%}): "
                    f"{ccy} strengthening signals risk-off positioning — "
                    "consistent with elevated X_C / systemic stress."
                )

    # ── Sector rotation ───────────────────────────────────────────────────────
    if not sector_flows.empty:
        top_inflow = sector_flows.nlargest(3, "flow_score")
        top_outflow = sector_flows.nsmallest(3, "flow_score")
        in_str  = ", ".join(f"{r['region']}|{r['bucket']} ({r['flow_score']:+.2f})"
                            for _, r in top_inflow.iterrows()
                            if abs(r["flow_score"]) > flow_strong_thr)
        out_str = ", ".join(f"{r['region']}|{r['bucket']} ({r['flow_score']:+.2f})"
                            for _, r in top_outflow.iterrows()
                            if abs(r["flow_score"]) > flow_strong_thr)
        if in_str:
            impl.append(f"SECTOR_INFLOW: {in_str}")
        if out_str:
            impl.append(f"SECTOR_OUTFLOW: {out_str}")

    # ── Country flows ─────────────────────────────────────────────────────────
    if not country_flows.empty:
        for _, row in country_flows.iterrows():
            direction = row.get("flow_direction", "neutral")
            if direction == "neutral":
                continue
            r4  = row.get("eur_ret_4w")
            impl.append(
                f"COUNTRY_{row['region']}_{direction.upper()} "
                f"(4W EUR-adj={r4:.2%})" if r4 is not None else
                f"COUNTRY_{row['region']}_{direction.upper()}"
            )

    # ── FX contribution vs local return ───────────────────────────────────────
    if not sector_flows.empty:
        large_fx = sector_flows[sector_flows["fx_contrib_4w"].abs() > 0.03].copy()
        for _, row in large_fx.iterrows():
            fc = row["fx_contrib_4w"]
            impl.append(
                f"FX_{'TAILWIND' if fc > 0 else 'HEADWIND'}_{row['region']}|{row['bucket']} "
                f"(4W FX contribution = {fc:.2%}): "
                f"{'adds to' if fc > 0 else 'subtracts from'} total EUR return."
            )

    if not impl:
        impl.append("FX_NEUTRAL: No material FX or capital flow signals detected.")

    return impl


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def compute_capital_flows(
    conn,
    universe_df: pd.DataFrame,
    params: dict,
    as_of_date: str,
) -> dict:
    """
    Full capital flow analysis.  Returns results dict consumed by the report.
    Degrades gracefully — any sub-section that fails returns an empty DataFrame.
    """
    result: dict = {
        "fx_momentum":   pd.DataFrame(),
        "usd_index":     0.0,
        "usd_trend":     "neutral",
        "sector_flows":  pd.DataFrame(),
        "country_flows": pd.DataFrame(),
        "implications":  [],
        "as_of_date":    as_of_date,
    }

    try:
        fx_df, usd_idx, usd_trend = compute_fx_momentum(conn, as_of_date, params)
        result["fx_momentum"] = fx_df
        result["usd_index"]   = usd_idx
        result["usd_trend"]   = usd_trend
    except Exception as e:
        logger.warning(f"FX momentum computation failed: {e}")

    try:
        sector_flows = compute_sector_flows(conn, universe_df, params, as_of_date)
        result["sector_flows"] = sector_flows
    except Exception as e:
        logger.warning(f"Sector flow computation failed: {e}")

    try:
        country_flows = compute_country_flows(conn, params, as_of_date)
        result["country_flows"] = country_flows
    except Exception as e:
        logger.warning(f"Country flow computation failed: {e}")

    try:
        implications = derive_implications(
            result["fx_momentum"],
            result["usd_index"],
            result["usd_trend"],
            result["sector_flows"],
            result["country_flows"],
            params,
        )
        result["implications"] = implications
    except Exception as e:
        logger.warning(f"Implication derivation failed: {e}")

    n_impl = len(result["implications"])
    logger.info(
        f"Capital flow analysis complete: USD={usd_trend} ({usd_idx:+.2f}), "
        f"{n_impl} implications"
    )
    return result
