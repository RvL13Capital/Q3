"""
Fractional Kelly position sizing — EARKE eq 12.

Full objective (eq 12 — Robust Kelly):

  f* = argmax_f [
      (μ_base + μ_NN) · f              ← total expected return (μ_NN=0 until Module III)
      − λ · σ_epist · f                ← epistemic risk penalty (Not-Aus trigger)
      − ½ · (σ²_alea + σ²_epist) · f² ← total variance (aleatoric + epistemic)
      − c · σ_alea · |f − f_old|^1.5 · √(W/V)  ← market impact cost
  ]

  c        = impact_scaling (≈1.0, dimensionless)
  W        = AUM in currency units (params: kelly.aum_eur)
  V        = average daily dollar volume (last 60 days)
  f_old    = previous fraction-scaled portfolio weight for this ticker (kelly_25pct from portfolio_snapshots)
  σ_epist  = epistemic uncertainty proxy (from Deep Ensembles, Phase 3;
             currently derived as 1 − composite_confidence)
  λ        = lambda_epist penalty coefficient (params: kelly.lambda_epist)

Not-Aus (emergency stop):
  σ_epist ≥ not_aus_threshold → f* = 0 immediately.
  In the composite_confidence proxy: not_aus_threshold = 1 − not_aus_confidence,
  so composite_confidence < not_aus_confidence triggers Not-Aus.

When W/V is small (liquid stock, small fund), the impact term is negligible and
f* ≈ fraction × (μ−rf−λ·σ_epist) / (σ²+σ²_epist)  (generalised fractional Kelly).
As AUM grows, the impact penalty rises, ultimately capping W_max — Goodhart immunity.

Optimised via scipy.optimize.minimize_scalar on [0, 1].
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# σ estimation
# ---------------------------------------------------------------------------

def estimate_sigma(
    ticker: str,
    conn,
    as_of_date: str,
    window_days: int = 252,
) -> tuple[float, bool]:
    """
    Annualized volatility from trailing log returns.
    Returns (sigma, is_valid).
    """
    from src.data.db import get_prices
    from src.data.prices import compute_log_returns

    start = (datetime.strptime(as_of_date, "%Y-%m-%d")
             - timedelta(days=window_days + 60)).strftime("%Y-%m-%d")
    price_df = get_prices(conn, [ticker], start, as_of_date)

    if price_df.empty or ticker not in price_df.columns:
        return 0.35, False

    returns = compute_log_returns(price_df[[ticker]])[ticker].dropna().tail(window_days)
    n = len(returns)
    if n < 30:
        return 0.35, False

    sigma = float(returns.std() * np.sqrt(252))
    return max(0.05, sigma), True


# ---------------------------------------------------------------------------
# Daily dollar volume estimation
# ---------------------------------------------------------------------------

def estimate_daily_dollar_volume(
    ticker: str,
    conn,
    as_of_date: str,
    window_days: int = 60,
) -> float:
    """
    Average daily dollar volume over the trailing window (price × volume).
    Returns 0.0 if unavailable, which disables the impact term.
    """
    from src.data.db import get_prices

    start = (datetime.strptime(as_of_date, "%Y-%m-%d")
             - timedelta(days=window_days + 10)).strftime("%Y-%m-%d")

    # Fetch raw OHLCV (need volume column, not in the wide adj_close-only format)
    try:
        df = conn.execute("""
            SELECT date, adj_close, volume
            FROM prices
            WHERE ticker = ? AND date >= ? AND date <= ?
            ORDER BY date
        """, [ticker, start, as_of_date]).df()
    except Exception:
        return 0.0

    if df.empty or df["volume"].isna().all():
        return 0.0

    df = df.dropna(subset=["adj_close", "volume"]).tail(window_days)
    if df.empty:
        return 0.0

    daily_vol = (df["adj_close"] * df["volume"]).mean()
    return float(daily_vol) if daily_vol > 0 else 0.0


# ---------------------------------------------------------------------------
# Last Kelly fraction from previous portfolio snapshot
# ---------------------------------------------------------------------------

def get_last_kelly_fraction(conn, ticker: str) -> float:
    """Return kelly_25pct from the most recent portfolio snapshot for this ticker."""
    try:
        result = conn.execute("""
            SELECT kelly_25pct
            FROM portfolio_snapshots
            WHERE ticker = ?
            ORDER BY snapshot_date DESC
            LIMIT 1
        """, [ticker]).fetchone()
        if result and result[0] is not None:
            return float(result[0])
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Core Kelly optimisation (eq 12, without σ_epist)
# ---------------------------------------------------------------------------

def kelly_fraction(
    mu: float,
    sigma: float,
    rf: float,
    fraction: float = 0.25,
    f_old: float = 0.0,
    aum: float = 0.0,
    daily_dollar_volume: float = 0.0,
    impact_scaling: float = 1.0,
    sigma_epist: float = 0.0,
    lambda_epist: float = 0.25,
    not_aus_threshold: float = 0.0,
) -> tuple[float, float]:
    """
    Optimise the eq 12 Kelly objective with epistemic penalty and market impact.

    Parameters
    ----------
    sigma_epist       : epistemic uncertainty proxy [0, ∞).  Derived from
                        composite_confidence as (1 − confidence) until Deep
                        Ensembles are implemented (Module III).
    lambda_epist      : linear penalty coefficient λ for the σ_epist term.
    not_aus_threshold : σ_epist ≥ this → f* = 0 (Not-Aus).  0 = disabled.

    Returns (f_adjusted, f_full) where:
      f_full     = unconstrained optimiser solution (full Kelly, pre-fraction)
      f_adjusted = fraction × f_full, clamped to [0, 1]

    Degrades gracefully: when sigma_epist=0 the result is identical to the
    pre-Module-III behaviour.
    """
    # ── Not-Aus gate (σ_epist → ∞ ⟹ f* = 0) ────────────────────────────────
    if not_aus_threshold > 0 and sigma_epist >= not_aus_threshold:
        return 0.0, 0.0

    if sigma <= 0 or mu <= rf:
        return 0.0, 0.0

    # ── Total variance (aleatoric + epistemic) ───────────────────────────────
    sigma_sq_total = sigma ** 2 + sigma_epist ** 2
    # Numerical stability: floor prevents division by near-zero variance
    sigma_sq_total = max(sigma_sq_total, 1e-10)

    # ── Effective excess return after linear epistemic penalty ───────────────
    mu_eff = (mu - rf) - lambda_epist * sigma_epist
    if mu_eff <= 0:
        return 0.0, 0.0

    # ── Impact term weight √(W/V); 0 when either unknown ────────────────────
    if daily_dollar_volume > 0 and aum > 0:
        sqrt_wv = np.sqrt(aum / daily_dollar_volume)
    else:
        sqrt_wv = 0.0

    if sqrt_wv < 1e-8:
        # No impact data — generalised fractional Kelly
        f_full     = mu_eff / sigma_sq_total
        f_adjusted = float(max(0.0, min(1.0, fraction * f_full)))
        return f_adjusted, f_full

    def neg_objective(f: float) -> float:
        growth   = mu_eff * f - 0.5 * sigma_sq_total * f ** 2
        # f is in full-Kelly space; f_old is the previous fraction-scaled weight
        # (kelly_25pct). Project f onto weight space before computing turnover so
        # that the impact penalty reflects the actual portfolio weight change.
        turnover = abs(f * fraction - f_old)
        impact   = impact_scaling * sigma * (turnover ** 1.5) * sqrt_wv
        return -(growth - impact)

    # Optimise over [0, 1/fraction] so fraction scaling is consistent with
    # classic fractional Kelly when impact and σ_epist are negligible.
    upper = min(4.0, 1.0 / max(fraction, 0.05))
    result = minimize_scalar(neg_objective, bounds=(0.0, upper), method="bounded")
    f_full     = float(result.x) if result.success else mu_eff / sigma_sq_total
    f_adjusted = float(max(0.0, min(1.0, fraction * f_full)))

    return f_adjusted, f_full


# ---------------------------------------------------------------------------
# AUM capacity ceiling (derived from eq 12)
# ---------------------------------------------------------------------------

def compute_w_max(
    mu_robust: float,
    sigma: float,
    daily_dollar_volume: float,
    f_old: float,
    target_turnover: float = 0.10,
    impact_scaling: float = 1.0,
) -> float:
    """
    Hard AUM ceiling beyond which alpha is fully absorbed by market impact.

    W_max = V × ( μ_robust / (c · σ · |Δf|^1.5) )²

    Returns 0.0 if underdetermined.
    """
    if daily_dollar_volume <= 0 or sigma <= 0 or mu_robust <= 0:
        return 0.0
    if target_turnover < 1e-6:
        return 0.0
    denominator = impact_scaling * sigma * (target_turnover ** 1.5)
    if denominator < 1e-10:
        return 0.0
    return float(daily_dollar_volume * (mu_robust / denominator) ** 2)


# ---------------------------------------------------------------------------
# ADV liquidity fraction (market impact diagnostic)
# ---------------------------------------------------------------------------

def compute_adv_fraction(
    weight: float,
    aum: float,
    daily_dollar_volume: float,
) -> Optional[float]:
    """
    Fraction of Average Daily Volume consumed by this position.
    Returns None if ADV data unavailable.

    Positions consuming >10% of ADV indicate market impact risk beyond
    what the Kelly impact term captures (slippage during execution).
    """
    if aum <= 0 or daily_dollar_volume <= 0 or weight <= 0:
        return None
    position_dollars = weight * aum
    return position_dollars / daily_dollar_volume


# ---------------------------------------------------------------------------
# Compute Kelly weights for scored DataFrame
# ---------------------------------------------------------------------------

def compute_kelly_weights(
    scored_df: pd.DataFrame,
    conn,
    params: dict,
    as_of_date: str,
    universe_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each stock with entry_signal=True, compute Kelly position size.
    Returns DataFrame with: ticker, mu_estimate, sigma_estimate, rf_rate,
    kelly_fraction, kelly_25pct, w_max, primary_bucket, composite_score.
    """
    from src.data.macro import get_risk_free_rate

    universe_map = universe_df.set_index("ticker")[["region", "primary_bucket"]]
    candidates   = scored_df[scored_df["entry_signal"] == True].copy()

    if candidates.empty:
        return pd.DataFrame()

    kelly_params    = params["kelly"]
    frac            = kelly_params["fraction"]
    aum             = float(kelly_params.get("aum_eur", 0.0))
    impact_scaling  = float(kelly_params.get("impact_scaling", 1.0))
    lambda_epist    = float(kelly_params.get("lambda_epist", 0.25))
    # Not-Aus fires when composite_confidence < not_aus_confidence.
    # Mapped: not_aus_threshold = 1 − not_aus_confidence (in σ_epist proxy space).
    not_aus_conf      = float(kelly_params.get("not_aus_confidence", 0.20))
    not_aus_threshold = max(0.0, 1.0 - not_aus_conf)

    # Pre-fetch rf rates once per region.
    regions  = universe_df["region"].unique().tolist()
    rf_cache = {r: get_risk_free_rate(conn, r, as_of_date, params) for r in regions}

    rows = []
    for _, row in candidates.iterrows():
        ticker      = row["ticker"]
        region_info = universe_map.loc[ticker] if ticker in universe_map.index else None
        region      = region_info["region"]        if region_info is not None else "US"
        bucket      = region_info["primary_bucket"] if region_info is not None else None

        rf = rf_cache.get(region, rf_cache.get("US", 0.04))
        mu = row.get("mu_estimate")
        if mu is None or pd.isna(mu):
            continue

        sigma, sigma_valid = estimate_sigma(ticker, conn, as_of_date)
        f_old              = get_last_kelly_fraction(conn, ticker)
        daily_vol          = estimate_daily_dollar_volume(ticker, conn, as_of_date)

        # σ_epist proxy: low confidence → high epistemic uncertainty.
        # composite_confidence ∈ [0.40, 1.0] after the confidence gate in composite.py.
        # proxy ∈ [0.0, 0.60]; proxy = 0 means full confidence, proxy → 1 triggers Not-Aus.
        comp_conf = row.get("composite_confidence", 1.0)
        if pd.isna(comp_conf) or comp_conf <= 0:
            comp_conf = 1.0
        sigma_epist_proxy = max(0.0, 1.0 - comp_conf)

        # Explicit Not-Aus check for logging (kelly_fraction enforces it internally too)
        not_aus_fired = not_aus_threshold > 0 and sigma_epist_proxy >= not_aus_threshold
        if not_aus_fired:
            logger.warning(
                "NOT-AUS: %s  composite_confidence=%.2f < %.2f threshold → f*=0",
                ticker, comp_conf, not_aus_conf,
            )

        f_adjusted, f_full = kelly_fraction(
            mu=mu, sigma=sigma, rf=rf, fraction=frac,
            f_old=f_old, aum=aum,
            daily_dollar_volume=daily_vol,
            impact_scaling=impact_scaling,
            sigma_epist=sigma_epist_proxy,
            lambda_epist=lambda_epist,
            not_aus_threshold=not_aus_threshold,
        )
        # f_adjusted already incorporates σ_epist via the objective (eq 12)

        w_max = compute_w_max(
            mu_robust=max(0.0, mu - rf),
            sigma=sigma,
            daily_dollar_volume=daily_vol,
            f_old=f_old,
            impact_scaling=impact_scaling,
        )

        # ADV liquidity diagnostic: flag positions consuming >10% of daily volume
        adv_frac = compute_adv_fraction(f_adjusted, aum, daily_vol)
        if adv_frac is not None and adv_frac > 0.10:
            logger.warning(
                "ADV_LIQUIDITY: %s  position=%.1f%% of ADV (%.1f%% of AUM × €%.0fM ADV) "
                "— execution slippage risk",
                ticker, adv_frac * 100, f_adjusted * 100, daily_vol / 1e6,
            )

        rows.append({
            "ticker":               ticker,
            "mu_estimate":          round(mu, 4),
            "sigma_estimate":       round(sigma, 4),
            "rf_rate":              round(rf, 4),
            "kelly_raw":            round(f_full, 4),    # full Kelly before fraction
            "kelly_25pct":          round(f_adjusted, 4),  # fraction-scaled + σ_epist-penalised
            "w_max_eur":            round(w_max, 0) if w_max > 0 else None,
            "adv_fraction":         round(adv_frac, 4) if adv_frac is not None else None,
            "primary_bucket":       bucket,
            "composite_score":      row.get("composite_score"),
            "composite_confidence": round(comp_conf, 4),
            "sigma_epist":          round(sigma_epist_proxy, 4),
            "not_aus":              not_aus_fired,
            "sigma_valid":          sigma_valid,
        })

    return pd.DataFrame(rows)
