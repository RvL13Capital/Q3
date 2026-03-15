"""
Unit tests for Kelly sizing and portfolio construction.
No external API calls or DuckDB needed.
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.portfolio.kelly import kelly_fraction, compute_kelly_weights
from src.portfolio.construction import apply_constraints

# ────────────────────────────────────────────────────────────────────────────
# Kelly formula tests
# ────────────────────────────────────────────────────────────────────────────

def test_kelly_basic():
    """μ=0.15, σ=0.30, rf=0.04 → f* = 0.25*(0.15-0.04)/0.09 ≈ 0.306"""
    f, _ = kelly_fraction(mu=0.15, sigma=0.30, rf=0.04, fraction=0.25)
    expected = 0.25 * (0.15 - 0.04) / (0.30 ** 2)
    assert f == pytest.approx(expected, abs=1e-6)
    assert 0 < f < 1


def test_kelly_zero_excess_return():
    """When μ == rf, f* = 0."""
    f, _ = kelly_fraction(mu=0.04, sigma=0.25, rf=0.04, fraction=0.25)
    assert f == 0.0


def test_kelly_negative_excess_return():
    """When μ < rf, f* = 0 (no short selling)."""
    f, _ = kelly_fraction(mu=0.02, sigma=0.25, rf=0.04, fraction=0.25)
    assert f == 0.0


def test_kelly_clamped_high():
    """Extreme signal → raw Kelly > 1 → clamped to 1.0."""
    f, _ = kelly_fraction(mu=0.50, sigma=0.10, rf=0.04, fraction=0.25)
    assert f == 1.0


def test_kelly_zero_sigma():
    """σ = 0 → return 0 (avoid division by zero)."""
    f, _ = kelly_fraction(mu=0.15, sigma=0.0, rf=0.04, fraction=0.25)
    assert f == 0.0


def test_kelly_full_fraction():
    """fraction=1.0 should give 4x the 0.25 fraction result."""
    f1, _ = kelly_fraction(mu=0.15, sigma=0.30, rf=0.04, fraction=0.25)
    f4, _ = kelly_fraction(mu=0.15, sigma=0.30, rf=0.04, fraction=1.0)
    assert f4 == pytest.approx(min(1.0, 4 * f1), abs=1e-6)


def test_kelly_raw_gte_25pct():
    """kelly_raw (full Kelly) must be >= kelly_25pct (fraction-scaled) when fraction < 1."""
    f_adj, f_full = kelly_fraction(mu=0.15, sigma=0.30, rf=0.04, fraction=0.25)
    assert f_full >= f_adj - 1e-9  # full Kelly >= fractioned


# ────────────────────────────────────────────────────────────────────────────
# σ_epist / Not-Aus tests (eq 12 — epistemic risk terms)
# ────────────────────────────────────────────────────────────────────────────

def test_kelly_sigma_epist_reduces_fraction():
    """High σ_epist (low confidence) must reduce f* compared to σ_epist=0."""
    f_no_epist, _ = kelly_fraction(mu=0.15, sigma=0.30, rf=0.04, fraction=0.25, sigma_epist=0.0)
    # sigma_epist=0.10, lambda=0.25: mu_eff = 0.11 - 0.025 = 0.085 > 0 but < 0.11
    f_with_epist, _ = kelly_fraction(mu=0.15, sigma=0.30, rf=0.04, fraction=0.25, sigma_epist=0.10)
    assert f_with_epist < f_no_epist


def test_kelly_sigma_epist_zero_unchanged():
    """sigma_epist=0 must give identical result to the baseline (backward compatible)."""
    f_base, fraw_base = kelly_fraction(mu=0.15, sigma=0.30, rf=0.04, fraction=0.25)
    f_epist, fraw_epist = kelly_fraction(mu=0.15, sigma=0.30, rf=0.04, fraction=0.25, sigma_epist=0.0)
    assert f_base == pytest.approx(f_epist, abs=1e-9)
    assert fraw_base == pytest.approx(fraw_epist, abs=1e-9)


def test_kelly_not_aus_fires():
    """Not-Aus: sigma_epist >= not_aus_threshold → f* = 0 immediately."""
    f, fraw = kelly_fraction(
        mu=0.15, sigma=0.30, rf=0.04, fraction=0.25,
        sigma_epist=0.80,
        not_aus_threshold=0.80,
    )
    assert f == 0.0
    assert fraw == 0.0


def test_kelly_not_aus_below_threshold():
    """Not-Aus does NOT fire when sigma_epist < not_aus_threshold."""
    # sigma_epist=0.05, lambda=0.25: mu_eff = 0.11 - 0.0125 = 0.0975 > 0; 0.05 < 0.80 threshold
    f, _ = kelly_fraction(
        mu=0.15, sigma=0.30, rf=0.04, fraction=0.25,
        sigma_epist=0.05,
        not_aus_threshold=0.80,
    )
    assert f > 0.0


def test_kelly_not_aus_disabled_when_threshold_zero():
    """not_aus_threshold=0 (default) means Not-Aus is disabled."""
    f, _ = kelly_fraction(
        mu=0.15, sigma=0.30, rf=0.04, fraction=0.25,
        sigma_epist=0.99,
        not_aus_threshold=0.0,
    )
    # sigma_epist is large but Not-Aus disabled; lambda penalty kills f anyway
    # Just assert no exception and result is ≥ 0
    assert f >= 0.0


def test_kelly_epist_large_kills_position():
    """When σ_epist is large enough, the linear penalty makes mu_eff ≤ 0 → f* = 0."""
    # lambda_epist=0.25, sigma_epist=0.50, mu-rf=0.11: mu_eff = 0.11 - 0.125 = -0.015 → 0
    f, _ = kelly_fraction(
        mu=0.15, sigma=0.30, rf=0.04, fraction=0.25,
        sigma_epist=0.50, lambda_epist=0.25,
    )
    assert f == 0.0


def test_kelly_lambda_epist_scale():
    """Higher lambda_epist → stronger penalty → lower f*."""
    f_low, _ = kelly_fraction(
        mu=0.20, sigma=0.30, rf=0.04, fraction=0.25,
        sigma_epist=0.10, lambda_epist=0.5,
    )
    f_high, _ = kelly_fraction(
        mu=0.20, sigma=0.30, rf=0.04, fraction=0.25,
        sigma_epist=0.10, lambda_epist=2.0,
    )
    assert f_high < f_low


def test_kelly_zero_turnover_no_phantom_impact():
    """
    Scale-alignment invariant (regression for the f_old dimension bug):

    When the previous fraction-scaled weight (f_old = kelly_25pct) already equals
    the target weight (fraction × f_full), turnover is zero and the impact term
    must vanish.  The result must therefore be identical to the no-impact baseline.

    Concretely: fraction=0.25, f_full≈1.222, f_adjusted≈0.306.
    If f_old=0.306 (matching the target weight), impact cost should be 0 and
    the returned fraction must equal the baseline (no AUM) result.

    Before the fix, turnover = |f_full - f_old| = |1.222 - 0.306| ≈ 0.916
    (phantom turnover), causing the optimizer to suppress f* artificially.
    After the fix, turnover = |f_full × fraction - f_old| = |0.306 - 0.306| = 0.
    """
    fraction = 0.25
    mu, sigma, rf = 0.15, 0.30, 0.04

    # Compute baseline (no impact) to know the exact target weight.
    f_baseline, _ = kelly_fraction(mu=mu, sigma=sigma, rf=rf, fraction=fraction)

    # Set f_old equal to the target weight — zero true turnover.
    f_with_impact, _ = kelly_fraction(
        mu=mu, sigma=sigma, rf=rf, fraction=fraction,
        f_old=f_baseline,
        aum=1_000_000,
        daily_dollar_volume=1_000_000,  # sqrt_wv = 1.0 → non-trivial impact term
        impact_scaling=1.0,
    )

    # With zero turnover, impact = 0 → result must match the no-impact baseline.
    assert f_with_impact == pytest.approx(f_baseline, abs=1e-4)


# ────────────────────────────────────────────────────────────────────────────
# Almgren-Chriss participation penalty tests
# ────────────────────────────────────────────────────────────────────────────

def test_participation_penalty_reduces_allocation():
    """
    With eta_participation > 0 and finite AUM/ADV, the optimizer allocates
    less than the no-participation baseline (endogenous liquidity avoidance).
    """
    mu, sigma, rf = 0.15, 0.30, 0.04
    aum = 50_000_000
    adv = 10_000_000  # position would be 5× daily volume → very illiquid

    f_baseline, _ = kelly_fraction(mu=mu, sigma=sigma, rf=rf, fraction=0.25,
                                   eta_participation=0.0,
                                   aum=aum, daily_dollar_volume=adv)
    f_penalized, _ = kelly_fraction(mu=mu, sigma=sigma, rf=rf, fraction=0.25,
                                    eta_participation=0.50,
                                    aum=aum, daily_dollar_volume=adv)

    assert f_penalized < f_baseline
    assert f_penalized > 0  # still positive (not killed entirely)


def test_participation_penalty_zero_eta_no_effect():
    """When eta_participation=0, result is identical to eta-absent baseline
    (both still include turnover impact if AUM/ADV are set)."""
    mu, sigma, rf = 0.15, 0.30, 0.04
    aum, adv = 50_000_000, 10_000_000
    # Both calls include AUM/ADV so turnover impact is present in both.
    f_absent, _ = kelly_fraction(mu=mu, sigma=sigma, rf=rf, fraction=0.25,
                                 aum=aum, daily_dollar_volume=adv)
    f_zero_eta, _ = kelly_fraction(mu=mu, sigma=sigma, rf=rf, fraction=0.25,
                                   eta_participation=0.0,
                                   aum=aum, daily_dollar_volume=adv)
    assert f_zero_eta == pytest.approx(f_absent, abs=1e-6)


def test_participation_penalty_scales_with_illiquidity():
    """More illiquid (lower ADV) → larger penalty → lower allocation."""
    mu, sigma, rf = 0.15, 0.30, 0.04
    aum = 50_000_000
    eta = 0.50

    f_liquid, _ = kelly_fraction(mu=mu, sigma=sigma, rf=rf, fraction=0.25,
                                 eta_participation=eta,
                                 aum=aum, daily_dollar_volume=100_000_000)  # very liquid
    f_illiquid, _ = kelly_fraction(mu=mu, sigma=sigma, rf=rf, fraction=0.25,
                                   eta_participation=eta,
                                   aum=aum, daily_dollar_volume=5_000_000)  # illiquid

    assert f_illiquid < f_liquid


def test_participation_penalty_higher_eta_lower_allocation():
    """Higher η → stronger penalty → lower allocation."""
    mu, sigma, rf = 0.15, 0.30, 0.04
    aum = 50_000_000
    adv = 20_000_000

    f_low_eta, _ = kelly_fraction(mu=mu, sigma=sigma, rf=rf, fraction=0.25,
                                  eta_participation=0.10,
                                  aum=aum, daily_dollar_volume=adv)
    f_high_eta, _ = kelly_fraction(mu=mu, sigma=sigma, rf=rf, fraction=0.25,
                                   eta_participation=1.0,
                                   aum=aum, daily_dollar_volume=adv)

    assert f_high_eta < f_low_eta


def test_participation_penalty_adv_zero_disabled():
    """When ADV=0, participation penalty is disabled (coefficient = 0)."""
    mu, sigma, rf = 0.15, 0.30, 0.04
    # With ADV=0, participation cannot be computed → same as no penalty
    f_no_adv, _ = kelly_fraction(mu=mu, sigma=sigma, rf=rf, fraction=0.25,
                                 eta_participation=0.50,
                                 aum=50_000_000, daily_dollar_volume=0.0)
    f_base, _ = kelly_fraction(mu=mu, sigma=sigma, rf=rf, fraction=0.25)
    assert f_no_adv == pytest.approx(f_base, abs=1e-6)


def test_participation_penalty_not_aus_dominates():
    """Not-Aus gate fires before participation penalty is evaluated."""
    mu, sigma, rf = 0.15, 0.30, 0.04
    f, _ = kelly_fraction(mu=mu, sigma=sigma, rf=rf, fraction=0.25,
                          eta_participation=0.50,
                          aum=50_000_000, daily_dollar_volume=10_000_000,
                          sigma_epist=0.90, not_aus_threshold=0.80)
    assert f == 0.0  # Not-Aus overrides everything


def test_participation_penalty_with_nonzero_f_old():
    """Participation penalty and turnover impact coexist correctly with f_old > 0."""
    mu, sigma, rf = 0.15, 0.30, 0.04
    aum, adv = 50_000_000, 20_000_000

    # With f_old > 0, both turnover impact and participation are active
    f_with_folds, _ = kelly_fraction(mu=mu, sigma=sigma, rf=rf, fraction=0.25,
                                     f_old=0.05, eta_participation=0.50,
                                     aum=aum, daily_dollar_volume=adv)
    # Should still produce a positive, bounded result
    assert 0 < f_with_folds < 1


# ────────────────────────────────────────────────────────────────────────────
# Portfolio construction constraint tests
# ────────────────────────────────────────────────────────────────────────────

PARAMS = {
    "kelly": {
        "fraction": 0.25,
        "max_position": 0.08,
        "max_bucket":   0.35,
        "min_position": 0.02,
        "cash_reserve": 0.10,
    }
}


def make_kelly_df(rows):
    """Helper to create a kelly_df-style DataFrame."""
    return pd.DataFrame(rows)


def test_constraints_single_position():
    """One stock, 15% raw Kelly → capped to 8%."""
    df = make_kelly_df([
        {"ticker": "A", "kelly_25pct": 0.15, "primary_bucket": "grid", "composite_score": 0.7, "kelly_raw": 0.15}
    ])
    result = apply_constraints(df, PARAMS)
    assert len(result) == 1
    assert result.iloc[0]["weight"] == pytest.approx(0.08)
    assert bool(result.iloc[0]["is_constrained"]) is True


def test_constraints_bucket_cap():
    """4 stocks in same bucket, each 12% Kelly → bucket capped to 35%."""
    df = make_kelly_df([
        {"ticker": f"A{i}", "kelly_25pct": 0.12, "primary_bucket": "defense", "composite_score": 0.7, "kelly_raw": 0.12}
        for i in range(4)
    ])
    result = apply_constraints(df, PARAMS)
    bucket_total = result["weight"].sum()
    assert bucket_total <= 0.35 + 1e-6


def test_constraints_stock_cap_after_bucket_scale():
    """After bucket scaling, no stock exceeds 8%."""
    df = make_kelly_df([
        {"ticker": f"A{i}", "kelly_25pct": 0.12, "primary_bucket": "defense", "composite_score": 0.7, "kelly_raw": 0.12}
        for i in range(4)
    ])
    result = apply_constraints(df, PARAMS)
    assert (result["weight"] <= 0.08 + 1e-6).all()


def test_constraints_cash_floor():
    """Total weights sum to ≤ 90% (cash_reserve = 10%)."""
    df = make_kelly_df([
        {"ticker": f"A{i}", "kelly_25pct": 0.08, "primary_bucket": f"b{i}", "composite_score": 0.7, "kelly_raw": 0.08}
        for i in range(12)  # 12 * 8% = 96% > 90%
    ])
    result = apply_constraints(df, PARAMS)
    total = result["weight"].sum()
    assert total <= 0.90 + 1e-6


def test_constraints_min_position_drop():
    """Stocks that fall below 2% after scaling are dropped."""
    df = make_kelly_df([
        {"ticker": "A", "kelly_25pct": 0.08, "primary_bucket": "grid", "composite_score": 0.8, "kelly_raw": 0.08},
        {"ticker": "B", "kelly_25pct": 0.015, "primary_bucket": "water", "composite_score": 0.56, "kelly_raw": 0.015},  # below min
    ])
    result = apply_constraints(df, PARAMS)
    assert "B" not in result["ticker"].values


def test_constraints_empty_input():
    """Empty DataFrame → empty result."""
    df = pd.DataFrame(columns=["ticker", "kelly_25pct", "primary_bucket"])
    result = apply_constraints(df, PARAMS)
    assert result.empty


def test_constraints_multi_bucket():
    """Two buckets each with stocks — constraints respected per bucket."""
    rows = []
    for i in range(5):
        rows.append({"ticker": f"G{i}", "kelly_25pct": 0.09, "primary_bucket": "grid", "composite_score": 0.7, "kelly_raw": 0.09})
    for i in range(3):
        rows.append({"ticker": f"D{i}", "kelly_25pct": 0.08, "primary_bucket": "defense", "composite_score": 0.68, "kelly_raw": 0.08})
    df = make_kelly_df(rows)
    result = apply_constraints(df, PARAMS)

    # Grid bucket total ≤ 35%
    grid_total = result[result["primary_bucket"] == "grid"]["weight"].sum()
    assert grid_total <= 0.35 + 1e-6

    # Defense bucket total ≤ 35%
    def_total = result[result["primary_bucket"] == "defense"]["weight"].sum()
    assert def_total <= 0.35 + 1e-6

    # Cash floor
    assert result["weight"].sum() <= 0.90 + 1e-6


def test_constraints_idempotent():
    """Calling apply_constraints twice gives the same result (weights already satisfy constraints)."""
    df = make_kelly_df([
        {"ticker": f"A{i}", "kelly_25pct": 0.07, "primary_bucket": f"b{i%3}", "composite_score": 0.6, "kelly_raw": 0.07}
        for i in range(8)
    ])
    result1 = apply_constraints(df, PARAMS)
    # Build a fresh df from result1's constrained weights as the new kelly_25pct
    df2 = make_kelly_df([
        {"ticker": row["ticker"], "kelly_25pct": row["weight"],
         "primary_bucket": row["primary_bucket"], "composite_score": row.get("composite_score", 0.6), "kelly_raw": row["weight"]}
        for _, row in result1.iterrows()
    ])
    result2 = apply_constraints(df2, PARAMS)
    # Weights should be the same (already within all constraints)
    w1 = result1.set_index("ticker")["weight"].sort_index()
    w2 = result2.set_index("ticker")["weight"].sort_index()
    assert (abs(w1 - w2) < 0.001).all()


# ────────────────────────────────────────────────────────────────────────────
# Momentum boost tests (proto-μ_NN — symmetric swing_score overlay)
# ────────────────────────────────────────────────────────────────────────────

def test_momentum_boost_positive():
    """Strong momentum (swing=0.85, conf=0.90, beta=0.50) → larger Kelly than baseline."""
    mu, sigma, rf = 0.15, 0.30, 0.04
    f_base, _ = kelly_fraction(mu=mu, sigma=sigma, rf=rf, fraction=0.25)
    f_boosted, _ = kelly_fraction(
        mu=mu, sigma=sigma, rf=rf, fraction=0.25,
        swing_score=0.85, swing_confidence=0.90, momentum_boost_beta=0.50,
    )
    assert f_boosted > f_base


def test_momentum_boost_negative():
    """Weak momentum (swing=0.15, conf=0.90, beta=0.50) → smaller Kelly than baseline."""
    mu, sigma, rf = 0.15, 0.30, 0.04
    f_base, _ = kelly_fraction(mu=mu, sigma=sigma, rf=rf, fraction=0.25)
    f_discounted, _ = kelly_fraction(
        mu=mu, sigma=sigma, rf=rf, fraction=0.25,
        swing_score=0.15, swing_confidence=0.90, momentum_boost_beta=0.50,
    )
    assert f_discounted < f_base


def test_momentum_boost_neutral_unchanged():
    """Neutral momentum (swing=0.50) → no change from baseline."""
    mu, sigma, rf = 0.15, 0.30, 0.04
    f_base, _ = kelly_fraction(mu=mu, sigma=sigma, rf=rf, fraction=0.25)
    f_neutral, _ = kelly_fraction(
        mu=mu, sigma=sigma, rf=rf, fraction=0.25,
        swing_score=0.50, swing_confidence=0.90, momentum_boost_beta=0.50,
    )
    assert f_neutral == pytest.approx(f_base, abs=1e-6)


def test_momentum_boost_none_swing_unchanged():
    """swing_score=None → no boost → identical to baseline."""
    mu, sigma, rf = 0.15, 0.30, 0.04
    f_base, fraw_base = kelly_fraction(mu=mu, sigma=sigma, rf=rf, fraction=0.25)
    f_none, fraw_none = kelly_fraction(
        mu=mu, sigma=sigma, rf=rf, fraction=0.25,
        swing_score=None, swing_confidence=None, momentum_boost_beta=0.50,
    )
    assert f_none == pytest.approx(f_base, abs=1e-9)
    assert fraw_none == pytest.approx(fraw_base, abs=1e-9)


def test_momentum_boost_beta_zero_disabled():
    """beta=0.0 → momentum has zero effect regardless of swing_score."""
    mu, sigma, rf = 0.15, 0.30, 0.04
    f_base, _ = kelly_fraction(mu=mu, sigma=sigma, rf=rf, fraction=0.25)
    f_beta0, _ = kelly_fraction(
        mu=mu, sigma=sigma, rf=rf, fraction=0.25,
        swing_score=1.0, swing_confidence=1.0, momentum_boost_beta=0.0,
    )
    assert f_beta0 == pytest.approx(f_base, abs=1e-9)


def test_momentum_boost_confidence_gates():
    """swing_confidence=0.0 → no boost even with extreme swing_score."""
    mu, sigma, rf = 0.15, 0.30, 0.04
    f_base, _ = kelly_fraction(mu=mu, sigma=sigma, rf=rf, fraction=0.25)
    f_no_conf, _ = kelly_fraction(
        mu=mu, sigma=sigma, rf=rf, fraction=0.25,
        swing_score=1.0, swing_confidence=0.0, momentum_boost_beta=0.50,
    )
    assert f_no_conf == pytest.approx(f_base, abs=1e-9)


def test_momentum_boost_bounded():
    """Boost multiplier is bounded: μ_boosted ∈ [(1-β)·μ, (1+β)·μ]."""
    mu, sigma, rf = 0.15, 0.30, 0.04
    beta = 0.50

    # Max boost: swing=1.0, conf=1.0 → μ *= 1.50
    f_max, _ = kelly_fraction(
        mu=mu, sigma=sigma, rf=rf, fraction=0.25,
        swing_score=1.0, swing_confidence=1.0, momentum_boost_beta=beta,
    )
    # Compare to manually boosted μ
    f_manual_max, _ = kelly_fraction(mu=mu * 1.50, sigma=sigma, rf=rf, fraction=0.25)
    assert f_max == pytest.approx(f_manual_max, abs=1e-6)

    # Max discount: swing=0.0, conf=1.0 → μ *= 0.50
    f_min, _ = kelly_fraction(
        mu=mu, sigma=sigma, rf=rf, fraction=0.25,
        swing_score=0.0, swing_confidence=1.0, momentum_boost_beta=beta,
    )
    f_manual_min, _ = kelly_fraction(mu=mu * 0.50, sigma=sigma, rf=rf, fraction=0.25)
    assert f_min == pytest.approx(f_manual_min, abs=1e-6)


def test_momentum_boost_with_epist_and_impact():
    """Momentum boost composes correctly with σ_epist and market impact."""
    mu, sigma, rf = 0.15, 0.30, 0.04
    # Baseline with epist + impact but no momentum
    f_base, _ = kelly_fraction(
        mu=mu, sigma=sigma, rf=rf, fraction=0.25,
        sigma_epist=0.10, lambda_epist=0.25,
        aum=50_000_000, daily_dollar_volume=20_000_000,
    )
    # Same but with positive momentum
    f_boosted, _ = kelly_fraction(
        mu=mu, sigma=sigma, rf=rf, fraction=0.25,
        sigma_epist=0.10, lambda_epist=0.25,
        aum=50_000_000, daily_dollar_volume=20_000_000,
        swing_score=0.80, swing_confidence=0.85, momentum_boost_beta=0.50,
    )
    assert f_boosted > f_base
    assert f_boosted > 0
    assert f_boosted <= 1.0


def test_momentum_boost_cannot_rescue_below_rf():
    """mu <= rf is checked BEFORE boost — momentum cannot create alpha where fundamentals say none.

    This is intentional (Physical Anchoring, Principle II): momentum modulates
    existing positive excess return, it does not generate it.  A stock whose
    fundamental μ ≤ rf is excluded regardless of momentum strength.
    """
    rf = 0.04
    mu_below_rf = 0.039  # just below risk-free rate
    # Even with maximum momentum boost (swing=1.0, conf=1.0, beta=0.50)
    # which would make μ_boosted = 0.039 * 1.50 = 0.0585 > rf,
    # the position is still killed because the guard fires first.
    f_adj, f_raw = kelly_fraction(
        mu=mu_below_rf, sigma=0.30, rf=rf, fraction=0.25,
        swing_score=1.0, swing_confidence=1.0, momentum_boost_beta=0.50,
    )
    assert f_adj == 0.0
    assert f_raw == 0.0


def test_momentum_boost_discount_kills_marginal_position():
    """Strong negative momentum can push μ_boosted below rf → position correctly killed."""
    mu, sigma, rf = 0.06, 0.30, 0.04
    # Without boost: mu - rf = 0.02 > 0, so position is taken
    f_base, _ = kelly_fraction(mu=mu, sigma=sigma, rf=rf, fraction=0.25)
    assert f_base > 0

    # Max discount: swing=0.0, conf=1.0, beta=0.50 → μ *= 0.50 → μ_boosted = 0.03 < rf
    f_killed, _ = kelly_fraction(
        mu=mu, sigma=sigma, rf=rf, fraction=0.25,
        swing_score=0.0, swing_confidence=1.0, momentum_boost_beta=0.50,
    )
    assert f_killed == 0.0


def test_momentum_boost_clamps_out_of_range_swing_score():
    """swing_score outside [0,1] is clamped — multiplier stays bounded."""
    mu, sigma, rf = 0.15, 0.30, 0.04

    # swing_score=1.5 should be clamped to 1.0 → same as swing_score=1.0
    f_over, _ = kelly_fraction(
        mu=mu, sigma=sigma, rf=rf, fraction=0.25,
        swing_score=1.5, swing_confidence=1.0, momentum_boost_beta=0.50,
    )
    f_max, _ = kelly_fraction(
        mu=mu, sigma=sigma, rf=rf, fraction=0.25,
        swing_score=1.0, swing_confidence=1.0, momentum_boost_beta=0.50,
    )
    assert f_over == pytest.approx(f_max, abs=1e-9)

    # swing_score=-0.5 should be clamped to 0.0 → same as swing_score=0.0
    f_under, _ = kelly_fraction(
        mu=mu, sigma=sigma, rf=rf, fraction=0.25,
        swing_score=-0.5, swing_confidence=1.0, momentum_boost_beta=0.50,
    )
    f_min, _ = kelly_fraction(
        mu=mu, sigma=sigma, rf=rf, fraction=0.25,
        swing_score=0.0, swing_confidence=1.0, momentum_boost_beta=0.50,
    )
    assert f_under == pytest.approx(f_min, abs=1e-9)


def test_momentum_boost_clamps_out_of_range_confidence():
    """swing_confidence outside [0,1] is clamped — multiplier stays bounded."""
    mu, sigma, rf = 0.15, 0.30, 0.04

    # confidence=2.0 should be clamped to 1.0
    f_over, _ = kelly_fraction(
        mu=mu, sigma=sigma, rf=rf, fraction=0.25,
        swing_score=0.85, swing_confidence=2.0, momentum_boost_beta=0.50,
    )
    f_max_conf, _ = kelly_fraction(
        mu=mu, sigma=sigma, rf=rf, fraction=0.25,
        swing_score=0.85, swing_confidence=1.0, momentum_boost_beta=0.50,
    )
    assert f_over == pytest.approx(f_max_conf, abs=1e-9)


def test_momentum_boost_clamps_beta_above_one():
    """beta > 1.0 is clamped to 1.0 — prevents μ negation."""
    mu, sigma, rf = 0.15, 0.30, 0.04

    # beta=2.0 should be clamped to 1.0
    f_beta2, _ = kelly_fraction(
        mu=mu, sigma=sigma, rf=rf, fraction=0.25,
        swing_score=0.85, swing_confidence=1.0, momentum_boost_beta=2.0,
    )
    f_beta1, _ = kelly_fraction(
        mu=mu, sigma=sigma, rf=rf, fraction=0.25,
        swing_score=0.85, swing_confidence=1.0, momentum_boost_beta=1.0,
    )
    assert f_beta2 == pytest.approx(f_beta1, abs=1e-9)


# ────────────────────────────────────────────────────────────────────────────
# Blended σ_epist tests (Fix 5)
# ────────────────────────────────────────────────────────────────────────────

def _make_scored_df(ticker="ETN", composite_confidence=0.80,
                    physical_confidence=0.5, quality_confidence=0.5,
                    crowding_confidence=0.5, mu_estimate=0.15,
                    composite_score=0.60):
    """Helper: minimal scored DataFrame for a single candidate."""
    return pd.DataFrame([{
        "ticker": ticker,
        "entry_signal": True,
        "composite_score": composite_score,
        "composite_confidence": composite_confidence,
        "physical_confidence": physical_confidence,
        "quality_confidence": quality_confidence,
        "crowding_confidence": crowding_confidence,
        "mu_estimate": mu_estimate,
        "swing_score": None,
        "swing_confidence": None,
    }])


def _make_universe_df(ticker="ETN", region="US", bucket="grid"):
    """Helper: minimal universe DataFrame."""
    return pd.DataFrame([{
        "ticker": ticker,
        "region": region,
        "primary_bucket": bucket,
    }])


def _base_params(**kelly_overrides):
    """Minimal params dict for compute_kelly_weights."""
    kelly = {
        "fraction": 0.25,
        "aum_eur": 0,
        "impact_scaling": 1.0,
        "lambda_epist": 0.25,
        "not_aus_confidence": 0.20,
        "eta_participation": 0.0,
        "momentum_boost_beta": 0.0,
        "use_shrunk_cov": False,
        "sigma_epist_model_weight": 0.0,
        "sigma_epist_vol_of_vol_window": 60,
    }
    kelly.update(kelly_overrides)
    return {"kelly": kelly, "macro": {"us_risk_free_series": "US_10Y", "eu_risk_free_series": "EU_10Y_DE"}}


@patch("src.portfolio.kelly.batch_estimate_daily_dollar_volume", return_value={})
@patch("src.portfolio.kelly.batch_get_last_kelly_fractions", return_value={})
@patch("src.data.macro.get_risk_free_rate", return_value=0.04)
@patch("src.portfolio.kelly.estimate_sigma")
def test_sigma_epist_model_weight_zero_matches_old(
    mock_sigma, mock_rf, mock_folds, mock_adv,
):
    """With sigma_epist_model_weight=0.0, behavior matches old proxy: sigma_epist = 1 - comp_conf."""
    mock_sigma.return_value = (0.30, True)
    conn = MagicMock()

    scored = _make_scored_df(composite_confidence=0.80)
    universe = _make_universe_df()
    params = _base_params(sigma_epist_model_weight=0.0)

    result = compute_kelly_weights(scored, conn, params, "2023-06-01", universe)

    assert len(result) == 1
    # sigma_epist should be 1.0 - 0.80 = 0.20 (pure data proxy)
    assert result.iloc[0]["sigma_epist"] == pytest.approx(0.20, abs=1e-4)


@patch("src.portfolio.kelly.batch_estimate_daily_dollar_volume", return_value={})
@patch("src.portfolio.kelly.batch_get_last_kelly_fractions", return_value={})
@patch("src.data.macro.get_risk_free_rate", return_value=0.04)
@patch("src.portfolio.kelly.estimate_sigma")
def test_sigma_epist_blended_higher_than_data_only(
    mock_sigma, mock_rf, mock_folds, mock_adv,
):
    """With model_weight > 0 and disagreeing signals, blended sigma_epist > data-only proxy."""
    # Long-term sigma=0.30, short-term sigma=0.50 → large vol-of-vol mismatch
    # Also: physical_confidence=0.90, quality=0.30, crowding=0.50 → high CV
    mock_sigma.side_effect = lambda ticker, conn, as_of_date, window_days=252: (
        (0.50, True) if window_days == 60 else (0.30, True)
    )
    conn = MagicMock()

    scored = _make_scored_df(
        composite_confidence=0.80,
        physical_confidence=0.90,
        quality_confidence=0.30,
        crowding_confidence=0.50,
    )
    universe = _make_universe_df()

    # Data-only (weight=0)
    params_data = _base_params(sigma_epist_model_weight=0.0)
    result_data = compute_kelly_weights(scored, conn, params_data, "2023-06-01", universe)
    sigma_data_only = result_data.iloc[0]["sigma_epist"]

    # Blended (weight=0.5)
    params_blend = _base_params(sigma_epist_model_weight=0.50)
    result_blend = compute_kelly_weights(scored, conn, params_blend, "2023-06-01", universe)
    sigma_blended = result_blend.iloc[0]["sigma_epist"]

    # Blended should be higher because model uncertainty (from vol mismatch + signal dispersion) > 0
    assert sigma_blended > sigma_data_only


@patch("src.portfolio.kelly.batch_estimate_daily_dollar_volume", return_value={})
@patch("src.portfolio.kelly.batch_get_last_kelly_fractions", return_value={})
@patch("src.data.macro.get_risk_free_rate", return_value=0.04)
@patch("src.portfolio.kelly.estimate_sigma")
def test_sigma_epist_blended_with_agreeing_signals(
    mock_sigma, mock_rf, mock_folds, mock_adv,
):
    """With model_weight > 0 but agreeing signals and stable vol, blended ~ data proxy."""
    # Same vol at both windows → vol_ratio ≈ 0
    mock_sigma.return_value = (0.30, True)
    conn = MagicMock()

    # All confidences equal → CV = 0
    scored = _make_scored_df(
        composite_confidence=0.80,
        physical_confidence=0.60,
        quality_confidence=0.60,
        crowding_confidence=0.60,
    )
    universe = _make_universe_df()

    params = _base_params(sigma_epist_model_weight=0.50)
    result = compute_kelly_weights(scored, conn, params, "2023-06-01", universe)
    sigma_blended = result.iloc[0]["sigma_epist"]

    # With CV=0 and vol_ratio=0, model_uncertainty=0
    # Blend = 0.5 * 0 + 0.5 * 0.20 = 0.10
    sigma_data = 0.20  # 1 - 0.80
    expected = 0.50 * 0.0 + 0.50 * sigma_data  # = 0.10
    assert sigma_blended == pytest.approx(expected, abs=1e-3)


@patch("src.portfolio.kelly.batch_estimate_daily_dollar_volume", return_value={})
@patch("src.portfolio.kelly.batch_get_last_kelly_fractions", return_value={})
@patch("src.data.macro.get_risk_free_rate", return_value=0.04)
@patch("src.portfolio.kelly.estimate_sigma")
def test_sigma_epist_vol_of_vol_dominates(
    mock_sigma, mock_rf, mock_folds, mock_adv,
):
    """When short-term vol diverges significantly from long-term, vol_ratio drives model uncertainty."""
    # sigma_short=0.60, sigma_long=0.30 → vol_ratio = |0.60-0.30|/0.30 = 1.0 (capped)
    mock_sigma.side_effect = lambda ticker, conn, as_of_date, window_days=252: (
        (0.60, True) if window_days == 60 else (0.30, True)
    )
    conn = MagicMock()

    # Equal confidences → CV = 0, but vol_ratio = 1.0
    scored = _make_scored_df(
        composite_confidence=0.90,
        physical_confidence=0.70,
        quality_confidence=0.70,
        crowding_confidence=0.70,
    )
    universe = _make_universe_df()

    params = _base_params(sigma_epist_model_weight=0.50)
    result = compute_kelly_weights(scored, conn, params, "2023-06-01", universe)
    sigma_blended = result.iloc[0]["sigma_epist"]

    # model_uncertainty = max(0, 1.0) = 1.0
    # sigma_data = 0.10
    # blend = 0.5 * 1.0 + 0.5 * 0.10 = 0.55
    assert sigma_blended == pytest.approx(0.55, abs=0.02)
