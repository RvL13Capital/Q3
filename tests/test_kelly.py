"""
Unit tests for Kelly sizing and portfolio construction.
No external API calls or DuckDB needed.
"""
import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.portfolio.kelly import kelly_fraction
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
