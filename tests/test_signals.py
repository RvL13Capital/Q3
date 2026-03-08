"""
Unit tests for signal computation.
Uses synthetic data — no external API calls, no DuckDB.
"""
import math
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ────────────────────────────────────────────────────────────────────────────
# Physical score tests
# ────────────────────────────────────────────────────────────────────────────

from src.signals.physical import compute_physical_score, batch_physical_scores


def test_physical_no_bucket():
    result = compute_physical_score({"ticker": "X", "buckets": [], "primary_bucket": None})
    assert result["physical_raw"] == 0.0
    assert result["physical_norm"] == 0.0
    assert result["physical_confidence"] == 1.0


def test_physical_single_bucket():
    result = compute_physical_score({"ticker": "RHM.DE", "buckets": ["defense"], "primary_bucket": "defense"})
    assert result["physical_raw"] == 1.5
    assert result["physical_norm"] == pytest.approx(0.5)
    assert result["physical_confidence"] == 1.0


def test_physical_two_buckets():
    result = compute_physical_score({"ticker": "ENR.DE", "buckets": ["grid", "critical_materials"], "primary_bucket": "grid"})
    assert result["physical_raw"] == 2.0
    assert result["physical_norm"] == pytest.approx(2.0 / 3.0)


def test_physical_three_or_more_buckets():
    result = compute_physical_score({"ticker": "X", "buckets": ["grid", "nuclear", "defense"], "primary_bucket": "grid"})
    assert result["physical_raw"] == 3.0
    assert result["physical_norm"] == 1.0


def test_physical_deduplicates_buckets():
    """Duplicate bucket entries should not inflate the count."""
    result = compute_physical_score({"ticker": "X", "buckets": ["grid", "grid", "grid"], "primary_bucket": "grid"})
    assert result["physical_raw"] == 1.5  # only 1 unique bucket


def test_batch_physical():
    universe = pd.DataFrame([
        {"ticker": "A", "buckets": ["defense"], "primary_bucket": "defense"},
        {"ticker": "B", "buckets": ["grid", "nuclear"], "primary_bucket": "grid"},
    ])
    df = batch_physical_scores(universe)
    assert len(df) == 2
    assert df[df["ticker"] == "A"]["physical_norm"].iloc[0] == pytest.approx(0.5)
    assert df[df["ticker"] == "B"]["physical_norm"].iloc[0] == pytest.approx(2.0 / 3.0)


# ────────────────────────────────────────────────────────────────────────────
# Quality sub-score normalization tests (no DB needed)
# ────────────────────────────────────────────────────────────────────────────

from src.signals.quality import _clamp_normalize


def test_clamp_normalize_below_low():
    assert _clamp_normalize(-0.20, -0.10, 0.20) == 0.0


def test_clamp_normalize_above_high():
    assert _clamp_normalize(0.30, -0.10, 0.20) == 1.0


def test_clamp_normalize_midpoint():
    result = _clamp_normalize(0.05, -0.10, 0.20)
    # (0.05 - (-0.10)) / (0.20 - (-0.10)) = 0.15 / 0.30 = 0.5
    assert result == pytest.approx(0.5)


def test_roic_wacc_spread_example():
    """ROIC=15%, WACC≈9.3% → spread=5.7pp → norm≈0.52."""
    spread = 0.15 - 0.093
    result = _clamp_normalize(spread, -0.10, 0.20)
    assert 0.45 < result < 0.65  # expect around 0.52


def test_roic_wacc_negative_spread():
    """ROIC=5%, WACC≈9.3% → spread=-4.3pp → norm≈0.19."""
    spread = 0.05 - 0.093
    result = _clamp_normalize(spread, -0.10, 0.20)
    assert 0.10 < result < 0.30


def test_margin_snr_stable():
    """Five stable years: high SNR → score close to 1."""
    margins = [0.42, 0.43, 0.41, 0.44, 0.42]
    mean_gm = sum(margins) / len(margins)
    std_gm  = pd.Series(margins).std()
    snr = mean_gm / std_gm
    score = _clamp_normalize(snr, 2.0, 10.0)
    assert score > 0.7  # very stable margins → high score


def test_margin_snr_volatile():
    """Volatile margins: low SNR → low score."""
    margins = [0.10, 0.50, 0.20, 0.45, 0.05]
    mean_gm = sum(margins) / len(margins)
    std_gm  = pd.Series(margins).std()
    snr = abs(mean_gm) / std_gm if std_gm > 0 else 0
    score = _clamp_normalize(snr, 2.0, 10.0)
    assert score < 0.5


def test_inflation_convexity_positive():
    """∂GM/∂PPI = +0.4 (strong anti-fragile) → high score on [-0.5, +0.5] range."""
    convexity = 0.40   # each 1% PPI rise → 0.4% GM expansion
    score = _clamp_normalize(convexity, -0.50, 0.50)
    assert score > 0.7   # (0.40 + 0.50) / 1.0 = 0.90


def test_inflation_convexity_negative():
    """∂GM/∂PPI = -0.4 (margin compression) → low score on [-0.5, +0.5] range."""
    convexity = -0.40
    score = _clamp_normalize(convexity, -0.50, 0.50)
    assert score < 0.3   # (-0.40 + 0.50) / 1.0 = 0.10


# ────────────────────────────────────────────────────────────────────────────
# Composite score tests
# ────────────────────────────────────────────────────────────────────────────

from src.signals.composite import compute_composite_score, MIN_COMPOSITE_CONFIDENCE

PARAMS = {
    "signals": {
        "entry_threshold":         0.30,
        "crowding_entry_max":      0.40,
        "crowding_exit_threshold": 0.75,
        "quality_exit_threshold":  0.25,
        "composite_decay_pct":     0.20,
    },
    "return_estimation": {
        "equity_risk_premium": 0.05,
        "theta_risk_premium":  0.30,
    },
}


def test_composite_full_confidence_entry():
    """All signals available, strong scores → entry signal.
    composite = X_E × X_P × (1−X_C) = 0.67 × 0.75 × 0.75 ≈ 0.377 > threshold 0.30.
    """
    physical = {"ticker": "T", "physical_norm": 0.67, "physical_confidence": 1.0}
    quality  = {"ticker": "T", "quality_score": 0.75,  "quality_confidence": 1.0}
    crowding = {"ticker": "T", "crowding_score": 0.25, "crowding_confidence": 1.0}
    result = compute_composite_score(physical, quality, crowding, PARAMS, rf=0.04)
    assert result["composite_score"] is not None
    assert result["composite_score"] > 0.30   # entry_threshold = 0.30
    assert result["entry_signal"] is True


def test_composite_confidence_gate():
    """If composite_confidence < 0.40, composite_score should be None."""
    physical = {"ticker": "T", "physical_norm": 0.0, "physical_confidence": 0.0}
    quality  = {"ticker": "T", "quality_score": 0.0,  "quality_confidence": 0.0}
    crowding = {"ticker": "T", "crowding_score": 0.5, "crowding_confidence": 1.0}
    # avg conf = (0 + 0 + 1) / 3 = 0.33 < 0.40 → gate
    result = compute_composite_score(physical, quality, crowding, PARAMS, rf=0.04)
    assert result["composite_score"] is None
    assert result["entry_signal"] is False


def test_composite_crowding_exit_trigger():
    """High crowding alone should trigger exit_signal."""
    physical = {"ticker": "T", "physical_norm": 0.67, "physical_confidence": 1.0}
    quality  = {"ticker": "T", "quality_score": 0.80,  "quality_confidence": 1.0}
    crowding = {"ticker": "T", "crowding_score": 0.80, "crowding_confidence": 1.0}
    result = compute_composite_score(physical, quality, crowding, PARAMS, rf=0.04)
    assert result["exit_signal"] is True
    assert result["entry_signal"] is False  # crowding > entry_max


def test_composite_quality_exit_trigger():
    """Quality below exit threshold should trigger exit_signal."""
    physical = {"ticker": "T", "physical_norm": 0.67, "physical_confidence": 1.0}
    quality  = {"ticker": "T", "quality_score": 0.20,  "quality_confidence": 1.0}
    crowding = {"ticker": "T", "crowding_score": 0.30, "crowding_confidence": 1.0}
    result = compute_composite_score(physical, quality, crowding, PARAMS, rf=0.04)
    assert result["exit_signal"] is True


def test_composite_no_action_zone():
    """Moderate scores, no threshold crossings → no entry, no exit."""
    physical = {"ticker": "T", "physical_norm": 0.50, "physical_confidence": 1.0}
    quality  = {"ticker": "T", "quality_score": 0.45,  "quality_confidence": 1.0}
    crowding = {"ticker": "T", "crowding_score": 0.45, "crowding_confidence": 1.0}
    result = compute_composite_score(physical, quality, crowding, PARAMS, rf=0.04)
    assert result["entry_signal"] is False
    assert result["exit_signal"] is False


def test_composite_missing_quality_no_crowding_data():
    """
    If quality and crowding have zero confidence but physical is full confidence,
    composite_confidence = (1 + 0 + 0) / 3 ≈ 0.33 < 0.40 → gate → None.
    """
    physical = {"ticker": "T", "physical_norm": 1.0, "physical_confidence": 1.0}
    quality  = {"ticker": "T", "quality_score": 0.0,  "quality_confidence": 0.0}
    crowding = {"ticker": "T", "crowding_score": 0.5, "crowding_confidence": 0.0}
    result = compute_composite_score(physical, quality, crowding, PARAMS, rf=0.04)
    assert result["composite_score"] is None


def test_mu_estimate_interpolation():
    """Entry threshold → mu ≈ rf + 6%."""
    physical = {"ticker": "T", "physical_norm": 0.67, "physical_confidence": 1.0}
    quality  = {"ticker": "T", "quality_score": 0.75,  "quality_confidence": 1.0}
    crowding = {"ticker": "T", "crowding_score": 0.20, "crowding_confidence": 1.0}
    result = compute_composite_score(physical, quality, crowding, PARAMS, rf=0.04)
    # composite should be above 0.55 for entry → mu should be > 0.10
    if result["mu_estimate"]:
        assert result["mu_estimate"] > 0.04 + 0.05  # at minimum rf + some excess
