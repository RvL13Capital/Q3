"""
Tests for Data Maturity Score (Fix 4).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.signals.composite import compute_data_maturity, compute_composite_score


class TestComputeDataMaturity:
    """Unit tests for the DMS computation."""

    def test_full_maturity(self, params):
        """Stock with full data → DMS = 1.0."""
        dms = compute_data_maturity(300, 5, 5, params)
        assert dms == pytest.approx(1.0)

    def test_fresh_ipo_floored(self, params):
        """IPO with 60 price days, 0 fundamentals → DMS = penalty_floor."""
        dms = compute_data_maturity(60, 0, 0, params)
        assert dms == pytest.approx(0.30)  # floor

    def test_partial_maturity(self, params):
        """Stock with partial data → DMS between floor and 1.0."""
        dms = compute_data_maturity(200, 2, 2, params)
        assert 0.30 < dms < 1.0

    def test_geometric_mean_penalizes_zeros(self, params):
        """If any dimension is 0, geometric mean → 0 → floor."""
        dms = compute_data_maturity(300, 0, 5, params)
        assert dms == pytest.approx(0.30)  # geomean with 0 → 0 → floor

    def test_disabled_returns_none_path(self, params):
        """When DMS disabled, composite should use signal_conf only."""
        params_disabled = {**params, "data_maturity": {"enabled": False}}
        p = {"ticker": "TEST", "physical_norm": 0.5, "physical_confidence": 0.8}
        q = {"ticker": "TEST", "quality_score": 0.5, "quality_confidence": 0.8}
        c = {"ticker": "TEST", "crowding_score": 0.3, "crowding_confidence": 0.8}
        result = compute_composite_score(p, q, c, params_disabled, rf=0.04)
        # comp_conf should be simple average: 0.8
        assert result["composite_confidence"] == pytest.approx(0.8, abs=0.01)


class TestDMSCompositeBlend:
    """Test that DMS blends correctly into composite confidence."""

    def test_low_dms_reduces_confidence(self, params):
        """Low DMS (IPO-like) → lower composite confidence than full maturity."""
        p = {"ticker": "TEST", "physical_norm": 0.5, "physical_confidence": 0.8}
        q = {"ticker": "TEST", "quality_score": 0.5, "quality_confidence": 0.8}
        c = {"ticker": "TEST", "crowding_score": 0.3, "crowding_confidence": 0.8}

        result_mature = compute_composite_score(p, q, c, params, rf=0.04, data_maturity=1.0)
        result_ipo = compute_composite_score(p, q, c, params, rf=0.04, data_maturity=0.30)

        assert result_ipo["composite_confidence"] < result_mature["composite_confidence"]

    def test_full_maturity_no_penalty(self, params):
        """Full maturity (1.0) → composite confidence ≈ signal_conf."""
        p = {"ticker": "TEST", "physical_norm": 0.5, "physical_confidence": 0.8}
        q = {"ticker": "TEST", "quality_score": 0.5, "quality_confidence": 0.8}
        c = {"ticker": "TEST", "crowding_score": 0.3, "crowding_confidence": 0.8}
        result = compute_composite_score(p, q, c, params, rf=0.04, data_maturity=1.0)
        # signal_conf = 0.8. Blend: 0.70*0.8 + 0.30*1.0 = 0.86
        assert result["composite_confidence"] == pytest.approx(0.86, abs=0.01)
