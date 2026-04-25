"""
tests/test_district.py
========================
Unit tests for DistrictState.

Coverage:
  - to_vector() shape and dtype
  - to_vector() value fidelity
  - is_collapsed property (both states)
  - clip_values() upper and lower bounds
  - to_dict() key completeness
"""

from __future__ import annotations

import numpy as np
import pytest

from district_accord.core.district import DistrictState


class TestDistrictState:
    """Unit tests for DistrictState data model."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make(
        district_id: int = 0,
        resources: float = 0.6,
        stability: float = 0.7,
        crisis_exposure: float = 0.1,
    ) -> DistrictState:
        return DistrictState(
            district_id=district_id,
            resources=resources,
            stability=stability,
            crisis_exposure=crisis_exposure,
        )

    # ------------------------------------------------------------------
    # to_vector()
    # ------------------------------------------------------------------

    def test_to_vector_returns_float32(self):
        d = self.make()
        assert d.to_vector().dtype == np.float32

    def test_to_vector_shape_is_3(self):
        d = self.make()
        assert d.to_vector().shape == (3,)

    def test_to_vector_correct_order(self):
        d = self.make(resources=0.3, stability=0.5, crisis_exposure=0.2)
        vec = d.to_vector()
        np.testing.assert_allclose(vec, [0.3, 0.5, 0.2], rtol=1e-6)

    def test_to_vector_reflects_mutation(self):
        """to_vector() reads live attributes, not a cached copy."""
        d = self.make(resources=0.5)
        d.resources = 0.9
        assert d.to_vector()[0] == pytest.approx(0.9)

    # ------------------------------------------------------------------
    # is_collapsed property
    # ------------------------------------------------------------------

    def test_is_collapsed_false_when_stable(self):
        d = self.make(stability=0.5)
        assert d.is_collapsed is False

    def test_is_collapsed_false_at_threshold_boundary(self):
        """Stability just above zero is NOT collapsed."""
        d = self.make(stability=0.001)
        assert d.is_collapsed is False

    def test_is_collapsed_true_at_zero(self):
        d = self.make(stability=0.0)
        assert d.is_collapsed is True

    def test_is_collapsed_true_when_negative(self):
        """Negative stability (before clipping) is also collapsed."""
        d = self.make(stability=-0.1)
        assert d.is_collapsed is True

    # ------------------------------------------------------------------
    # clip_values()
    # ------------------------------------------------------------------

    def test_clip_values_upper_bound(self):
        d = self.make(resources=2.0, stability=5.0, crisis_exposure=1.5)
        d.clip_values()
        assert d.resources == pytest.approx(1.0)
        assert d.stability == pytest.approx(1.0)
        assert d.crisis_exposure == pytest.approx(1.0)

    def test_clip_values_lower_bound(self):
        d = self.make(resources=-0.5, stability=-1.0, crisis_exposure=-0.3)
        d.clip_values()
        assert d.resources == pytest.approx(0.0)
        assert d.stability == pytest.approx(0.0)
        assert d.crisis_exposure == pytest.approx(0.0)

    def test_clip_values_leaves_valid_values_unchanged(self):
        d = self.make(resources=0.4, stability=0.6, crisis_exposure=0.2)
        d.clip_values()
        assert d.resources == pytest.approx(0.4)
        assert d.stability == pytest.approx(0.6)
        assert d.crisis_exposure == pytest.approx(0.2)

    # ------------------------------------------------------------------
    # to_dict()
    # ------------------------------------------------------------------

    def test_to_dict_has_required_keys(self):
        d = self.make()
        info = d.to_dict()
        assert "district_id" in info
        assert "resources" in info
        assert "stability" in info
        assert "crisis_exposure" in info
        assert "is_collapsed" in info
        assert "coalition_id" in info

    def test_to_dict_values_match_attributes(self):
        d = self.make(district_id=3, resources=0.55, stability=0.45, crisis_exposure=0.15)
        info = d.to_dict()
        assert info["district_id"] == 3
        assert info["resources"] == pytest.approx(0.55)
        assert info["stability"] == pytest.approx(0.45)
        assert info["crisis_exposure"] == pytest.approx(0.15)
        assert info["is_collapsed"] is False

    def test_to_dict_is_collapsed_true_when_zero_stability(self):
        d = self.make(stability=0.0)
        assert d.to_dict()["is_collapsed"] is True
