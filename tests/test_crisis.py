"""
tests/test_crisis.py
======================
Unit tests for CrisisSystem.

Coverage:
  - reset() initialises crisis_level within [0, 1]
  - reset() seeds history correctly
  - step() advances and bounds crisis_level
  - step() accumulates history
  - tier property maps correctly at all 5 thresholds
  - compute_exposure() equals crisis_level (Phase 1 invariant)
  - to_vector() shape and dtype
  - to_dict() keys and types
"""

from __future__ import annotations

import numpy as np
import pytest

from district_accord.core.crisis import CrisisSystem
from district_accord.utils.config import EnvConfig
from district_accord.utils.types import CrisisTier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_crisis(
    seed: int = 42,
    crisis_init_mean: float = 0.15,
    crisis_init_std: float = 0.05,
    crisis_drift: float = 0.02,
    crisis_noise_std: float = 0.03,
) -> CrisisSystem:
    config = EnvConfig(
        seed=seed,
        crisis_init_mean=crisis_init_mean,
        crisis_init_std=crisis_init_std,
        crisis_drift=crisis_drift,
        crisis_noise_std=crisis_noise_std,
    )
    rng = np.random.default_rng(seed)
    crisis = CrisisSystem(config, rng)
    crisis.reset()
    return crisis


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestCrisisReset:
    def test_crisis_level_in_unit_interval(self):
        crisis = make_crisis()
        assert 0.0 <= crisis.crisis_level <= 1.0

    def test_history_has_one_entry_after_reset(self):
        crisis = make_crisis()
        assert len(crisis.history) == 1

    def test_history_first_entry_equals_crisis_level(self):
        crisis = make_crisis()
        assert crisis.history[0] == pytest.approx(crisis.crisis_level)

    def test_reset_clears_previous_history(self):
        crisis = make_crisis()
        for _ in range(5):
            crisis.step()
        crisis.reset()
        assert len(crisis.history) == 1

    def test_repeated_reset_with_same_seed_gives_same_level(self):
        """Deterministic: same seed → same initial crisis_level."""
        config = EnvConfig(seed=7, crisis_init_mean=0.2, crisis_init_std=0.05)
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        c1 = CrisisSystem(config, rng1)
        c2 = CrisisSystem(config, rng2)
        c1.reset()
        c2.reset()
        assert c1.crisis_level == pytest.approx(c2.crisis_level)


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------

class TestCrisisStep:
    def test_crisis_level_bounded_after_step(self):
        crisis = make_crisis()
        for _ in range(20):
            crisis.step()
        assert 0.0 <= crisis.crisis_level <= 1.0

    def test_history_length_after_n_steps(self):
        crisis = make_crisis()
        n = 10
        for _ in range(n):
            crisis.step()
        # history[0] = reset value, history[1..n] = after each step
        assert len(crisis.history) == n + 1

    def test_step_returns_crisis_level(self):
        crisis = make_crisis()
        returned = crisis.step()
        assert returned == pytest.approx(crisis.crisis_level)

    def test_step_generally_increases_with_positive_drift(self):
        """
        With drift=0.1 and low noise, crisis level should trend upward.
        We run 20 steps and expect the final level to exceed the initial.
        """
        crisis = make_crisis(crisis_drift=0.1, crisis_noise_std=0.001, seed=0)
        initial = crisis.crisis_level
        for _ in range(20):
            crisis.step()
        assert crisis.crisis_level > initial

    def test_history_is_defensive_copy(self):
        """Mutating the returned list must not alter internal history."""
        crisis = make_crisis()
        crisis.step()
        history = crisis.history
        original_len = len(history)
        history.append(99.0)
        assert len(crisis.history) == original_len


# ---------------------------------------------------------------------------
# tier property
# ---------------------------------------------------------------------------

class TestCrisisTier:
    def _crisis_at(self, level: float) -> CrisisSystem:
        crisis = make_crisis()
        crisis.crisis_level = level
        return crisis

    def test_tier_calm_below_0_2(self):
        assert self._crisis_at(0.0).tier == CrisisTier.CALM
        assert self._crisis_at(0.19).tier == CrisisTier.CALM

    def test_tier_elevated_0_2_to_0_4(self):
        assert self._crisis_at(0.2).tier == CrisisTier.ELEVATED
        assert self._crisis_at(0.39).tier == CrisisTier.ELEVATED

    def test_tier_critical_0_4_to_0_6(self):
        assert self._crisis_at(0.4).tier == CrisisTier.CRITICAL
        assert self._crisis_at(0.59).tier == CrisisTier.CRITICAL

    def test_tier_emergency_0_6_to_0_8(self):
        assert self._crisis_at(0.6).tier == CrisisTier.EMERGENCY
        assert self._crisis_at(0.79).tier == CrisisTier.EMERGENCY

    def test_tier_collapse_at_0_8_and_above(self):
        assert self._crisis_at(0.8).tier == CrisisTier.COLLAPSE
        assert self._crisis_at(1.0).tier == CrisisTier.COLLAPSE


# ---------------------------------------------------------------------------
# compute_exposure()
# ---------------------------------------------------------------------------

class TestCrisisExposure:
    def test_exposure_equals_crisis_level(self):
        """Phase 1 invariant: all districts share the same global exposure."""
        crisis = make_crisis()
        for district_id in range(12):
            assert crisis.compute_exposure(district_id) == pytest.approx(
                crisis.crisis_level
            )

    def test_exposure_updates_after_step(self):
        crisis = make_crisis()
        initial_exposure = crisis.compute_exposure(0)
        crisis.step()
        # After a step, crisis_level may have changed.
        # The exposure must equal the *new* crisis_level, not the old one.
        assert crisis.compute_exposure(0) == pytest.approx(crisis.crisis_level)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestCrisisSerialisation:
    def test_to_vector_shape(self):
        crisis = make_crisis()
        assert crisis.to_vector().shape == (1,)

    def test_to_vector_dtype(self):
        crisis = make_crisis()
        assert crisis.to_vector().dtype == np.float32

    def test_to_vector_value_matches_crisis_level(self):
        crisis = make_crisis()
        np.testing.assert_allclose(
            crisis.to_vector(), [crisis.crisis_level], rtol=1e-6
        )

    def test_to_dict_required_keys(self):
        crisis = make_crisis()
        d = crisis.to_dict()
        assert "crisis_level" in d
        assert "tier" in d
        assert "tier_value" in d

    def test_to_dict_tier_is_string(self):
        crisis = make_crisis()
        assert isinstance(crisis.to_dict()["tier"], str)

    def test_to_dict_tier_value_is_int(self):
        crisis = make_crisis()
        assert isinstance(crisis.to_dict()["tier_value"], int)
