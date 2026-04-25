"""
tests/test_phase3_trust.py
============================
Unit tests for TrustSystem.

Covers:
    - Initialisation (matrix size, value range, self-trust)
    - update_accept (bilateral, bounded)
    - update_reject (asymmetric)
    - update_betrayal
    - decay (per-turn damping toward 0)
    - get / matrix / as_matrix accessors
    - Value bounds always [-1, 1]
"""

import numpy as np
import pytest

from district_accord.core.trust import TrustSystem
from district_accord.utils.config import EnvConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_trust(
    n: int = 4,
    seed: int = 42,
    init_mean: float = 0.5,
    accept_bonus: float = 0.10,
    reject_penalty: float = 0.05,
    betrayal_penalty: float = 0.20,
    decay: float = 0.99,
) -> TrustSystem:
    cfg = EnvConfig(
        trust_init_mean=init_mean,
        trust_init_std=0.0,   # zero std → all start at init_mean (deterministic)
        trust_accept_bonus=accept_bonus,
        trust_reject_penalty=reject_penalty,
        trust_betrayal_penalty=betrayal_penalty,
        trust_decay=decay,
    )
    rng = np.random.default_rng(seed)
    ts = TrustSystem(cfg, rng)
    ts.reset(n)
    return ts


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_matrix_has_all_agents(self):
        ts = make_trust(4)
        mat = ts.as_matrix()
        assert set(mat.keys()) == {0, 1, 2, 3}

    def test_initial_values_in_range(self):
        """Use non-zero std to test clamping."""
        cfg = EnvConfig(trust_init_mean=0.5, trust_init_std=0.3)
        rng = np.random.default_rng(7)
        ts = TrustSystem(cfg, rng)
        ts.reset(6)
        mat = ts.as_matrix()
        for i, row in mat.items():
            for j, v in row.items():
                assert -1.0 <= v <= 1.0, f"trust[{i}][{j}]={v} out of bounds"

    def test_self_trust_is_zero(self):
        ts = make_trust(4)
        for i in range(4):
            assert ts.get(i, i) == pytest.approx(0.0)

    def test_initial_cross_trust_is_init_mean(self):
        ts = make_trust(4, init_mean=0.5)
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert ts.get(i, j) == pytest.approx(0.5)

    def test_reset_clears_previous_state(self):
        ts = make_trust(4)
        ts.update_accept(0, 1)
        ts.reset(4)
        assert ts.get(0, 1) == pytest.approx(0.5)  # back to init_mean


# ---------------------------------------------------------------------------
# update_accept
# ---------------------------------------------------------------------------

class TestUpdateAccept:
    def test_accept_increases_bilateral_trust(self):
        ts = make_trust(4, init_mean=0.5, accept_bonus=0.10)
        ts.update_accept(0, 1)
        assert ts.get(0, 1) == pytest.approx(0.60)
        assert ts.get(1, 0) == pytest.approx(0.60)

    def test_accept_symmetric(self):
        ts = make_trust(4, init_mean=0.3, accept_bonus=0.15)
        ts.update_accept(0, 2)
        assert ts.get(0, 2) == pytest.approx(ts.get(2, 0))

    def test_accept_capped_at_1(self):
        ts = make_trust(4, init_mean=0.95, accept_bonus=0.10)
        ts.update_accept(0, 1)
        assert ts.get(0, 1) == pytest.approx(1.0)
        assert ts.get(1, 0) == pytest.approx(1.0)

    def test_accept_does_not_affect_third_party(self):
        ts = make_trust(4, init_mean=0.5)
        ts.update_accept(0, 1)
        assert ts.get(0, 2) == pytest.approx(0.5)
        assert ts.get(2, 0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# update_reject
# ---------------------------------------------------------------------------

class TestUpdateReject:
    def test_reject_unilateral_rejector_toward_proposer(self):
        ts = make_trust(4, init_mean=0.5, reject_penalty=0.05)
        # rejector=1 refuses proposer=0
        ts.update_reject(rejector=1, proposer=0)
        # rejector's trust of proposer decreases
        assert ts.get(1, 0) == pytest.approx(0.45)
        # proposer's trust of rejector unchanged
        assert ts.get(0, 1) == pytest.approx(0.5)

    def test_reject_floored_at_minus_1(self):
        ts = make_trust(4, init_mean=-0.98, reject_penalty=0.05)
        ts.update_reject(rejector=0, proposer=1)
        assert ts.get(0, 1) >= -1.0


# ---------------------------------------------------------------------------
# update_betrayal
# ---------------------------------------------------------------------------

class TestUpdateBetrayal:
    def test_betrayal_decreases_victim_trust_of_betrayer(self):
        ts = make_trust(4, init_mean=0.5, betrayal_penalty=0.20)
        ts.update_betrayal(victim=0, betrayer=1)
        assert ts.get(0, 1) == pytest.approx(0.30)

    def test_betrayal_does_not_affect_betrayer_trust_of_victim(self):
        ts = make_trust(4, init_mean=0.5)
        ts.update_betrayal(victim=0, betrayer=1)
        assert ts.get(1, 0) == pytest.approx(0.5)  # unchanged

    def test_betrayal_floored_at_minus_1(self):
        ts = make_trust(4, init_mean=-0.85, betrayal_penalty=0.20)
        ts.update_betrayal(victim=0, betrayer=1)
        assert ts.get(0, 1) >= -1.0


# ---------------------------------------------------------------------------
# decay
# ---------------------------------------------------------------------------

class TestDecay:
    def test_positive_trust_decays_toward_zero(self):
        ts = make_trust(4, init_mean=1.0, decay=0.90)
        ts.decay()
        # 1.0 * 0.90 = 0.90
        assert ts.get(0, 1) == pytest.approx(0.90, abs=1e-5)

    def test_negative_trust_decays_toward_zero(self):
        ts = make_trust(4, init_mean=-0.5, decay=0.90)
        ts.decay()
        # -0.5 * 0.90 = -0.45
        assert ts.get(0, 1) == pytest.approx(-0.45, abs=1e-5)

    def test_self_trust_unaffected_by_decay(self):
        ts = make_trust(4, init_mean=0.5, decay=0.90)
        ts.decay()
        for i in range(4):
            assert ts.get(i, i) == pytest.approx(0.0)

    def test_decay_multiple_turns(self):
        ts = make_trust(4, init_mean=1.0, decay=0.99)
        for _ in range(100):
            ts.decay()
        # 1.0 * 0.99^100 ≈ 0.366
        assert ts.get(0, 1) == pytest.approx(0.99 ** 100, abs=1e-4)


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

class TestAccessors:
    def test_get_unknown_pair_returns_init_mean(self):
        ts = make_trust(4, init_mean=0.5)
        # Get a pair that is out of range (new agent)
        assert ts.get(99, 0) == pytest.approx(0.5)

    def test_matrix_returns_row(self):
        ts = make_trust(4, init_mean=0.5)
        row = ts.matrix(0)
        assert set(row.keys()) == {0, 1, 2, 3}

    def test_as_matrix_all_agents(self):
        ts = make_trust(4, init_mean=0.5)
        m = ts.as_matrix()
        assert len(m) == 4
        for row in m.values():
            assert len(row) == 4

    def test_to_dict_serialisable(self):
        ts = make_trust(4)
        d = ts.to_dict()
        assert "trust_matrix" in d
        # Keys should be strings
        assert "0" in d["trust_matrix"]
