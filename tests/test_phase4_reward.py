"""
tests/test_phase4_reward.py
=============================
Unit tests for RewardEngine and RewardBreakdown (Phase 4).

Tests every component independently, then tests bounds, determinism,
and integration with the environment's info["reward_breakdown"] feed.
"""

from __future__ import annotations

import pytest
import numpy as np

from district_accord.engine.reward import RewardBreakdown, RewardEngine
from district_accord.utils.config import EnvConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_engine(
    survival_reward: float = 1.0,
    stability_weight: float = 1.0,
    crisis_weight: float = 1.0,
    coop_per_peer: float = 0.05,
    trust_alignment: float = 0.02,
    mask_penalty: float = -0.5,
    spam_penalty: float = 0.0,
    collapse_penalty: float = -10.0,
    max_pending: int = 2,
) -> RewardEngine:
    cfg = EnvConfig(
        survival_reward=survival_reward,
        reward_stability_weight=stability_weight,
        reward_crisis_weight=crisis_weight,
        reward_cooperation_per_peer=coop_per_peer,
        reward_trust_alignment=trust_alignment,
        mask_violation_penalty=mask_penalty,
        reward_spam_penalty=spam_penalty,
        collapse_penalty=collapse_penalty,
        max_pending_proposals=max_pending,
    )
    return RewardEngine(cfg)


def base_kwargs(**overrides):
    """Sensible defaults for RewardEngine.compute() calls."""
    defaults = dict(
        agent_id=0,
        is_newly_collapsed=False,
        is_collapsed=False,
        prev_stability=0.60,
        curr_stability=0.60,
        prev_exposure=0.10,   # Phase 6: same as curr_exposure → delta = 0
        curr_exposure=0.10,
        prev_avg_trust=0.0,   # Phase 6: no prior trust baseline by default
        coalition_size=0,
        trust_row={1: 0.5, 2: 0.5, 3: 0.5},
        mask_violated=False,
        pending_outgoing=0,
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# RewardBreakdown
# ---------------------------------------------------------------------------

class TestRewardBreakdown:
    def test_total_equals_sum_of_components(self):
        bd = RewardBreakdown(
            agent_id=0,
            survival=1.0,
            stability_delta=0.05,
            crisis_mitigation=-0.10,
            cooperation=0.05,
            trust_alignment=0.01,
            mask_penalty=0.0,
            spam_penalty=0.0,
            collapse_penalty=0.0,
        )
        expected = 1.0 + 0.05 - 0.10 + 0.05 + 0.01
        assert bd.total == pytest.approx(expected)

    def test_to_dict_has_all_keys(self):
        bd = RewardBreakdown(agent_id=0)
        d = bd.to_dict()
        for key in (
            "survival", "stability_delta", "crisis_mitigation",
            "cooperation", "trust_alignment", "mask_penalty",
            "spam_penalty", "collapse_penalty", "total",
        ):
            assert key in d

    def test_to_dict_total_matches_property(self):
        bd = RewardBreakdown(agent_id=0, survival=1.0, crisis_mitigation=-0.2)
        d = bd.to_dict()
        assert d["total"] == pytest.approx(bd.total, abs=1e-4)

    def test_zeroed_breakdown_total_zero(self):
        bd = RewardBreakdown(agent_id=0)
        assert bd.total == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# A. Survival
# ---------------------------------------------------------------------------

class TestSurvivalComponent:
    def test_survival_reward_present_when_alive(self):
        engine = make_engine(survival_reward=1.0)
        r, bd = engine.compute(**base_kwargs())
        assert bd.survival == pytest.approx(1.0)

    def test_survival_configurable(self):
        engine = make_engine(survival_reward=2.5)
        r, bd = engine.compute(**base_kwargs())
        assert bd.survival == pytest.approx(2.5)

    def test_no_survival_on_collapse(self):
        engine = make_engine()
        r, bd = engine.compute(**base_kwargs(is_newly_collapsed=True))
        assert bd.survival == pytest.approx(0.0)

    def test_no_survival_when_already_dead(self):
        engine = make_engine()
        r, bd = engine.compute(**base_kwargs(is_collapsed=True))
        assert bd.survival == pytest.approx(0.0)
        assert r == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# B. Stability delta
# ---------------------------------------------------------------------------

class TestStabilityDelta:
    def test_stability_increase_positive_signal(self):
        engine = make_engine(stability_weight=1.0)
        r, bd = engine.compute(**base_kwargs(prev_stability=0.50, curr_stability=0.60))
        assert bd.stability_delta == pytest.approx(0.10, abs=1e-4)

    def test_stability_decrease_negative_signal(self):
        engine = make_engine(stability_weight=1.0)
        r, bd = engine.compute(**base_kwargs(prev_stability=0.70, curr_stability=0.60))
        assert bd.stability_delta == pytest.approx(-0.10, abs=1e-4)

    def test_no_stability_change_zero_delta(self):
        engine = make_engine()
        r, bd = engine.compute(**base_kwargs(prev_stability=0.60, curr_stability=0.60))
        assert bd.stability_delta == pytest.approx(0.0)

    def test_stability_delta_clipped_upper(self):
        """Single-turn spike of +0.9 is clipped to +0.5."""
        engine = make_engine(stability_weight=1.0)
        r, bd = engine.compute(**base_kwargs(prev_stability=0.0, curr_stability=0.9))
        assert bd.stability_delta == pytest.approx(0.5)

    def test_stability_delta_clipped_lower(self):
        """Single-turn drop of -0.9 is clipped to -0.5."""
        engine = make_engine(stability_weight=1.0)
        r, bd = engine.compute(**base_kwargs(prev_stability=0.9, curr_stability=0.0))
        assert bd.stability_delta == pytest.approx(-0.5)

    def test_stability_weight_applied(self):
        engine = make_engine(stability_weight=2.0)
        r, bd = engine.compute(**base_kwargs(prev_stability=0.50, curr_stability=0.60))
        # 2.0 * 0.10 = 0.20, within clip
        assert bd.stability_delta == pytest.approx(0.20, abs=1e-4)


# ---------------------------------------------------------------------------
# C. Crisis mitigation
# ---------------------------------------------------------------------------

class TestCrisisMitigation:
    def test_high_exposure_negative_signal(self):
        """Exposure rises from 0 → 0.5: delta = +0.5 → crisis_mitigation = -0.5."""
        engine = make_engine(crisis_weight=1.0)
        r, bd = engine.compute(**base_kwargs(prev_exposure=0.0, curr_exposure=0.5))
        assert bd.crisis_mitigation == pytest.approx(-0.5)

    def test_zero_exposure_delta_zero_penalty(self):
        """No change in exposure → zero crisis signal."""
        engine = make_engine()
        r, bd = engine.compute(**base_kwargs(prev_exposure=0.0, curr_exposure=0.0))
        assert bd.crisis_mitigation == pytest.approx(0.0)

    def test_full_exposure_rise_max_penalty(self):
        """Exposure rises from 0 → 1.0: delta = +1.0 → crisis_mitigation = -0.5 (clipped)."""
        engine = make_engine(crisis_weight=1.0)
        r, bd = engine.compute(**base_kwargs(prev_exposure=0.0, curr_exposure=1.0))
        assert bd.crisis_mitigation == pytest.approx(-0.5)  # clipped at _DELTA_CAP

    def test_exposure_decrease_positive_signal(self):
        """DEFEND reduced exposure → positive crisis_mitigation."""
        engine = make_engine(crisis_weight=1.0)
        r, bd = engine.compute(**base_kwargs(prev_exposure=0.6, curr_exposure=0.2))
        assert bd.crisis_mitigation > 0.0

    def test_crisis_weight_applied(self):
        """Weight scales the delta."""
        engine = make_engine(crisis_weight=0.5)
        r, bd = engine.compute(**base_kwargs(prev_exposure=0.0, curr_exposure=0.4))
        assert bd.crisis_mitigation == pytest.approx(-0.2)


# ---------------------------------------------------------------------------
# D. Cooperation
# ---------------------------------------------------------------------------

class TestCooperation:
    def test_no_coalition_zero_bonus(self):
        engine = make_engine(coop_per_peer=0.05)
        r, bd = engine.compute(**base_kwargs(coalition_size=0))
        assert bd.cooperation == pytest.approx(0.0)

    def test_solo_in_coalition_zero_bonus(self):
        engine = make_engine(coop_per_peer=0.05)
        r, bd = engine.compute(**base_kwargs(coalition_size=1))
        assert bd.cooperation == pytest.approx(0.0)

    def test_two_member_coalition_one_peer(self):
        engine = make_engine(coop_per_peer=0.05)
        r, bd = engine.compute(**base_kwargs(coalition_size=2))
        assert bd.cooperation == pytest.approx(0.05)

    def test_four_member_coalition_three_peers(self):
        engine = make_engine(coop_per_peer=0.05)
        r, bd = engine.compute(**base_kwargs(coalition_size=4))
        assert bd.cooperation == pytest.approx(0.15)  # capped at 0.15

    def test_large_coalition_capped(self):
        """Even 10 peers can't exceed the hard cap."""
        engine = make_engine(coop_per_peer=0.05)
        r, bd = engine.compute(**base_kwargs(coalition_size=50))
        assert bd.cooperation == pytest.approx(0.15)

    def test_cooperation_increases_reward(self):
        engine = make_engine()
        _, bd_solo  = engine.compute(**base_kwargs(coalition_size=0))
        _, bd_pair  = engine.compute(**base_kwargs(coalition_size=2))
        assert bd_pair.total > bd_solo.total


# ---------------------------------------------------------------------------
# E. Trust alignment
# ---------------------------------------------------------------------------

class TestTrustAlignment:
    def test_zero_weight_zero_signal(self):
        engine = make_engine(trust_alignment=0.0)
        r, bd = engine.compute(**base_kwargs(trust_row={1: 0.8, 2: 0.9, 3: 0.7}, prev_avg_trust=0.0))
        assert bd.trust_alignment == pytest.approx(0.0)

    def test_trust_improvement_positive_signal(self):
        """Trust went from 0 → 1.0 (all peers) → delta positive."""
        engine = make_engine(trust_alignment=0.02)
        r, bd = engine.compute(**base_kwargs(
            trust_row={1: 1.0, 2: 1.0, 3: 1.0},
            prev_avg_trust=0.0,   # before step: no positive trust
        ))
        # delta = 1.0 - 0.0 = 1.0; 0.02 * 1.0 = 0.02, capped at 0.05
        assert bd.trust_alignment == pytest.approx(0.02)

    def test_negative_trust_not_rewarded(self):
        """Negative trust values don't contribute (max(v, 0))."""
        engine = make_engine(trust_alignment=0.02)
        r, bd = engine.compute(**base_kwargs(
            trust_row={1: -1.0, 2: -0.5, 3: -0.9},
            prev_avg_trust=0.0,
        ))
        assert bd.trust_alignment == pytest.approx(0.0)

    def test_stable_trust_gives_zero(self):
        """Trust didn't change → delta = 0 → trust_alignment = 0."""
        engine = make_engine(trust_alignment=0.10)
        val = 0.6
        r, bd = engine.compute(**base_kwargs(
            trust_row={1: val, 2: val, 3: val},
            prev_avg_trust=val,
        ))
        assert bd.trust_alignment == pytest.approx(0.0, abs=1e-9)

    def test_trust_hard_cap(self):
        """High weight + large improvement can't exceed 0.05."""
        engine = make_engine(trust_alignment=1.0)
        r, bd = engine.compute(**base_kwargs(
            trust_row={1: 1.0, 2: 1.0},
            prev_avg_trust=0.0,
        ))
        assert bd.trust_alignment == pytest.approx(0.05)

    def test_empty_trust_row_zero_signal(self):
        engine = make_engine()
        r, bd = engine.compute(**base_kwargs(trust_row={}))
        assert bd.trust_alignment == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# F. Mask penalty
# ---------------------------------------------------------------------------

class TestMaskPenalty:
    def test_mask_penalty_applied_when_violated(self):
        engine = make_engine(mask_penalty=-0.5)
        r, bd = engine.compute(**base_kwargs(mask_violated=True))
        assert bd.mask_penalty == pytest.approx(-0.5)

    def test_no_mask_penalty_for_valid_action(self):
        engine = make_engine(mask_penalty=-0.5)
        r, bd = engine.compute(**base_kwargs(mask_violated=False))
        assert bd.mask_penalty == pytest.approx(0.0)

    def test_mask_penalty_lowers_total(self):
        engine = make_engine(mask_penalty=-0.5)
        _, bd_clean    = engine.compute(**base_kwargs(mask_violated=False))
        _, bd_violated = engine.compute(**base_kwargs(mask_violated=True))
        assert bd_violated.total < bd_clean.total
        assert bd_clean.total - bd_violated.total == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# G. Spam penalty
# ---------------------------------------------------------------------------

class TestSpamPenalty:
    def test_spam_penalty_off_by_default(self):
        """Default reward_spam_penalty=0.0 means no penalty regardless."""
        engine = make_engine(spam_penalty=0.0, max_pending=2)
        r, bd = engine.compute(**base_kwargs(pending_outgoing=10))
        assert bd.spam_penalty == pytest.approx(0.0)

    def test_spam_penalty_activates_when_enabled(self):
        engine = make_engine(spam_penalty=0.1, max_pending=2)
        r, bd = engine.compute(**base_kwargs(pending_outgoing=3))  # 1 over limit
        assert bd.spam_penalty == pytest.approx(-0.1)

    def test_spam_penalty_zero_within_limit(self):
        engine = make_engine(spam_penalty=0.1, max_pending=2)
        r, bd = engine.compute(**base_kwargs(pending_outgoing=2))  # exactly at limit
        assert bd.spam_penalty == pytest.approx(0.0)

    def test_no_double_penalty_for_proposal_cost(self):
        """Proposal cost is a resource drain; reward doesn't penalise it again."""
        # With spam_penalty=0 (default), pending proposals have zero reward impact.
        engine = make_engine(spam_penalty=0.0, max_pending=2)
        _, bd_0 = engine.compute(**base_kwargs(pending_outgoing=0))
        _, bd_1 = engine.compute(**base_kwargs(pending_outgoing=1))
        assert bd_0.total == pytest.approx(bd_1.total)


# ---------------------------------------------------------------------------
# H. Collapse penalty
# ---------------------------------------------------------------------------

class TestCollapsePenalty:
    def test_newly_collapsed_gives_collapse_penalty(self):
        engine = make_engine(collapse_penalty=-10.0)
        r, bd = engine.compute(**base_kwargs(is_newly_collapsed=True))
        assert bd.collapse_penalty == pytest.approx(-10.0)
        assert r == pytest.approx(-10.0)

    def test_newly_collapsed_overrides_all_other_components(self):
        """On collapse turn, only collapse_penalty is returned."""
        engine = make_engine(collapse_penalty=-10.0)
        r, bd = engine.compute(
            **base_kwargs(
                is_newly_collapsed=True,
                curr_exposure=1.0,  # would give crisis_mitigation otherwise
                coalition_size=4,   # would give cooperation otherwise
                mask_violated=True, # would give mask_penalty otherwise
            )
        )
        assert bd.survival == pytest.approx(0.0)
        assert bd.stability_delta == pytest.approx(0.0)
        assert bd.crisis_mitigation == pytest.approx(0.0)
        assert bd.cooperation == pytest.approx(0.0)
        assert bd.mask_penalty == pytest.approx(0.0)
        assert bd.collapse_penalty == pytest.approx(-10.0)

    def test_already_collapsed_gives_zero(self):
        engine = make_engine()
        r, bd = engine.compute(**base_kwargs(is_collapsed=True))
        assert r == pytest.approx(0.0)
        assert bd.total == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Bounds & determinism
# ---------------------------------------------------------------------------

class TestBoundsAndDeterminism:
    def test_reward_within_bounds_alive(self):
        """Typical alive agent reward must stay in [-2, +2]."""
        engine = make_engine()
        kwargs = base_kwargs(
            curr_stability=0.40,
            prev_stability=0.30,
            prev_exposure=0.30,
            curr_exposure=0.50,
            coalition_size=4,
            trust_row={1: 1.0, 2: 1.0, 3: 1.0},
            prev_avg_trust=0.0,
            mask_violated=True,
        )
        r, bd = engine.compute(**kwargs)
        assert -2.0 <= r <= 2.0

    def test_deterministic_same_inputs(self):
        engine = make_engine()
        kwargs = base_kwargs(
            prev_stability=0.55, curr_stability=0.60,
            prev_exposure=0.15, curr_exposure=0.15,
            coalition_size=2, trust_row={1: 0.6, 2: 0.4, 3: 0.3},
            prev_avg_trust=0.4,
            mask_violated=False,
        )
        r1, bd1 = engine.compute(**kwargs)
        r2, bd2 = engine.compute(**kwargs)
        assert r1 == pytest.approx(r2)
        assert bd1.to_dict() == bd2.to_dict()

    def test_cooperation_never_dominates(self):
        """Cooperation ≤ 0.15 always; survival ≥ 0.5+ so cooperation < 30% of total."""
        engine = make_engine(survival_reward=1.0, coop_per_peer=0.05)
        r, bd = engine.compute(**base_kwargs(coalition_size=100))  # absurdly large
        assert bd.cooperation <= 0.15
        assert bd.cooperation / bd.survival <= 0.20

    def test_trust_never_dominates(self):
        """Trust ≤ 0.05 always."""
        engine = make_engine(survival_reward=1.0, trust_alignment=1.0)  # weight=1 (extreme)
        r, bd = engine.compute(**base_kwargs(
            trust_row={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            prev_avg_trust=0.0,   # large delta → still capped
        ))
        assert bd.trust_alignment <= 0.05


# ---------------------------------------------------------------------------
# compute_all (batch API)
# ---------------------------------------------------------------------------

class TestComputeAll:
    def _all_kwargs(self, agents):
        """Common kwargs with Phase 6 delta fields."""
        return dict(
            agents=agents,
            newly_collapsed=set(),
            collapsed={i: False for i in agents},
            prev_stability={i: 0.60 for i in agents},
            curr_stability={i: 0.65 for i in agents},
            prev_exposure={i: 0.10 for i in agents},
            curr_exposure={i: 0.10 for i in agents},
            prev_avg_trust={i: 0.0 for i in agents},
            coalition_sizes={i: 0 for i in agents},
            trust_matrix={i: {j: 0.5 for j in agents if j != i} for i in agents},
            mask_violated=set(),
            pending_outgoing={i: 0 for i in agents},
        )

    def test_all_agents_receive_reward(self):
        engine = make_engine()
        agents = [0, 1, 2, 3]
        rewards, breakdowns = engine.compute_all(**self._all_kwargs(agents))
        assert set(rewards.keys()) == set(agents)
        assert set(breakdowns.keys()) == set(agents)

    def test_collapsed_agent_gets_zero(self):
        engine = make_engine()
        agents = [0, 1]
        kw = self._all_kwargs(agents)
        kw["collapsed"] = {0: True, 1: False}
        rewards, _ = engine.compute_all(**kw)
        assert rewards[0] == pytest.approx(0.0)
        assert rewards[1] > 0.0  # alive

    def test_newly_collapsed_gets_penalty(self):
        engine = make_engine(collapse_penalty=-10.0)
        agents = [0, 1]
        kw = self._all_kwargs(agents)
        kw["newly_collapsed"] = {0}
        kw["collapsed"]       = {0: True, 1: False}
        rewards, breakdowns = engine.compute_all(**kw)
        assert rewards[0] == pytest.approx(-10.0)


# ---------------------------------------------------------------------------
# Environment integration — info["reward_breakdown"]
# ---------------------------------------------------------------------------

class TestEnvIntegration:
    def test_info_has_reward_breakdown(self):
        from district_accord.env import DistrictAccordEnv
        from district_accord.utils.types import DiscreteAction
        env = DistrictAccordEnv(EnvConfig(num_districts=4, max_turns=10))
        env.reset()
        _, _, _, _, info = env.step({i: DiscreteAction.IGNORE for i in range(4)})
        assert "reward_breakdown" in info
        assert set(info["reward_breakdown"].keys()) == {0, 1, 2, 3}

    def test_breakdown_total_matches_reward(self):
        from district_accord.env import DistrictAccordEnv
        from district_accord.utils.types import DiscreteAction
        env = DistrictAccordEnv(EnvConfig(num_districts=4, max_turns=10))
        env.reset()
        _, rewards, _, _, info = env.step({i: DiscreteAction.IGNORE for i in range(4)})
        for agent_id in range(4):
            bd = info["reward_breakdown"][agent_id]
            assert bd["total"] == pytest.approx(rewards[agent_id], abs=1e-4)

    def test_stability_increase_leads_to_positive_delta(self):
        """Force-inject stability improvement; verify breakdown captures it."""
        from district_accord.env import DistrictAccordEnv
        from district_accord.utils.types import DiscreteAction
        env = DistrictAccordEnv(EnvConfig(
            num_districts=4, max_turns=10,
            reward_stability_weight=1.0,
        ))
        env.reset()
        # Boost stability before step so prev is low and curr will be high
        for d in env._districts.values():
            d.stability = 0.30
        # One DEFEND turn should raise stability
        _, _, _, _, info = env.step(
            {0: DiscreteAction.DEFEND, 1: DiscreteAction.IGNORE,
             2: DiscreteAction.IGNORE, 3: DiscreteAction.IGNORE}
        )
        # Agent 0 used DEFEND → stability should rise → positive delta
        bd_defender = info["reward_breakdown"][0]
        assert bd_defender["stability_delta"] > 0.0
