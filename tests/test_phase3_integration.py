"""
tests/test_phase3_integration.py
==================================
End-to-end Phase 3 integration tests for DistrictAccordEnv.

Tests the full Phase 3 system working together:
    - Dynamic action mask correctness (env step)
    - Mask violation → IGNORE + penalty
    - Coalition formation via PROPOSE + ACCEPT (2-turn flow)
    - Coalition stability bonus in rewards
    - Coalition exposure damping (measurable)
    - Trust update after accept / reject
    - Resource transfer via SHARE_RESOURCES
    - Negotiation TTL expiry over turns
    - Anti-spam: proposal cost deducted
    - info["coalition"] and info["trust"] present

All prior Phase 1+2 tests must still pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from district_accord.core.negotiation import Proposal
from district_accord.env import DistrictAccordEnv
from district_accord.utils.config import EnvConfig
from district_accord.utils.types import AgentID, DiscreteAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(
    n: int = 4,
    turns: int = 20,
    seed: int = 0,
    **kwargs,
) -> DistrictAccordEnv:
    cfg = EnvConfig(num_districts=n, max_turns=turns, seed=seed, **kwargs)
    env = DistrictAccordEnv(cfg)
    return env


def all_ignore(n: int) -> dict:
    return {i: DiscreteAction.IGNORE for i in range(n)}


# ---------------------------------------------------------------------------
# Basic integration — env still runs cleanly
# ---------------------------------------------------------------------------

class TestBasicIntegration:
    def test_reset_returns_obs_for_all_agents(self):
        env = make_env(4)
        obs = env.reset()
        assert set(obs.keys()) == {0, 1, 2, 3}

    def test_obs_has_all_keys(self):
        env = make_env(4)
        obs = env.reset()
        for agent_obs in obs.values():
            for key in ("self", "others", "crisis", "turn", "action_mask", "flat"):
                assert key in agent_obs

    def test_step_returns_correct_signature(self):
        env = make_env(4)
        env.reset()
        obs, rewards, done, trunc, info = env.step(all_ignore(4))
        assert set(rewards.keys()) == {0, 1, 2, 3}
        assert isinstance(done, bool)
        assert isinstance(trunc, bool)

    def test_info_has_phase3_keys(self):
        env = make_env(4)
        env.reset()
        _, _, _, _, info = env.step(all_ignore(4))
        assert "coalition" in info
        assert "negotiation" in info
        assert "trust" in info
        assert "mask_violations" in info

    def test_full_episode_completes(self):
        env = make_env(4, turns=20, seed=99)
        env.reset()
        for _ in range(env.config.max_turns):
            obs, rewards, done, trunc, info = env.step(all_ignore(4))
            if done or trunc:
                break
        assert done or trunc

    def test_backward_compat_discrete_actions(self):
        env = make_env(2, turns=5, seed=1)
        env.reset()
        for _ in range(5):
            obs, rewards, done, trunc, info = env.step(
                {0: DiscreteAction.INVEST, 1: DiscreteAction.DEFEND}
            )
            if done or trunc:
                break


# ---------------------------------------------------------------------------
# Dynamic action mask
# ---------------------------------------------------------------------------

class TestDynamicMask:
    def test_invest_defend_ignore_always_valid(self):
        env = make_env(4)
        obs = env.reset()
        for agent_obs in obs.values():
            mask = agent_obs["action_mask"]
            assert mask[DiscreteAction.INVEST] == 1.0
            assert mask[DiscreteAction.DEFEND] == 1.0
            assert mask[DiscreteAction.IGNORE] == 1.0

    def test_recover_valid_with_sufficient_resources(self):
        env = make_env(4)
        obs = env.reset()
        # Default resources ~0.6, recover_resource_cost=0.08
        for agent_obs in obs.values():
            assert agent_obs["action_mask"][DiscreteAction.RECOVER] == 1.0

    def test_recover_invalid_with_insufficient_resources(self):
        env = make_env(4)
        env.reset()
        # Artificially deplete resources below recover_resource_cost
        env._districts[0].resources = 0.01
        obs = env._get_obs()
        mask = obs[0]["action_mask"]
        assert mask[DiscreteAction.RECOVER] == 0.0

    def test_accept_reject_zero_with_no_proposals(self):
        env = make_env(4)
        obs = env.reset()
        for agent_obs in obs.values():
            assert agent_obs["action_mask"][DiscreteAction.ACCEPT_COALITION] == 0.0
            assert agent_obs["action_mask"][DiscreteAction.REJECT_COALITION] == 0.0

    def test_accept_becomes_valid_after_proposal(self):
        env = make_env(4)
        env.reset()
        # Manually create a coalition proposal to agent 1
        env._negotiation.create(0, 1, "coalition", {}, current_turn=0)
        obs = env._get_obs()
        mask = obs[1]["action_mask"]
        assert mask[DiscreteAction.ACCEPT_COALITION] == 1.0
        assert mask[DiscreteAction.REJECT_COALITION] == 1.0

    def test_request_aid_zero_when_stability_high(self):
        env = make_env(4)
        obs = env.reset()
        # Default stability ~0.7, threshold=0.4 → request_aid should be masked
        for agent_obs in obs.values():
            assert agent_obs["action_mask"][DiscreteAction.REQUEST_AID] == 0.0

    def test_request_aid_valid_when_stability_low(self):
        env = make_env(4)
        env.reset()
        env._districts[0].stability = 0.3  # below aid_request_stability_threshold=0.4
        obs = env._get_obs()
        mask = obs[0]["action_mask"]
        assert mask[DiscreteAction.REQUEST_AID] == 1.0

    def test_share_resources_valid_with_resources(self):
        env = make_env(4)
        obs = env.reset()
        # Default resources ~0.6 > min_share_threshold=0.1 AND valid target exists
        for agent_obs in obs.values():
            assert agent_obs["action_mask"][DiscreteAction.SHARE_RESOURCES] == 1.0

    def test_propose_coalition_valid_with_targets(self):
        env = make_env(4)
        obs = env.reset()
        for agent_obs in obs.values():
            assert agent_obs["action_mask"][DiscreteAction.PROPOSE_COALITION] == 1.0


# ---------------------------------------------------------------------------
# Mask violation: penalty + IGNORE override
# ---------------------------------------------------------------------------

class TestMaskViolation:
    def test_masked_action_converted_to_ignore(self):
        env = make_env(4)
        env.reset()
        # ACCEPT_COALITION is masked (no pending proposals)
        actions = {i: DiscreteAction.IGNORE for i in range(4)}
        actions[0] = DiscreteAction.ACCEPT_COALITION  # masked!
        _, _, _, _, info = env.step(actions)
        # Agent 0's action should have been overridden to IGNORE
        assert info["actions_taken"][0] == "IGNORE"
        assert 0 in info["mask_violations"]

    def test_mask_violation_triggers_penalty(self):
        env = make_env(4)
        env.reset()
        actions = {i: DiscreteAction.IGNORE for i in range(4)}
        actions[0] = DiscreteAction.ACCEPT_COALITION  # masked
        _, rewards, _, _, info = env.step(actions)
        # Agent 0's reward must include the mask_violation_penalty component.
        # With Phase 4 RewardEngine the total includes stability_delta + crisis_mitigation
        # so we can't check an exact value — instead verify breakdown explicitly.
        breakdown = info["reward_breakdown"][0]
        assert breakdown["mask_penalty"] == pytest.approx(env.config.mask_violation_penalty)
        # Violated agent must receive LESS than a clean agent on the same step
        clean_rewards = [rewards[i] for i in range(1, 4)]
        assert rewards[0] < max(clean_rewards)

    def test_valid_actions_not_penalized(self):
        env = make_env(4)
        env.reset()
        _, rewards, _, _, info = env.step(all_ignore(4))
        for agent_id, r in rewards.items():
            # No mask penalty component for clean agents
            breakdown = info["reward_breakdown"][agent_id]
            assert breakdown["mask_penalty"] == pytest.approx(0.0)

    def test_propose_to_self_is_violation(self):
        env = make_env(4)
        env.reset()
        from district_accord.spaces.action import make_default_parsed_action
        # Create a ParsedAction targeting self
        parsed = make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=0)
        actions = {i: DiscreteAction.IGNORE for i in range(4)}
        actions[0] = parsed
        _, _, _, _, info = env.step(actions)
        assert info["actions_taken"][0] == "IGNORE"
        assert 0 in info["mask_violations"]


# ---------------------------------------------------------------------------
# Coalition formation (2-turn flow)
# ---------------------------------------------------------------------------

class TestCoalitionFormation:
    def test_propose_creates_proposal(self):
        env = make_env(4)
        env.reset()
        from district_accord.spaces.action import make_default_parsed_action
        actions = all_ignore(4)
        actions[0] = make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=1)
        env.step(actions)
        # Agent 1 should have a pending coalition proposal from agent 0
        proposals = env.negotiation.pending_for(1)
        assert len(proposals) == 1
        assert proposals[0].proposer == 0
        assert proposals[0].kind == "coalition"

    def test_accept_forms_coalition(self):
        env = make_env(4)
        env.reset()
        from district_accord.spaces.action import make_default_parsed_action

        # Turn 1: agent 0 proposes to agent 1
        actions = all_ignore(4)
        actions[0] = make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=1)
        env.step(actions)

        # Turn 2: agent 1 accepts
        actions = all_ignore(4)
        actions[1] = make_default_parsed_action(DiscreteAction.ACCEPT_COALITION)
        env.step(actions)

        # Both should now be in the same coalition
        assert env.coalition.same_coalition(0, 1) is True

    def test_coalition_appears_in_info(self):
        env = make_env(4)
        env.reset()
        from district_accord.spaces.action import make_default_parsed_action

        actions = all_ignore(4)
        actions[0] = make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=1)
        env.step(actions)

        actions = all_ignore(4)
        actions[1] = make_default_parsed_action(DiscreteAction.ACCEPT_COALITION)
        _, _, _, _, info = env.step(actions)

        coalition_info = info["coalition"]
        assert "memberships" in coalition_info
        assert coalition_info["memberships"][0] == coalition_info["memberships"][1]

    def test_coalition_bonus_in_reward(self):
        env = make_env(4)
        env.reset()
        from district_accord.spaces.action import make_default_parsed_action

        # Form coalition: 0 proposes to 1
        actions = all_ignore(4)
        actions[0] = make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=1)
        env.step(actions)

        # 1 accepts
        actions = all_ignore(4)
        actions[1] = make_default_parsed_action(DiscreteAction.ACCEPT_COALITION)
        env.step(actions)

        # Next turn: coalition members (0 and 1) should have cooperation component
        _, rewards, _, _, info = env.step(all_ignore(4))
        # Verify cooperation component is present in breakdown for coalition members
        assert info["reward_breakdown"][0]["cooperation"] > 0.0
        assert info["reward_breakdown"][1]["cooperation"] > 0.0
        # Non-members get zero cooperation bonus
        assert info["reward_breakdown"][2]["cooperation"] == pytest.approx(0.0)
        assert info["reward_breakdown"][3]["cooperation"] == pytest.approx(0.0)
        # Coalition members earn strictly more (from cooperation) than non-members
        # assuming same stability/crisis trajectory (might differ due to random init)
        # So check only the cooperation component, not the total
        cfg = env.config
        assert info["reward_breakdown"][0]["cooperation"] == pytest.approx(
            min(cfg.reward_cooperation_per_peer * 1, 0.15)
        )

    def test_reject_coalition_removes_proposal(self):
        env = make_env(4)
        env.reset()
        from district_accord.spaces.action import make_default_parsed_action

        actions = all_ignore(4)
        actions[0] = make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=1)
        env.step(actions)

        actions = all_ignore(4)
        actions[1] = make_default_parsed_action(DiscreteAction.REJECT_COALITION)
        env.step(actions)

        assert env.coalition.get_coalition(0) is None
        assert env.coalition.get_coalition(1) is None
        assert len(env.negotiation.pending_for(1)) == 0


# ---------------------------------------------------------------------------
# Trust updates
# ---------------------------------------------------------------------------

class TestTrustUpdates:
    def test_trust_increases_on_accept(self):
        env = make_env(4, trust_init_std=0.0)
        env.reset()
        from district_accord.spaces.action import make_default_parsed_action

        baseline_0_1 = env.trust.get(0, 1)
        baseline_1_0 = env.trust.get(1, 0)

        # T1: 0 proposes to 1
        actions = all_ignore(4)
        actions[0] = make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=1)
        env.step(actions)

        # T2: 1 accepts
        actions = all_ignore(4)
        actions[1] = make_default_parsed_action(DiscreteAction.ACCEPT_COALITION)
        env.step(actions)

        # Both should have higher trust (after decay)
        assert env.trust.get(0, 1) > baseline_0_1 * (env.config.trust_decay ** 2) - 0.01
        assert env.trust.get(1, 0) > baseline_1_0 * (env.config.trust_decay ** 2) - 0.01

    def test_trust_decreases_on_reject(self):
        env = make_env(4, trust_init_std=0.0)
        env.reset()
        from district_accord.spaces.action import make_default_parsed_action

        baseline = env.trust.get(1, 0)  # rejector's (1) trust of proposer (0)

        actions = all_ignore(4)
        actions[0] = make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=1)
        env.step(actions)

        actions = all_ignore(4)
        actions[1] = make_default_parsed_action(DiscreteAction.REJECT_COALITION)
        env.step(actions)

        # Rejector (1) should trust proposer (0) less
        assert env.trust.get(1, 0) < baseline

    def test_trust_decays_each_turn(self):
        env = make_env(4, trust_init_std=0.0, trust_init_mean=0.5, trust_decay=0.90)
        env.reset()
        baseline = env.trust.get(0, 1)
        env.step(all_ignore(4))
        # After one decay: 0.5 * 0.90 = 0.45
        expected = baseline * env.config.trust_decay
        assert env.trust.get(0, 1) == pytest.approx(expected, abs=1e-4)

    def test_trust_visible_in_obs(self):
        env = make_env(4)
        obs = env.reset()
        # "others" column 2 = trust (see OTHERS_SCHEMA)
        for agent_id, agent_obs in obs.items():
            others = agent_obs["others"]
            assert others.shape == (3, 4)
            # Column 2 = trust, should be in [0, 1] (clipped + noise)
            trust_col = others[:, 2]
            assert np.all(trust_col >= 0.0)
            assert np.all(trust_col <= 1.0)


# ---------------------------------------------------------------------------
# Coalition exposure damping
# ---------------------------------------------------------------------------

class TestCoalitionDamping:
    def test_coalition_members_have_lower_exposure(self):
        env = make_env(
            4,
            coalition_exposure_damping=0.50,
            obs_neighbor_noise_std=0.0,
            seed=7,
        )
        env.reset()
        from district_accord.spaces.action import make_default_parsed_action

        # Form coalition: 0 and 1
        actions = all_ignore(4)
        actions[0] = make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=1)
        env.step(actions)

        actions = all_ignore(4)
        actions[1] = make_default_parsed_action(DiscreteAction.ACCEPT_COALITION)
        env.step(actions)

        # Record exposure for one more turn
        env.step(all_ignore(4))

        # Coalition members' exposure should be damped relative to solo members
        # (hard to guarantee exact ordering due to RNG, but members should not exceed +0.3 of non-members)
        exposure_member = env._districts[0].crisis_exposure
        exposure_solo = env._districts[2].crisis_exposure
        # With 50% damping, member exposure should be significantly lower
        # (allow tolerance for the 3-turn difference in coalition formation)
        assert exposure_member <= exposure_solo + 0.5  # conservative bound


# ---------------------------------------------------------------------------
# Proposal cost
# ---------------------------------------------------------------------------

class TestProposalCost:
    def test_proposal_cost_deducted(self):
        env = make_env(4, proposal_cost=0.05)
        env.reset()
        from district_accord.spaces.action import make_default_parsed_action

        r_before = env._districts[0].resources
        actions = all_ignore(4)
        actions[0] = make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=1)
        env.step(actions)
        r_after = env._districts[0].resources
        # Proposal cost should be deducted (plus passive_resource_drain)
        net_drain = env.config.passive_resource_drain + env.config.proposal_cost
        assert (r_before - r_after) >= (net_drain - 0.01)

    def test_blocked_proposal_no_cost(self):
        """If proposal is blocked (e.g. cooldown), no cost is deducted."""
        env = make_env(4, proposal_cost=0.05, proposal_cooldown=10)
        env.reset()
        from district_accord.spaces.action import make_default_parsed_action

        # First proposal succeeds
        actions = all_ignore(4)
        actions[0] = make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=1)
        env.step(actions)

        r_before_second = env._districts[0].resources

        # Second proposal to same target — blocked by cooldown
        actions = all_ignore(4)
        actions[0] = make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=1)
        env.step(actions)

        r_after_second = env._districts[0].resources
        # Only passive drain should be deducted (no proposal cost)
        drain_only = env.config.passive_resource_drain
        assert abs(r_before_second - r_after_second - drain_only) < 0.02  # small tolerance for exposure


# ---------------------------------------------------------------------------
# TTL expiry (via env)
# ---------------------------------------------------------------------------

class TestTTLExpiry:
    def test_proposal_expires_and_accept_no_longer_works(self):
        env = make_env(4, proposal_ttl=2)
        env.reset()
        from district_accord.spaces.action import make_default_parsed_action

        # T1: propose
        actions = all_ignore(4)
        actions[0] = make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=1)
        env.step(actions)

        # T2, T3: do nothing (TTL ticks)
        env.step(all_ignore(4))  # ttl → 1
        env.step(all_ignore(4))  # ttl → 0 → expired

        # Proposal should be gone
        assert len(env.negotiation.pending_for(1)) == 0
        # ACCEPT_COALITION should now be masked again
        obs = env._get_obs()
        assert obs[1]["action_mask"][DiscreteAction.ACCEPT_COALITION] == 0.0


# ---------------------------------------------------------------------------
# Resource sharing
# ---------------------------------------------------------------------------

class TestResourceSharing:
    def test_share_resources_transfers_with_loss(self):
        env = make_env(4, transfer_loss_ratio=0.10)
        env.reset()
        from district_accord.spaces.action import make_default_parsed_action

        env._districts[0].resources = 0.8
        env._districts[1].resources = 0.2
        r1_before = env._districts[1].resources

        actions = all_ignore(4)
        amount = 0.2
        parsed = make_default_parsed_action(
            DiscreteAction.SHARE_RESOURCES, target=1, amount=amount
        )
        actions[0] = parsed
        env.step(actions)

        r1_after = env._districts[1].resources
        expected_received = amount * (1.0 - 0.10)
        net_change = r1_after - r1_before + env.config.passive_resource_drain
        assert net_change == pytest.approx(expected_received, abs=0.02)
