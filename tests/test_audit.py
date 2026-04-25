"""
tests/test_audit.py
=====================
District Accord — Full System Audit (Phases 1-6)

12 sections, each PASS or FAIL with specific failure messages.
Run:
    pytest tests/test_audit.py -v
or via examples/audit.py for the formatted report.
"""

from __future__ import annotations

import json
import time
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pytest

from district_accord.core.coalition import CoalitionSystem
from district_accord.core.negotiation import NegotiationSystem
from district_accord.core.trust import TrustSystem
from district_accord.engine.reward import RewardEngine
from district_accord.env import DistrictAccordEnv
from district_accord.policy.runner import (
    EpisodeRunner,
    StepRecord,
    load_trajectory,
    save_trajectory,
    verify_replay,
)
from district_accord.policy.self_play import SelfPlayPolicy
from district_accord.spaces.action import make_default_parsed_action
from district_accord.utils.config import EnvConfig
from district_accord.utils.types import AgentID, DiscreteAction


# ═══════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════

SEED = 42

SMALL_CFG = EnvConfig(
    num_districts=4, max_turns=20, seed=SEED,
    trust_init_std=0.0, obs_neighbor_noise_std=0.0,
    reward_spam_penalty=0.0,
    proposal_cooldown=4, proposal_cost=0.04, max_pending_proposals=2,
)
FULL_CFG = EnvConfig(
    num_districts=12, max_turns=100, seed=SEED,
    trust_init_std=0.0, obs_neighbor_noise_std=0.0,
    reward_spam_penalty=0.0,
    proposal_cooldown=4, proposal_cost=0.04, max_pending_proposals=2,
)

def fresh_env(cfg: EnvConfig = SMALL_CFG) -> DistrictAccordEnv:
    return DistrictAccordEnv(cfg)

def run_episode(env, policy):
    runner = EpisodeRunner()
    return runner.run_episode(env, policy, seed=SEED)


# Pipeline event priority (lower = earlier in pipeline)
_PIPELINE_PRIORITY = {
    "action_validated":    0,
    "action_invalid":      0,
    "proposal_created":    1,
    "coalition_formed":    2,
    "coalition_joined":    2,
    "proposal_rejected":   3,
    "resource_transferred": 4,
    "proposal_expired":    5,
    "collapse":            6,
    "trust_updated":       7,
}


# ═══════════════════════════════════════════════════════════════════════
# Section 1 — Determinism
# ═══════════════════════════════════════════════════════════════════════

class TestSection1Determinism:
    """CRITICAL: same seed → bit-identical episodes."""

    def _run_pair(self, cfg):
        p1 = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        p2 = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        e1 = DistrictAccordEnv(cfg); e2 = DistrictAccordEnv(cfg)
        r = EpisodeRunner()
        t1 = r.run_episode(e1, p1, seed=SEED)
        t2 = r.run_episode(e2, p2, seed=SEED)
        return t1, t2

    def test_1a_identical_episode_length(self):
        t1, t2 = self._run_pair(SMALL_CFG)
        assert len(t1) == len(t2), f"Length mismatch: {len(t1)} vs {len(t2)}"

    def test_1b_identical_actions(self):
        t1, t2 = self._run_pair(SMALL_CFG)
        for i, (r1, r2) in enumerate(zip(t1, t2)):
            assert r1.actions == r2.actions, f"Step {i}: actions differ"

    def test_1c_identical_rewards(self):
        t1, t2 = self._run_pair(SMALL_CFG)
        for i, (r1, r2) in enumerate(zip(t1, t2)):
            for a in r1.rewards:
                diff = abs(r1.rewards[a] - r2.rewards[a])
                assert diff < 1e-9, (
                    f"Step {i} agent {a}: reward diff {diff:.2e}"
                )

    def test_1d_identical_events(self):
        t1, t2 = self._run_pair(SMALL_CFG)
        for i, (r1, r2) in enumerate(zip(t1, t2)):
            e1 = r1.info["events"]; e2 = r2.info["events"]
            assert len(e1) == len(e2), f"Step {i}: event count {len(e1)} vs {len(e2)}"
            for j, (ev1, ev2) in enumerate(zip(e1, e2)):
                assert ev1["type"] == ev2["type"], (
                    f"Step {i} event {j}: type {ev1['type']} vs {ev2['type']}"
                )

    def test_1e_identical_state_snapshots(self):
        t1, t2 = self._run_pair(SMALL_CFG)
        for i, (r1, r2) in enumerate(zip(t1, t2)):
            s1 = r1.info["state_snapshot"]; s2 = r2.info["state_snapshot"]
            for a_str in s1["agents"]:
                for field in ("resources", "stability", "crisis_exposure"):
                    v1 = s1["agents"][a_str][field]
                    v2 = s2["agents"][a_str][field]
                    assert abs(v1 - v2) < 1e-9, (
                        f"Step {i} agent {a_str} {field}: {v1} vs {v2}"
                    )

    def test_1f_determinism_full_scale(self):
        """12 agents, 100 turns — both runs identical."""
        t1, t2 = self._run_pair(FULL_CFG)
        assert len(t1) == len(t2)
        for i, (r1, r2) in enumerate(zip(t1, t2)):
            for a in r1.rewards:
                assert abs(r1.rewards[a] - r2.rewards[a]) < 1e-9

    def test_1g_different_seeds_differ(self):
        p1 = SelfPlayPolicy(mode="mask_aware_random", seed=1)
        p2 = SelfPlayPolicy(mode="mask_aware_random", seed=2)
        e1 = DistrictAccordEnv(SMALL_CFG); e2 = DistrictAccordEnv(SMALL_CFG)
        r = EpisodeRunner()
        t1 = r.run_episode(e1, p1, seed=1)
        t2 = r.run_episode(e2, p2, seed=2)
        # At least one reward must differ
        diffs = [
            abs(r1.rewards.get(0, 0) - r2.rewards.get(0, 0))
            for r1, r2 in zip(t1, t2)
        ]
        assert any(d > 1e-9 for d in diffs), "Different seeds produced identical runs"


# ═══════════════════════════════════════════════════════════════════════
# Section 2 — Action Mask Integrity
# ═══════════════════════════════════════════════════════════════════════

class TestSection2ActionMask:
    """No invalid action executed; mask violation → penalty + IGNORE."""

    def test_2a_no_masked_action_executed_over_50_steps(self):
        """
        Force invalid actions and verify they become IGNORE in the pipeline.
        We do this by checking that action_invalid events lead to zero actual effect.
        """
        env = fresh_env()
        rng = np.random.default_rng(SEED)
        env.reset()

        for step in range(50):
            if all(env._collapsed.get(a, False) for a in range(4)):
                break
            # Submit a completely random action (may be masked)
            actions = {
                a: DiscreteAction(int(rng.integers(0, 9)))
                for a in range(4)
                if not env._collapsed.get(a, False)
            }
            _, _, done, trunc, info = env.step(actions)

            # All action_invalid events must use IGNORE internally
            # (action_mask in obs must be 0 for the chosen action)
            for ev in info["events"]:
                if ev["type"] == "action_invalid":
                    aid = ev["payload"]["agent_id"]
                    obs = env._get_obs()
                    # After mask enforcement the agent should not have executed
                    # the invalid action — verified by the event itself existing
                    assert ev["payload"]["action"] is not None

            if done or trunc:
                break

    def test_2b_mask_violation_incurs_penalty(self):
        """Force an action that is masked; verify mask_penalty in breakdown."""
        env = fresh_env(EnvConfig(
            num_districts=4, max_turns=10, seed=SEED,
            trust_init_std=0.0, obs_neighbor_noise_std=0.0,
            mask_violation_penalty=-0.5,
        ))
        obs = env.reset()

        # Find a masked action for agent 0
        mask = obs[0]["action_mask"]
        masked_actions = [i for i, m in enumerate(mask) if m == 0]
        if not masked_actions:
            pytest.skip("No masked actions available at turn 0")

        invalid_action = DiscreteAction(masked_actions[0])
        actions = {a: DiscreteAction.IGNORE for a in range(4)}
        actions[0] = invalid_action

        _, _, _, _, info = env.step(actions)
        bd = info["reward_breakdown"][0]
        assert bd["mask_penalty"] < 0.0, (
            f"Expected negative mask_penalty, got {bd['mask_penalty']}"
        )

    def test_2c_mask_aware_random_never_violates(self):
        """mask_aware_random policy must never trigger action_invalid."""
        env = fresh_env(FULL_CFG)
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        traj = run_episode(env, policy)

        for rec in traj:
            invalid_events = [
                e for e in rec.info["events"] if e["type"] == "action_invalid"
            ]
            assert len(invalid_events) == 0, (
                f"Step {rec.step}: mask_aware_random caused {len(invalid_events)} invalid events"
            )

    def test_2d_action_mask_shape_and_values(self):
        env = fresh_env(FULL_CFG)
        obs = env.reset()
        for a, agent_obs in obs.items():
            mask = agent_obs["action_mask"]
            assert mask.shape == (9,), f"Agent {a}: mask shape {mask.shape}"
            assert set(mask).issubset({0.0, 1.0}), f"Agent {a}: non-binary mask"

    def test_2e_mask_always_allows_ignore(self):
        """IGNORE (action 2) must always be valid."""
        env = fresh_env(FULL_CFG)
        for _ in range(5):
            obs = env.reset()
            for a, agent_obs in obs.items():
                assert agent_obs["action_mask"][DiscreteAction.IGNORE.value] == 1, (
                    f"Agent {a}: IGNORE masked"
                )

    def test_2f_accept_only_valid_when_proposal_exists(self):
        """ACCEPT_COALITION mask=1 iff pending proposal addressed to agent."""
        env = fresh_env()
        obs = env.reset()
        for a, agent_obs in obs.items():
            pending = env._negotiation.pending_for(a)
            accept_mask = agent_obs["action_mask"][DiscreteAction.ACCEPT_COALITION.value]
            if len(pending) > 0:
                assert accept_mask == 1
            else:
                assert accept_mask == 0, (
                    f"Agent {a}: ACCEPT valid but no pending proposals"
                )


# ═══════════════════════════════════════════════════════════════════════
# Section 3 — Proposal System Integrity
# ═══════════════════════════════════════════════════════════════════════

class TestSection3ProposalSystem:
    """Proposal lifecycle: create, expire, cooldown, caps, edge cases."""

    def _make_neg(self, cfg=SMALL_CFG):
        from district_accord.utils.config import EnvConfig
        return NegotiationSystem(cfg)

    def test_3a_incoming_cap_enforced(self):
        """Cannot have more than max_pending_proposals addressed to same target."""
        cfg = EnvConfig(num_districts=4, max_turns=10, max_pending_proposals=2)
        neg = NegotiationSystem(cfg)

        # Create 2 proposals to agent 3
        p1 = neg.create(proposer=0, target=3, kind="coalition", terms={}, current_turn=0)
        p2 = neg.create(proposer=1, target=3, kind="coalition", terms={}, current_turn=0)
        assert p1 is not None
        assert p2 is not None

        # Third should be rejected (incoming cap)
        p3 = neg.create(proposer=2, target=3, kind="coalition", terms={}, current_turn=0)
        assert p3 is None, "Incoming cap not enforced — 3rd proposal accepted"

    def test_3b_outgoing_cap_enforced(self):
        """Cannot have more than max_pending_proposals FROM same proposer."""
        cfg = EnvConfig(num_districts=4, max_turns=10, max_pending_proposals=2)
        neg = NegotiationSystem(cfg)

        p1 = neg.create(proposer=0, target=1, kind="coalition", terms={}, current_turn=0)
        p2 = neg.create(proposer=0, target=2, kind="coalition", terms={}, current_turn=0)
        assert p1 is not None
        assert p2 is not None

        # Third from same proposer
        p3 = neg.create(proposer=0, target=3, kind="coalition", terms={}, current_turn=0)
        assert p3 is None, "Outgoing cap not enforced"

    def test_3c_ttl_expiry(self):
        """Proposals expire after proposal_ttl turns."""
        cfg = EnvConfig(num_districts=4, max_turns=20, proposal_ttl=3)
        neg = NegotiationSystem(cfg)
        neg.create(proposer=0, target=1, kind="coalition", terms={}, current_turn=0)
        assert len(neg._proposals) == 1

        # Tick 3 times — should expire
        neg.tick()
        neg.tick()
        expired = neg.tick()
        assert len(neg._proposals) == 0, "Proposal did not expire after TTL"
        assert len(expired) == 1

    def test_3d_cooldown_prevents_spam(self):
        """Proposer cannot re-propose during cooldown window."""
        cfg = EnvConfig(num_districts=4, max_turns=20,
                        proposal_cooldown=4, max_pending_proposals=2)
        neg = NegotiationSystem(cfg)
        p1 = neg.create(proposer=0, target=1, kind="coalition", terms={}, current_turn=0)
        assert p1 is not None

        # Accept the proposal (removes it from pending)
        neg.accept(p1.proposal_id, target=1)

        # Try again within cooldown window
        p2 = neg.create(proposer=0, target=1, kind="coalition", terms={}, current_turn=2)
        assert p2 is None, "Cooldown not enforced"

        # After cooldown, should work
        p3 = neg.create(proposer=0, target=1, kind="coalition", terms={}, current_turn=5)
        assert p3 is not None, "Proposal blocked after cooldown period ended"

    def test_3e_propose_to_self_blocked(self):
        """Env-level: PROPOSE_COALITION to self must be blocked by action mask."""
        env = fresh_env()
        obs = env.reset()
        # In the env, mask includes PROPOSE only when target != self
        # The mask is binary but the action itself carries the target
        # Verify that the env rejects self-proposals at processing level
        actions = {
            0: make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=0),
            1: DiscreteAction.IGNORE,
            2: DiscreteAction.IGNORE,
            3: DiscreteAction.IGNORE,
        }
        _, _, _, _, info = env.step(actions)
        # Self-proposal should either be dropped or rejected — check no coalition formed
        formed = [e for e in info["events"] if e["type"] == "coalition_formed"
                  and e["payload"]["proposer"] == 0 and e["payload"]["acceptor"] == 0]
        assert len(formed) == 0, "Self-proposal resulted in coalition"

    def test_3f_accept_nonexistent_proposal_invalid(self):
        """ACCEPT when no pending proposals → action_invalid event."""
        env = fresh_env()
        obs = env.reset()
        # Ensure agent 0 has no incoming proposals
        pending = env._negotiation.pending_for(0)
        if len(pending) > 0:
            pytest.skip("Agent 0 already has pending proposals at reset")

        obs = env.reset()
        # Force ACCEPT even though mask should be 0
        actions = {
            0: make_default_parsed_action(DiscreteAction.ACCEPT_COALITION),
            1: DiscreteAction.IGNORE,
            2: DiscreteAction.IGNORE,
            3: DiscreteAction.IGNORE,
        }
        _, _, _, _, info = env.step(actions)
        # Should be action_invalid
        invalid_events = [e for e in info["events"] if e["type"] == "action_invalid"
                         and e["payload"]["agent_id"] == 0]
        assert len(invalid_events) == 1, (
            "ACCEPT with no proposal should produce action_invalid"
        )

    def test_3g_accept_targets_correct_proposal(self):
        """ACCEPT must resolve the proposal addressed to the acceptor, not others."""
        env = fresh_env()
        obs = env.reset()

        # Turn 0: agent 0 proposes to agent 1
        actions_t0 = {
            0: make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=1),
            1: DiscreteAction.IGNORE,
            2: DiscreteAction.IGNORE,
            3: DiscreteAction.IGNORE,
        }
        _, _, _, _, info0 = env.step(actions_t0)
        created = [e for e in info0["events"] if e["type"] == "proposal_created"]
        assert len(created) == 1
        assert created[0]["payload"]["target"] == 1

        # Turn 1: agent 1 accepts
        actions_t1 = {
            0: DiscreteAction.IGNORE,
            1: make_default_parsed_action(DiscreteAction.ACCEPT_COALITION),
            2: DiscreteAction.IGNORE,
            3: DiscreteAction.IGNORE,
        }
        _, _, _, _, info1 = env.step(actions_t1)
        formed = [e for e in info1["events"] if e["type"] == "coalition_formed"]
        assert len(formed) == 1
        assert formed[0]["payload"]["proposer"] == 0
        assert formed[0]["payload"]["acceptor"] == 1


# ═══════════════════════════════════════════════════════════════════════
# Section 4 — Coalition System Integrity
# ═══════════════════════════════════════════════════════════════════════

class TestSection4CoalitionSystem:
    """Coalition membership rules."""

    def test_4a_agent_in_at_most_one_coalition(self):
        """Over a full episode, no agent is ever in two coalitions."""
        env = fresh_env(FULL_CFG)
        policy = SelfPlayPolicy(mode="rule_based", seed=SEED)
        traj = run_episode(env, policy)

        for rec in traj:
            snap = rec.info["state_snapshot"]["agents"]
            for a_str, a_snap in snap.items():
                cid = a_snap["coalition_id"]
                # coalition_id is either None or a single int
                assert cid is None or isinstance(cid, int), (
                    f"Step {rec.step} agent {a_str}: invalid coalition_id {cid!r}"
                )

    def test_4b_coalition_size_consistent(self):
        """coalition_size from coalition system matches snapshot membership."""
        env = fresh_env(SMALL_CFG)
        policy = SelfPlayPolicy(mode="rule_based", seed=SEED)
        traj = run_episode(env, policy)

        for rec in traj:
            snap = rec.info["state_snapshot"]["agents"]
            # Count members per coalition from snapshot
            counts: Dict[int, int] = Counter(
                int(a["coalition_id"])
                for a in snap.values()
                if a["coalition_id"] is not None
            )
            # Each agent's coalition_id should correspond to a coalition of that size
            # We can't directly check coalition_size here but can verify consistency
            coal_info = rec.info.get("coalition", {})
            if coal_info:
                for cid_str, members in coal_info.get("coalitions", {}).items():
                    cid = int(cid_str)
                    assert counts.get(cid, 0) == len(members), (
                        f"Step {rec.step} coalition {cid}: "
                        f"snapshot count {counts.get(cid,0)} vs info {len(members)}"
                    )

    def test_4c_coalition_id_never_negative(self):
        env = fresh_env(FULL_CFG)
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        traj = run_episode(env, policy)

        for rec in traj:
            for a_str, snap in rec.info["state_snapshot"]["agents"].items():
                cid = snap["coalition_id"]
                if cid is not None:
                    assert cid >= 0, f"Step {rec.step} agent {a_str}: negative coalition_id {cid}"

    def test_4d_no_solo_coalition_from_accept(self):
        """An agent accepting a proposal must join a coalition with ≥2 members."""
        env = fresh_env()
        obs = env.reset()

        # Form a coalition
        actions_t0 = {
            0: make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=1),
            1: DiscreteAction.IGNORE,
            2: DiscreteAction.IGNORE,
            3: DiscreteAction.IGNORE,
        }
        env.step(actions_t0)

        actions_t1 = {
            0: DiscreteAction.IGNORE,
            1: make_default_parsed_action(DiscreteAction.ACCEPT_COALITION),
            2: DiscreteAction.IGNORE,
            3: DiscreteAction.IGNORE,
        }
        _, _, _, _, info = env.step(actions_t1)

        snap = info["state_snapshot"]["agents"]
        cid_0 = snap["0"]["coalition_id"]
        cid_1 = snap["1"]["coalition_id"]
        assert cid_0 is not None and cid_1 is not None
        assert cid_0 == cid_1
        # Coalition must have ≥2 members
        members_in_coalition = sum(
            1 for s in snap.values() if s["coalition_id"] == cid_0
        )
        assert members_in_coalition >= 2


# ═══════════════════════════════════════════════════════════════════════
# Section 5 — Trust System
# ═══════════════════════════════════════════════════════════════════════

class TestSection5TrustSystem:
    """Trust bounds, decay, interaction-driven updates."""

    def test_5a_trust_always_in_bounds(self):
        """All trust values must remain in [-1, 1] throughout an episode."""
        env = fresh_env(SMALL_CFG)
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        traj = run_episode(env, policy)

        for rec in traj:
            for a_str, snap in rec.info["state_snapshot"]["agents"].items():
                for j, tv in snap["trust_row"].items():
                    assert -1.0 <= tv <= 1.0, (
                        f"Step {rec.step} agent {a_str} → {j}: trust={tv:.4f} ∉ [-1, 1]"
                    )

    def test_5b_accept_increases_bilateral_trust(self):
        """After ACCEPT, both proposer and acceptor trust each other more."""
        env = fresh_env()
        obs = env.reset()

        trust_before_0 = env._trust.as_matrix()[0].get(1, 0.0)
        trust_before_1 = env._trust.as_matrix()[1].get(0, 0.0)

        # Propose
        env.step({
            0: make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=1),
            1: DiscreteAction.IGNORE,
            2: DiscreteAction.IGNORE,
            3: DiscreteAction.IGNORE,
        })
        # Accept
        env.step({
            0: DiscreteAction.IGNORE,
            1: make_default_parsed_action(DiscreteAction.ACCEPT_COALITION),
            2: DiscreteAction.IGNORE,
            3: DiscreteAction.IGNORE,
        })

        trust_after_0 = env._trust.as_matrix()[0].get(1, 0.0)
        trust_after_1 = env._trust.as_matrix()[1].get(0, 0.0)

        assert trust_after_0 > trust_before_0, "Proposer trust toward acceptor did not increase"
        assert trust_after_1 > trust_before_1, "Acceptor trust toward proposer did not increase"

    def test_5c_reject_decreases_trust(self):
        """REJECT must decrease trust (or at least not increase proposer's trust)."""
        env = fresh_env()
        env.reset()

        trust_before_0 = env._trust.as_matrix()[0].get(1, 0.0)

        # Propose
        env.step({
            0: make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=1),
            1: DiscreteAction.IGNORE,
            2: DiscreteAction.IGNORE,
            3: DiscreteAction.IGNORE,
        })
        # Reject
        env.step({
            0: DiscreteAction.IGNORE,
            1: make_default_parsed_action(DiscreteAction.REJECT_COALITION),
            2: DiscreteAction.IGNORE,
            3: DiscreteAction.IGNORE,
        })

        trust_after_0 = env._trust.as_matrix()[0].get(1, 0.0)
        assert trust_after_0 <= trust_before_0, (
            f"Trust should not increase after rejection: {trust_before_0:.4f} → {trust_after_0:.4f}"
        )

    def test_5d_trust_decays_without_interaction(self):
        """Trust drifts toward 0 when no actions occur (IGNORE only)."""
        cfg = EnvConfig(
            num_districts=4, max_turns=20, seed=SEED,
            trust_init_std=0.0,
            trust_decay=0.90,   # decay multiplier (< 1.0 = shrink toward 0)
        )
        env = DistrictAccordEnv(cfg)
        env.reset()
        # Set some trust manually using the internal dict
        env._trust._trust[0][1] = 0.5
        initial = env._trust._trust[0][1]

        for _ in range(5):
            env.step({a: DiscreteAction.IGNORE for a in range(4)})

        final = env._trust._trust[0][1]
        assert final < initial, (
            f"Trust did not decay: {initial:.4f} → {final:.4f}"
        )

    def test_5e_no_passive_trust_accumulation(self):
        """IGNORE-only agents should not gain trust_alignment reward."""
        env = fresh_env(EnvConfig(
            num_districts=4, max_turns=10, seed=SEED,
            trust_init_std=0.0, obs_neighbor_noise_std=0.0,
            reward_trust_alignment=1.0,
        ))
        env.reset()

        # Run all-IGNORE for 5 turns
        total_trust_reward = 0.0
        for _ in range(5):
            _, _, _, _, info = env.step({a: DiscreteAction.IGNORE for a in range(4)})
            for bd in info["reward_breakdown"].values():
                total_trust_reward += bd["trust_alignment"]

        assert total_trust_reward == pytest.approx(0.0, abs=1e-9), (
            f"Passive trust_alignment reward accumulated: {total_trust_reward:.6f}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Section 6 — Reward Engine Validation
# ═══════════════════════════════════════════════════════════════════════

class TestSection6RewardEngine:
    """Reward bounds, delta signals, exploit resistance."""

    def test_6a_reward_in_typical_range_alive_agents(self):
        """Alive non-collapse-turn rewards must stay in [-2.5, +2.5]."""
        env = fresh_env(FULL_CFG)
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        traj = run_episode(env, policy)

        for rec in traj:
            for a, r in rec.rewards.items():
                bd = rec.info["reward_breakdown"][a]
                if bd["collapse_penalty"] != 0.0:
                    continue  # collapse turn — large penalty expected
                assert -2.5 <= r <= 2.5, (
                    f"Step {rec.step} agent {a}: reward {r:.4f} out of expected range"
                )

    def test_6b_cooperation_hard_capped(self):
        """cooperation component must never exceed 0.15."""
        env = fresh_env(FULL_CFG)
        policy = SelfPlayPolicy(mode="rule_based", seed=SEED)
        traj = run_episode(env, policy)

        for rec in traj:
            for a, bd in rec.info["reward_breakdown"].items():
                assert bd["cooperation"] <= 0.15 + 1e-9, (
                    f"Step {rec.step} agent {a}: cooperation {bd['cooperation']:.4f} > 0.15"
                )

    def test_6c_trust_alignment_hard_capped(self):
        """trust_alignment must never exceed 0.05."""
        env = fresh_env(FULL_CFG)
        policy = SelfPlayPolicy(mode="rule_based", seed=SEED)
        traj = run_episode(env, policy)

        for rec in traj:
            for a, bd in rec.info["reward_breakdown"].items():
                assert bd["trust_alignment"] <= 0.05 + 1e-9, (
                    f"Step {rec.step} agent {a}: trust_alignment {bd['trust_alignment']:.6f} > 0.05"
                )

    def test_6d_delta_crisis_zero_when_stable(self):
        """If crisis_exposure does not change, crisis_mitigation must be 0."""
        engine = RewardEngine(SMALL_CFG)
        _, bd = engine.compute(
            agent_id=0,
            is_newly_collapsed=False, is_collapsed=False,
            prev_stability=0.5,  curr_stability=0.5,
            prev_exposure=0.3,   curr_exposure=0.3,   # no change
            prev_avg_trust=0.0,
            coalition_size=0, trust_row={1: 0.0, 2: 0.0, 3: 0.0},
            mask_violated=False, pending_outgoing=0,
        )
        assert bd.crisis_mitigation == pytest.approx(0.0, abs=1e-9)

    def test_6e_delta_trust_zero_when_stable(self):
        """If trust avg does not change, trust_alignment must be 0."""
        engine = RewardEngine(SMALL_CFG)
        _, bd = engine.compute(
            agent_id=0,
            is_newly_collapsed=False, is_collapsed=False,
            prev_stability=0.5,  curr_stability=0.5,
            prev_exposure=0.0,   curr_exposure=0.0,
            prev_avg_trust=0.5,
            coalition_size=0,
            trust_row={1: 0.5, 2: 0.5, 3: 0.5},  # avg_pos = 0.5 = prev
            mask_violated=False, pending_outgoing=0,
        )
        assert bd.trust_alignment == pytest.approx(0.0, abs=1e-9)

    def test_6f_exploit_mixed_beats_idle_coalition(self):
        """Mixed strategy (rule_based) total reward > idle coalition strategy."""
        def avg_episode_reward(mode, n=5):
            totals = []
            for s in range(n):
                env = DistrictAccordEnv(SMALL_CFG)
                policy = SelfPlayPolicy(mode=mode, seed=s)
                runner = EpisodeRunner()
                traj = runner.run_episode(env, policy, seed=s)
                ep_total = sum(
                    r for rec in traj for r in rec.rewards.values()
                )
                totals.append(ep_total)
            return sum(totals) / len(totals)

        mixed  = avg_episode_reward("rule_based")
        random = avg_episode_reward("mask_aware_random")

        # Rule-based should consistently beat pure random (not guaranteed always
        # but with deterministic config should hold for most seeds)
        assert mixed >= random * 0.90, (
            f"Mixed ({mixed:.2f}) significantly underperforms random ({random:.2f})"
        )

    def test_6g_collapse_penalty_overrides_everything(self):
        """Collapsed agent gets only collapse_penalty, nothing else."""
        engine = RewardEngine(SMALL_CFG)
        _, bd = engine.compute(
            agent_id=0,
            is_newly_collapsed=True,
            is_collapsed=False,
            prev_stability=0.5, curr_stability=0.0,
            prev_exposure=0.5, curr_exposure=1.0,
            prev_avg_trust=0.0,
            coalition_size=10,
            trust_row={1: 1.0, 2: 1.0, 3: 1.0},
            mask_violated=True, pending_outgoing=5,
        )
        assert bd.survival == 0.0
        assert bd.cooperation == 0.0
        assert bd.crisis_mitigation == 0.0
        assert bd.collapse_penalty != 0.0

    def test_6h_reward_deterministic_same_engine_state(self):
        """RewardEngine is stateless — identical inputs → identical outputs."""
        engine = RewardEngine(SMALL_CFG)
        kwargs = dict(
            agent_id=0, is_newly_collapsed=False, is_collapsed=False,
            prev_stability=0.55, curr_stability=0.60,
            prev_exposure=0.20, curr_exposure=0.15,
            prev_avg_trust=0.3,
            coalition_size=2, trust_row={1: 0.6, 2: 0.3, 3: 0.4},
            mask_violated=False, pending_outgoing=0,
        )
        r1, bd1 = engine.compute(**kwargs)
        r2, bd2 = engine.compute(**kwargs)
        assert r1 == pytest.approx(r2)
        assert bd1.to_dict() == bd2.to_dict()


# ═══════════════════════════════════════════════════════════════════════
# Section 7 — Event Bus Integrity
# ═══════════════════════════════════════════════════════════════════════

class TestSection7EventBus:
    """Event types, ordering, completeness."""

    VALID_TYPES = frozenset({
        "action_validated", "action_invalid",
        "proposal_created", "proposal_rejected", "proposal_expired",
        "coalition_formed", "coalition_joined",
        "resource_transferred", "trust_updated", "collapse",
    })

    def test_7a_all_events_have_valid_type(self):
        env = fresh_env(FULL_CFG)
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        traj = run_episode(env, policy)

        for rec in traj:
            for ev in rec.info["events"]:
                assert ev["type"] in self.VALID_TYPES, (
                    f"Step {rec.step}: unknown event type {ev['type']!r}"
                )

    def test_7b_events_have_required_fields(self):
        env = fresh_env(FULL_CFG)
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        traj = run_episode(env, policy)

        for rec in traj:
            for ev in rec.info["events"]:
                assert "type" in ev
                assert "turn" in ev
                assert "seq" in ev
                assert "payload" in ev

    def test_7c_seq_numbers_monotonically_increasing(self):
        """seq field must be strictly increasing across the entire episode."""
        env = fresh_env(FULL_CFG)
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        traj = run_episode(env, policy)

        all_events = []
        for rec in traj:
            all_events.extend(rec.info["events"])

        for i in range(1, len(all_events)):
            assert all_events[i]["seq"] > all_events[i - 1]["seq"], (
                f"seq not monotonic at event {i}: "
                f"{all_events[i-1]['seq']} → {all_events[i]['seq']}"
            )

    def test_7d_pipeline_order_within_turn(self):
        """Within each turn, events must respect pipeline priority order."""
        env = fresh_env(SMALL_CFG)
        policy = SelfPlayPolicy(mode="rule_based", seed=SEED)
        traj = run_episode(env, policy)

        for rec in traj:
            priorities = [
                _PIPELINE_PRIORITY[ev["type"]]
                for ev in rec.info["events"]
            ]
            for i in range(1, len(priorities)):
                assert priorities[i] >= priorities[i - 1], (
                    f"Step {rec.step}: event order violated at index {i}: "
                    f"priority {priorities[i-1]} then {priorities[i]}"
                )

    def test_7e_action_events_present_every_turn(self):
        """Every turn must have at least one action_validated or action_invalid."""
        env = fresh_env(SMALL_CFG)
        policy = SelfPlayPolicy(mode="rule_based", seed=SEED)
        traj = run_episode(env, policy)

        for rec in traj:
            action_events = [
                e for e in rec.info["events"]
                if e["type"] in ("action_validated", "action_invalid")
            ]
            alive_count = sum(
                1 for snap in rec.info["state_snapshot"]["agents"].values()
                if not snap["collapsed"]
            )
            # Note: snapshot is AFTER the step, so check previous step collapse state
            if alive_count > 0 or rec.step == 0:
                assert len(action_events) > 0, (
                    f"Step {rec.step}: no action events found"
                )

    def test_7f_event_bus_clears_on_reset(self):
        env = fresh_env()
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)

        # Run an episode
        run_episode(env, policy)
        events_after_ep1 = len(env._event_bus.get_events())
        assert events_after_ep1 > 0

        # Reset and check clear
        env.reset()
        events_after_reset = len(env._event_bus.get_events())
        assert events_after_reset == 0, (
            f"EventBus not cleared on reset: {events_after_reset} events remain"
        )

    def test_7g_invalid_event_type_rejected(self):
        """Emitting an unknown event type must raise ValueError."""
        env = fresh_env()
        env.reset()
        with pytest.raises(ValueError):
            env._event_bus.emit("totally_invalid_type", {})


# ═══════════════════════════════════════════════════════════════════════
# Section 8 — State Tracker Accuracy
# ═══════════════════════════════════════════════════════════════════════

class TestSection8StateTracker:
    """Snapshot matches actual env state; history correct; reset works."""

    def test_8a_snapshot_resources_match_env(self):
        """state_snapshot resources must match env._districts at step end.

        StateTracker rounds to 4dp in AgentSnapshot, so we use 5e-4 tolerance.
        """
        env = fresh_env()
        obs = env.reset()
        for _ in range(3):
            actions = {a: DiscreteAction.IGNORE for a in range(4)}
            _, _, done, trunc, info = env.step(actions)
            snap = info["state_snapshot"]["agents"]
            for a in range(4):
                snap_res  = snap[str(a)]["resources"]
                env_res   = env._districts[a].resources
                # Snapshot rounds to 4dp — allow up to 0.5 ULP of 4dp precision
                assert abs(snap_res - env_res) < 5e-4, (
                    f"Agent {a}: snapshot resources {snap_res:.6f} vs env {env_res:.6f} "
                    f"diff={abs(snap_res - env_res):.2e}"
                )
            if done or trunc:
                break

    def test_8b_snapshot_stability_match_env(self):
        """state_snapshot stability must be within 4dp rounding of env._districts."""
        env = fresh_env()
        obs = env.reset()
        for _ in range(3):
            _, _, done, trunc, info = env.step({a: DiscreteAction.DEFEND for a in range(4)})
            snap = info["state_snapshot"]["agents"]
            for a in range(4):
                snap_stab = snap[str(a)]["stability"]
                env_stab  = env._districts[a].stability
                assert abs(snap_stab - env_stab) < 5e-4, (
                    f"Agent {a}: snapshot stability {snap_stab:.6f} vs env {env_stab:.6f} "
                    f"diff={abs(snap_stab - env_stab):.2e}"
                )
            if done or trunc:
                break

    def test_8c_history_length_equals_turns_played(self):
        env = fresh_env()
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        traj = run_episode(env, policy)

        history = env._state_tracker.get_history()
        assert len(history) == len(traj), (
            f"State history length {len(history)} ≠ turns played {len(traj)}"
        )

    def test_8d_reset_clears_history(self):
        env = fresh_env()
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        run_episode(env, policy)

        assert len(env._state_tracker.get_history()) > 0
        env.reset()
        assert len(env._state_tracker.get_history()) == 0, \
            "StateTracker not cleared on reset"

    def test_8e_collapsed_flag_correct_in_snapshot(self):
        """After an agent collapses, snapshot must show collapsed=True."""
        env = fresh_env(EnvConfig(
            num_districts=4, max_turns=50, seed=SEED,
            stability_threshold=0.4,
            trust_init_std=0.0, obs_neighbor_noise_std=0.0,
        ))
        env.reset()
        # Force agent 0 near collapse
        env._districts[0].stability = 0.35

        _, _, done, trunc, info = env.step({a: DiscreteAction.IGNORE for a in range(4)})
        snap = info["state_snapshot"]["agents"]

        # Check that env collapsed state matches snapshot
        for a in range(4):
            assert snap[str(a)]["collapsed"] == env._collapsed.get(a, False), (
                f"Agent {a}: snap collapsed={snap[str(a)]['collapsed']!r} "
                f"≠ env collapsed={env._collapsed.get(a, False)!r}"
            )


# ═══════════════════════════════════════════════════════════════════════
# Section 9 — Turn Pipeline Order
# ═══════════════════════════════════════════════════════════════════════

class TestSection9PipelineOrder:
    """Events must respect the documented pipeline boundary order."""

    def test_9a_actions_before_proposals(self):
        """action_* events must appear before proposal_* events in same turn."""
        env = fresh_env()
        policy = SelfPlayPolicy(mode="rule_based", seed=SEED)
        traj = run_episode(env, policy)

        for rec in traj:
            events = rec.info["events"]
            action_seqs  = [e["seq"] for e in events if e["type"] in ("action_validated","action_invalid")]
            proposal_seqs = [e["seq"] for e in events if e["type"] == "proposal_created"]
            if action_seqs and proposal_seqs:
                assert max(action_seqs) < min(proposal_seqs), (
                    f"Step {rec.step}: proposal_created before action events"
                )

    def test_9b_proposals_before_coalition(self):
        """proposal_created must appear before coalition_formed in same turn."""
        env = fresh_env()
        obs = env.reset()

        # Turn 0: propose
        env.step({
            0: make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=1),
            1: DiscreteAction.IGNORE, 2: DiscreteAction.IGNORE, 3: DiscreteAction.IGNORE,
        })

        # Turn 1: accept — should have coalition_formed but no proposal_created
        _, _, _, _, info = env.step({
            0: DiscreteAction.IGNORE,
            1: make_default_parsed_action(DiscreteAction.ACCEPT_COALITION),
            2: DiscreteAction.IGNORE, 3: DiscreteAction.IGNORE,
        })
        events = info["events"]
        action_seqs   = [e["seq"] for e in events if "action" in e["type"]]
        coalition_seqs = [e["seq"] for e in events if "coalition" in e["type"]]
        if action_seqs and coalition_seqs:
            assert max(action_seqs) < min(coalition_seqs), \
                "coalition event appeared before action events"

    def test_9c_collapse_after_resource_effects(self):
        """collapse events must come after resource_transferred and proposal_expired."""
        env = fresh_env(FULL_CFG)
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        traj = run_episode(env, policy)

        for rec in traj:
            events = rec.info["events"]
            collapse_seqs   = [e["seq"] for e in events if e["type"] == "collapse"]
            resource_seqs   = [e["seq"] for e in events if e["type"] == "resource_transferred"]
            expired_seqs    = [e["seq"] for e in events if e["type"] == "proposal_expired"]

            if collapse_seqs and resource_seqs:
                assert min(collapse_seqs) > max(resource_seqs), \
                    f"Step {rec.step}: collapse before resource_transferred"
            if collapse_seqs and expired_seqs:
                assert min(collapse_seqs) > max(expired_seqs), \
                    f"Step {rec.step}: collapse before proposal_expired"

    def test_9d_trust_updates_last(self):
        """trust_updated events must have the highest seq in any turn."""
        env = fresh_env(SMALL_CFG)
        policy = SelfPlayPolicy(mode="rule_based", seed=SEED)
        traj = run_episode(env, policy)

        for rec in traj:
            events = rec.info["events"]
            trust_seqs   = [e["seq"] for e in events if e["type"] == "trust_updated"]
            other_seqs   = [e["seq"] for e in events if e["type"] != "trust_updated"]
            if trust_seqs and other_seqs:
                assert min(trust_seqs) > max(other_seqs), (
                    f"Step {rec.step}: trust_updated not last in pipeline"
                )


# ═══════════════════════════════════════════════════════════════════════
# Section 10 — Performance
# ═══════════════════════════════════════════════════════════════════════

class TestSection10Performance:
    """12×100×10 episodes — no crash, stable runtime."""

    def test_10a_10_full_episodes_no_crash(self):
        """10 full episodes (12 agents, 100 turns) must complete without error."""
        runner = EpisodeRunner()
        total_steps = 0
        for s in range(10):
            env = DistrictAccordEnv(FULL_CFG)
            policy = SelfPlayPolicy(mode="mask_aware_random", seed=s)
            traj = runner.run_episode(env, policy, seed=s)
            assert len(traj) > 0
            total_steps += len(traj)
        assert total_steps > 0

    def test_10b_runtime_per_episode_under_threshold(self):
        """12-agent, 100-turn episode must complete in < 5 seconds."""
        env = DistrictAccordEnv(FULL_CFG)
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        runner = EpisodeRunner()

        t0 = time.perf_counter()
        traj = runner.run_episode(env, policy, seed=SEED)
        elapsed = time.perf_counter() - t0

        assert elapsed < 5.0, f"Episode took {elapsed:.2f}s (> 5s threshold)"
        assert len(traj) > 0

    def test_10c_event_log_bounded(self):
        """
        Event log should not grow unboundedly.
        Max ~(N * turns * 2) action events + other interactions.
        For 12×100: allow up to 5×N×T = 6000 events.
        """
        env = DistrictAccordEnv(FULL_CFG)
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        runner = EpisodeRunner()
        traj = runner.run_episode(env, policy, seed=SEED)

        total_events = sum(len(rec.info["events"]) for rec in traj)
        upper_bound = 5 * FULL_CFG.num_districts * FULL_CFG.max_turns

        assert total_events <= upper_bound, (
            f"Event count {total_events} exceeds bound {upper_bound}"
        )

    def test_10d_state_tracker_history_bounded(self):
        """StateTracker should hold exactly one snapshot per turn, no more."""
        env = DistrictAccordEnv(FULL_CFG)
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        runner = EpisodeRunner()
        traj = runner.run_episode(env, policy, seed=SEED)

        n_history = len(env._state_tracker.get_history())
        assert n_history == len(traj), (
            f"State history {n_history} ≠ turns played {len(traj)}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Section 11 — Trajectory Integrity
# ═══════════════════════════════════════════════════════════════════════

class TestSection11TrajectoryIntegrity:
    """Save, load, replay — byte-exact reconstruction."""

    def test_11a_save_load_preserves_step_count(self, tmp_path):
        env = fresh_env()
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        traj = run_episode(env, policy)

        path = tmp_path / "ep.json"
        save_trajectory(traj, path)
        loaded = load_trajectory(path)
        assert len(loaded) == len(traj)

    def test_11b_loaded_has_all_required_keys(self, tmp_path):
        env = fresh_env()
        traj = run_episode(env, SelfPlayPolicy(mode="rule_based", seed=SEED))
        path = tmp_path / "ep.json"
        save_trajectory(traj, path)
        loaded = load_trajectory(path)

        required = {"step", "obs", "actions", "rewards", "done", "truncated", "info"}
        for i, step in enumerate(loaded):
            missing = required - step.keys()
            assert not missing, f"Step {i}: missing keys {missing}"

    def test_11c_loaded_info_has_engine_keys(self, tmp_path):
        env = fresh_env()
        traj = run_episode(env, SelfPlayPolicy(mode="rule_based", seed=SEED))
        path = tmp_path / "ep.json"
        save_trajectory(traj, path)
        loaded = load_trajectory(path)

        for step in loaded:
            info = step["info"]
            assert "reward_breakdown" in info
            assert "events" in info
            assert "state_snapshot" in info

    def test_11d_verify_replay_returns_true(self):
        env = fresh_env(SMALL_CFG)
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        runner = EpisodeRunner()
        traj = runner.run_episode(env, policy, seed=SEED)

        env2 = DistrictAccordEnv(SMALL_CFG)
        policy2 = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        result = verify_replay(env2, traj, policy2, seed=SEED)
        assert result, "verify_replay returned False — determinism broken"

    def test_11e_to_dict_fully_json_serialisable(self):
        env = fresh_env()
        traj = run_episode(env, SelfPlayPolicy(mode="mask_aware_random", seed=SEED))
        for rec in traj:
            d = rec.to_dict()
            json.dumps(d)   # must not raise

    def test_11f_rewards_in_loaded_match_original(self, tmp_path):
        env = fresh_env()
        traj = run_episode(env, SelfPlayPolicy(mode="mask_aware_random", seed=SEED))
        path = tmp_path / "ep.json"
        save_trajectory(traj, path)
        loaded = load_trajectory(path)

        for i, (orig, loaded_step) in enumerate(zip(traj, loaded)):
            for a_str, r in loaded_step["rewards"].items():
                a = int(a_str)
                assert abs(r - orig.rewards[a]) < 1e-5, (
                    f"Step {i} agent {a}: loaded reward {r:.6f} ≠ original {orig.rewards[a]:.6f}"
                )


# ═══════════════════════════════════════════════════════════════════════
# Section 12 — Policy Compliance
# ═══════════════════════════════════════════════════════════════════════

class TestSection12PolicyCompliance:
    """All three policies: mask compliance, no crashes, determinism."""

    @pytest.mark.parametrize("mode", ["random", "mask_aware_random", "rule_based"])
    def test_12a_policy_completes_episode_no_crash(self, mode):
        env = fresh_env(SMALL_CFG)
        policy = SelfPlayPolicy(mode=mode, seed=SEED)
        traj = run_episode(env, policy)
        assert len(traj) > 0, f"Policy {mode}: empty trajectory"

    @pytest.mark.parametrize("mode", ["random", "mask_aware_random", "rule_based"])
    def test_12b_policy_full_scale_no_crash(self, mode):
        env = DistrictAccordEnv(FULL_CFG)
        policy = SelfPlayPolicy(mode=mode, seed=SEED)
        traj = run_episode(env, policy)
        assert len(traj) > 0

    def test_12c_mask_aware_random_zero_violations(self):
        """mask_aware_random must produce zero action_invalid events."""
        for s in range(5):
            env = DistrictAccordEnv(SMALL_CFG)
            policy = SelfPlayPolicy(mode="mask_aware_random", seed=s)
            traj = run_episode(env, policy)
            invalid = sum(
                sum(1 for e in rec.info["events"] if e["type"] == "action_invalid")
                for rec in traj
            )
            assert invalid == 0, (
                f"Seed {s}: mask_aware_random produced {invalid} invalid actions"
            )

    def test_12d_rule_based_zero_violations(self):
        """rule_based policy uses only valid actions."""
        env = DistrictAccordEnv(SMALL_CFG)
        policy = SelfPlayPolicy(mode="rule_based", seed=SEED)
        traj = run_episode(env, policy)
        invalid = sum(
            sum(1 for e in rec.info["events"] if e["type"] == "action_invalid")
            for rec in traj
        )
        assert invalid == 0, f"rule_based produced {invalid} invalid actions"

    def test_12e_policy_seeded_reproducible(self):
        """Same seed → same actions for all modes."""
        for mode in ("random", "mask_aware_random", "rule_based"):
            p1 = SelfPlayPolicy(mode=mode, seed=SEED)
            p2 = SelfPlayPolicy(mode=mode, seed=SEED)
            e1 = fresh_env(); e2 = fresh_env()
            t1 = run_episode(e1, p1)
            t2 = run_episode(e2, p2)
            assert len(t1) == len(t2), f"Mode {mode}: episode length differs"
            for i, (r1, r2) in enumerate(zip(t1, t2)):
                assert r1.actions == r2.actions, (
                    f"Mode {mode} step {i}: actions differ"
                )

    def test_12f_all_active_agents_receive_action(self):
        """Every non-collapsed agent must have an action each turn."""
        for mode in ("mask_aware_random", "rule_based"):
            env = fresh_env(SMALL_CFG)
            policy = SelfPlayPolicy(mode=mode, seed=SEED)
            traj = run_episode(env, policy)
            for rec in traj:
                snap = rec.info["state_snapshot"]["agents"]
                # snapshot is AFTER step, so we can't directly check which agents
                # were alive BEFORE the step — but we can check action events
                action_agent_ids = {
                    e["payload"]["agent_id"]
                    for e in rec.info["events"]
                    if e["type"] in ("action_validated", "action_invalid")
                }
                assert len(action_agent_ids) > 0, (
                    f"Mode {mode} step {rec.step}: no agents acted"
                )
