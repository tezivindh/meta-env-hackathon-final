"""
tests/test_phase6_training.py
================================
Phase 6 tests: full-scale env, self-play policies, episode runner,
trajectories, deterministic replay, and delta-based reward signals.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

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
from district_accord.utils.config import EnvConfig
from district_accord.utils.types import DiscreteAction


# ─── Configs ─────────────────────────────────────────────────────────────────

SEED = 99

FULL_CFG = EnvConfig(
    num_districts=12,
    max_turns=100,
    seed=SEED,
    trust_init_std=0.0,
    obs_neighbor_noise_std=0.0,
)

SMALL_CFG = EnvConfig(
    num_districts=4,
    max_turns=20,
    seed=SEED,
    trust_init_std=0.0,
    obs_neighbor_noise_std=0.0,
    reward_spam_penalty=0.0,
)


def make_env(cfg: EnvConfig = SMALL_CFG) -> DistrictAccordEnv:
    env = DistrictAccordEnv(cfg)
    env.reset()
    return env


# ─── Full-scale stability ─────────────────────────────────────────────────────

class TestFullScaleEnvironment:

    def test_12_agents_100_turns_no_crash(self):
        """12 agents, 100 turns — must complete without error."""
        env = DistrictAccordEnv(FULL_CFG)
        env.reset()
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        runner = EpisodeRunner()
        traj = runner.run_episode(env, policy, seed=SEED)
        assert len(traj) > 0
        assert len(traj) <= FULL_CFG.max_turns

    def test_12_agents_observation_shapes_correct(self):
        """obs["flat"] must have dimension 4N+4 = 52 for N=12."""
        env = DistrictAccordEnv(FULL_CFG)
        obs = env.reset()
        N = FULL_CFG.num_districts
        expected_flat = 4 * N + 4
        for agent_id, agent_obs in obs.items():
            assert agent_obs["flat"].shape == (expected_flat,), (
                f"Agent {agent_id}: expected flat dim {expected_flat}, "
                f"got {agent_obs['flat'].shape}"
            )

    def test_12_agents_action_mask_correct_shape(self):
        env = DistrictAccordEnv(FULL_CFG)
        obs = env.reset()
        for agent_id, agent_obs in obs.items():
            assert agent_obs["action_mask"].shape == (9,)
            assert set(agent_obs["action_mask"]).issubset({0, 1})

    def test_all_loops_o_n(self):
        """Verify env does not hardcode agent counts (N=12 produces N rewards)."""
        env = DistrictAccordEnv(FULL_CFG)
        env.reset()
        actions = {a: DiscreteAction.IGNORE for a in range(12)}
        _, rewards, _, _, info = env.step(actions)
        assert len(rewards) == 12
        assert len(info["reward_breakdown"]) == 12
        assert len(info["state_snapshot"]["agents"]) == 12


# ─── SelfPlayPolicy ───────────────────────────────────────────────────────────

class TestSelfPlayPolicy:

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            SelfPlayPolicy(mode="unknown_mode")

    def test_valid_modes_accepted(self):
        for mode in ("random", "mask_aware_random", "rule_based"):
            p = SelfPlayPolicy(mode=mode)
            assert p.mode == mode

    def test_act_returns_action_per_active_agent(self):
        env = make_env()
        obs = env.reset()
        for mode in ("random", "mask_aware_random", "rule_based"):
            policy = SelfPlayPolicy(mode=mode, seed=SEED)
            actions = policy.act(obs, env)
            # All non-collapsed agents should have an action
            active = {a for a in range(4) if not env._collapsed.get(a, False)}
            assert set(actions.keys()) == active

    def test_mask_aware_random_respects_mask(self):
        """All chosen actions must have mask=1 for mask_aware_random."""
        env = make_env()
        obs = env.reset()
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)

        for _ in range(10):
            actions = policy.act(obs, env)
            for agent_id, parsed in actions.items():
                action_type = parsed["action_type"]
                mask = obs[agent_id]["action_mask"]
                assert mask[action_type.value] == 1, (
                    f"Agent {agent_id}: chose {action_type.name} but mask=0"
                )
            obs, _, done, trunc, _ = env.step(actions)
            if done or trunc:
                break

    def test_random_policy_seeded_deterministic(self):
        """Same seed → same action sequence."""
        env = make_env()
        obs = env.reset()

        p1 = SelfPlayPolicy(mode="random", seed=7)
        p2 = SelfPlayPolicy(mode="random", seed=7)

        for _ in range(5):
            a1 = p1.act(obs, env)
            a2 = p2.act(obs, env)
            for agent_id in a1:
                assert a1[agent_id]["action_type"] == a2[agent_id]["action_type"]
            obs, _, done, trunc, _ = env.step(a1)
            if done or trunc:
                break

    def test_rule_based_policy_runs_full_episode(self):
        env = DistrictAccordEnv(SMALL_CFG)
        policy = SelfPlayPolicy(mode="rule_based", seed=SEED)
        runner = EpisodeRunner()
        traj = runner.run_episode(env, policy, seed=SEED)
        assert len(traj) > 0

    def test_repr_contains_mode(self):
        p = SelfPlayPolicy(mode="rule_based")
        assert "rule_based" in repr(p)


# ─── EpisodeRunner ────────────────────────────────────────────────────────────

class TestEpisodeRunner:

    def test_trajectory_length_bounded_by_max_turns(self):
        env = make_env()
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        runner = EpisodeRunner()
        traj = runner.run_episode(env, policy, seed=SEED)
        assert 1 <= len(traj) <= SMALL_CFG.max_turns

    def test_each_step_record_has_required_fields(self):
        env = make_env()
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        runner = EpisodeRunner()
        traj = runner.run_episode(env, policy, seed=SEED)
        for rec in traj:
            assert isinstance(rec, StepRecord)
            assert isinstance(rec.obs, dict)
            assert isinstance(rec.actions, dict)
            assert isinstance(rec.rewards, dict)
            assert isinstance(rec.done, bool)
            assert isinstance(rec.truncated, bool)
            assert isinstance(rec.info, dict)

    def test_info_has_all_engine_keys(self):
        env = make_env()
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        runner = EpisodeRunner()
        traj = runner.run_episode(env, policy, seed=SEED)
        for rec in traj:
            assert "reward_breakdown" in rec.info
            assert "events" in rec.info
            assert "state_snapshot" in rec.info

    def test_step_indices_sequential(self):
        env = make_env()
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        runner = EpisodeRunner()
        traj = runner.run_episode(env, policy, seed=SEED)
        for i, rec in enumerate(traj):
            assert rec.step == i

    def test_last_step_done_or_truncated(self):
        env = make_env()
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        runner = EpisodeRunner()
        traj = runner.run_episode(env, policy, seed=SEED)
        last = traj[-1]
        assert last.done or last.truncated

    def test_action_names_are_strings(self):
        env = make_env()
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        runner = EpisodeRunner()
        traj = runner.run_episode(env, policy, seed=SEED)
        for rec in traj:
            for a, name in rec.actions.items():
                assert isinstance(name, str)

    def test_episode_summary_keys(self):
        env = make_env()
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        runner = EpisodeRunner()
        traj = runner.run_episode(env, policy, seed=SEED)
        summary = runner.episode_summary(traj)
        for key in ("turns_played", "total_rewards", "collapses",
                    "coalition_events", "total_events", "avg_reward_per_turn_per_agent"):
            assert key in summary

    def test_full_12_agent_trajectory_correct_rewards_keys(self):
        env = DistrictAccordEnv(FULL_CFG)
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        runner = EpisodeRunner()
        traj = runner.run_episode(env, policy, seed=SEED)
        for rec in traj:
            # all 12 agents in rewards each step
            assert len(rec.rewards) == 12


# ─── Deterministic Replay ─────────────────────────────────────────────────────

class TestDeterministicReplay:

    def test_same_seed_same_trajectory(self):
        """Two runners with same seed must produce identical trajectories."""
        policy1 = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        policy2 = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        runner = EpisodeRunner()

        env1 = DistrictAccordEnv(SMALL_CFG)
        env2 = DistrictAccordEnv(SMALL_CFG)

        t1 = runner.run_episode(env1, policy1, seed=SEED)
        t2 = runner.run_episode(env2, policy2, seed=SEED)

        assert len(t1) == len(t2)
        for r1, r2 in zip(t1, t2):
            for a in r1.rewards:
                assert abs(r1.rewards[a] - r2.rewards[a]) < 1e-9
            e1 = [e["type"] for e in r1.info["events"]]
            e2 = [e["type"] for e in r2.info["events"]]
            assert e1 == e2

    def test_verify_replay_returns_true(self):
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        runner = EpisodeRunner()
        env = DistrictAccordEnv(SMALL_CFG)
        traj = runner.run_episode(env, policy, seed=SEED)

        policy2 = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        env2 = DistrictAccordEnv(SMALL_CFG)
        result = verify_replay(env2, traj, policy2, seed=SEED)
        assert result is True

    def test_different_seeds_different_trajectories(self):
        runner = EpisodeRunner()
        p1 = SelfPlayPolicy(mode="mask_aware_random", seed=1)
        p2 = SelfPlayPolicy(mode="mask_aware_random", seed=2)
        env1 = DistrictAccordEnv(SMALL_CFG)
        env2 = DistrictAccordEnv(SMALL_CFG)
        t1 = runner.run_episode(env1, p1, seed=1)
        t2 = runner.run_episode(env2, p2, seed=2)
        # At least one step should differ (with overwhelming probability)
        any_diff = any(
            abs(r1.rewards.get(0, 0) - r2.rewards.get(0, 0)) > 1e-9
            for r1, r2 in zip(t1, t2)
        )
        assert any_diff


# ─── Save / Load Trajectory ───────────────────────────────────────────────────

class TestTrajectoryIO:

    def test_save_and_load_roundtrip(self, tmp_path):
        env = make_env()
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        runner = EpisodeRunner()
        traj = runner.run_episode(env, policy, seed=SEED)

        path = tmp_path / "test_episode.json"
        save_trajectory(traj, path)
        assert path.exists()

        loaded = load_trajectory(path)
        assert isinstance(loaded, list)
        assert len(loaded) == len(traj)

    def test_saved_json_has_required_keys(self, tmp_path):
        env = make_env()
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        runner = EpisodeRunner()
        traj = runner.run_episode(env, policy, seed=SEED)

        path = tmp_path / "test_ep.json"
        save_trajectory(traj, path)
        loaded = load_trajectory(path)

        first = loaded[0]
        for key in ("step", "obs", "actions", "rewards", "done", "truncated", "info"):
            assert key in first

    def test_to_dict_json_serialisable(self):
        env = make_env()
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=SEED)
        runner = EpisodeRunner()
        traj = runner.run_episode(env, policy, seed=SEED)

        # Should not raise
        for rec in traj:
            d = rec.to_dict()
            json.dumps(d)   # raises if not serialisable


# ─── Delta-based Reward Signals ──────────────────────────────────────────────

class TestDeltaRewardSignals:

    def _make_engine(self):
        return RewardEngine(EnvConfig(
            num_districts=4, max_turns=20, seed=0,
            reward_crisis_weight=1.0,
            reward_trust_alignment=1.0,
        ))

    def test_crisis_mitigation_zero_when_exposure_stable(self):
        """If exposure does not change, crisis_mitigation must be 0."""
        engine = self._make_engine()
        _, bd = engine.compute(
            agent_id=0,
            is_newly_collapsed=False,
            is_collapsed=False,
            prev_stability=0.5, curr_stability=0.5,
            prev_exposure=0.3,  curr_exposure=0.3,
            prev_avg_trust=0.0,
            coalition_size=0,
            trust_row={1: 0.0, 2: 0.0, 3: 0.0},
            mask_violated=False,
            pending_outgoing=0,
        )
        assert bd.crisis_mitigation == pytest.approx(0.0, abs=1e-9)

    def test_crisis_mitigation_positive_when_exposure_decreases(self):
        """DEFEND reduced exposure → positive crisis_mitigation."""
        engine = self._make_engine()
        _, bd = engine.compute(
            agent_id=0,
            is_newly_collapsed=False, is_collapsed=False,
            prev_stability=0.5, curr_stability=0.5,
            prev_exposure=0.5, curr_exposure=0.2,   # dropped by 0.3
            prev_avg_trust=0.0,
            coalition_size=0,
            trust_row={1: 0.0, 2: 0.0, 3: 0.0},
            mask_violated=False, pending_outgoing=0,
        )
        assert bd.crisis_mitigation > 0.0

    def test_crisis_mitigation_negative_when_exposure_rises(self):
        """Crisis worsening → negative crisis_mitigation."""
        engine = self._make_engine()
        _, bd = engine.compute(
            agent_id=0,
            is_newly_collapsed=False, is_collapsed=False,
            prev_stability=0.5, curr_stability=0.5,
            prev_exposure=0.1, curr_exposure=0.5,   # rose by 0.4
            prev_avg_trust=0.0,
            coalition_size=0,
            trust_row={1: 0.0, 2: 0.0, 3: 0.0},
            mask_violated=False, pending_outgoing=0,
        )
        assert bd.crisis_mitigation < 0.0

    def test_trust_alignment_zero_when_trust_stable(self):
        """No trust change → trust_alignment must be 0."""
        engine = self._make_engine()
        trust_val = 0.5
        _, bd = engine.compute(
            agent_id=0,
            is_newly_collapsed=False, is_collapsed=False,
            prev_stability=0.5, curr_stability=0.5,
            prev_exposure=0.0, curr_exposure=0.0,
            prev_avg_trust=trust_val,               # same as current avg_pos
            coalition_size=0,
            trust_row={1: trust_val, 2: trust_val, 3: trust_val},
            mask_violated=False, pending_outgoing=0,
        )
        assert bd.trust_alignment == pytest.approx(0.0, abs=1e-9)

    def test_trust_alignment_positive_when_trust_improves(self):
        """Trust increased from 0 → 0.5 → positive trust_alignment."""
        engine = self._make_engine()
        _, bd = engine.compute(
            agent_id=0,
            is_newly_collapsed=False, is_collapsed=False,
            prev_stability=0.5, curr_stability=0.5,
            prev_exposure=0.0, curr_exposure=0.0,
            prev_avg_trust=0.0,         # was 0
            coalition_size=0,
            trust_row={1: 0.5, 2: 0.5, 3: 0.5},   # now 0.5
            mask_violated=False, pending_outgoing=0,
        )
        assert bd.trust_alignment > 0.0

    def test_trust_alignment_zero_when_trust_decreases(self):
        """Trust decreased → trust_alignment is 0 (not double-penalised)."""
        engine = self._make_engine()
        _, bd = engine.compute(
            agent_id=0,
            is_newly_collapsed=False, is_collapsed=False,
            prev_stability=0.5, curr_stability=0.5,
            prev_exposure=0.0, curr_exposure=0.0,
            prev_avg_trust=0.8,         # was high
            coalition_size=0,
            trust_row={1: 0.1, 2: 0.1, 3: 0.1},   # dropped
            mask_violated=False, pending_outgoing=0,
        )
        assert bd.trust_alignment == pytest.approx(0.0, abs=1e-9)
