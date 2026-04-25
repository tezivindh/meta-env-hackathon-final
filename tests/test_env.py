"""
tests/test_env.py
===================
Integration and unit tests for DistrictAccordEnv.

Coverage:
  - reset() returns correctly structured observation
  - Observation shape and dtype for all keys
  - flatten_observation produces "flat" key
  - step() return types
  - Rewards contain all agent IDs
  - Full episode completion (random policy)
  - Truncation at max_turns
  - AssertionError on post-termination step
  - ValueError on missing action
  - TypeError on non-DiscreteAction action
  - Survival reward positive when alive
  - Collapse penalty on collapse
  - ActionParser integration
  - Determinism: same seed → same trajectory
  - Info dict structure
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest

from district_accord.env import DistrictAccordEnv
from district_accord.spaces.action_parser import ActionParser
from district_accord.utils.config import EnvConfig
from district_accord.utils.types import AgentID, DiscreteAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(
    num_districts: int = 2,
    max_turns: int = 20,
    seed: int = 42,
    flatten: bool = True,
) -> DistrictAccordEnv:
    cfg = EnvConfig(
        num_districts=num_districts,
        max_turns=max_turns,
        seed=seed,
        flatten_observation=flatten,
    )
    return DistrictAccordEnv(cfg)


def all_invest(num: int) -> Dict[AgentID, DiscreteAction]:
    return {i: DiscreteAction.INVEST for i in range(num)}


def all_ignore(num: int) -> Dict[AgentID, DiscreteAction]:
    return {i: DiscreteAction.IGNORE for i in range(num)}


def random_actions(
    num: int, rng: np.random.Generator
) -> Dict[AgentID, DiscreteAction]:
    choices = list(DiscreteAction)
    return {i: rng.choice(choices) for i in range(num)}


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestEnvReset:
    def test_returns_dict_with_all_agent_ids(self):
        env = make_env()
        obs = env.reset()
        assert isinstance(obs, dict)
        assert set(obs.keys()) == {0, 1}

    def test_obs_has_required_keys(self):
        env = make_env()
        obs = env.reset()
        for agent_obs in obs.values():
            assert "self" in agent_obs
            assert "crisis" in agent_obs       # renamed from "global_crisis" in Phase 2
            assert "turn" in agent_obs
            assert "others" in agent_obs       # new in Phase 2
            assert "action_mask" in agent_obs  # new in Phase 2

    def test_self_obs_shape(self):
        env = make_env()
        obs = env.reset()
        for agent_obs in obs.values():
            assert agent_obs["self"].shape == (4,)   # Phase 2: +stability_delta
            assert agent_obs["self"].dtype == np.float32

    def test_global_crisis_obs_shape(self):
        """Phase 2: key renamed to 'crisis', shape extended to (2,)."""
        env = make_env()
        obs = env.reset()
        for agent_obs in obs.values():
            assert agent_obs["crisis"].shape == (2,)

    def test_turn_obs_shape(self):
        env = make_env()
        obs = env.reset()
        for agent_obs in obs.values():
            assert agent_obs["turn"].shape == (2,)

    def test_flat_obs_present_when_flatten_true(self):
        env = make_env(flatten=True)
        obs = env.reset()
        for agent_obs in obs.values():
            assert "flat" in agent_obs
            assert agent_obs["flat"].shape == (env.observation_shape,)

    def test_flat_obs_absent_when_flatten_false(self):
        env = make_env(flatten=False)
        obs = env.reset()
        for agent_obs in obs.values():
            assert "flat" not in agent_obs

    def test_turn_progress_zero_after_reset(self):
        """Both turn components encode turn=0."""
        env = make_env()
        obs = env.reset()
        for agent_obs in obs.values():
            assert agent_obs["turn"][0] == pytest.approx(0.0)    # progress
            assert agent_obs["turn"][1] == pytest.approx(1.0)    # remaining

    def test_reset_resets_turn_counter(self):
        env = make_env()
        env.reset()
        env.step(all_invest(2))
        env.reset()
        assert env.turn == 0

    def test_reset_allows_new_episode_after_done(self):
        env = make_env(max_turns=5)
        env.reset()
        for _ in range(5):
            _, _, done, trunc, _ = env.step(all_invest(2))
            if done or trunc:
                break
        # Must not raise:
        obs = env.reset()
        assert isinstance(obs, dict)


# ---------------------------------------------------------------------------
# step() return types and shapes
# ---------------------------------------------------------------------------

class TestEnvStep:
    def test_step_returns_5_tuple(self):
        env = make_env()
        env.reset()
        result = env.step(all_invest(2))
        assert len(result) == 5

    def test_step_obs_has_all_agents(self):
        env = make_env()
        env.reset()
        obs, *_ = env.step(all_invest(2))
        assert set(obs.keys()) == {0, 1}

    def test_step_rewards_dict_has_all_agents(self):
        env = make_env()
        env.reset()
        _, rewards, *_ = env.step(all_invest(2))
        assert set(rewards.keys()) == {0, 1}

    def test_done_is_bool(self):
        env = make_env()
        env.reset()
        _, _, done, truncated, _ = env.step(all_invest(2))
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)

    def test_info_is_dict(self):
        env = make_env()
        env.reset()
        _, _, _, _, info = env.step(all_invest(2))
        assert isinstance(info, dict)

    def test_info_has_expected_keys(self):
        env = make_env()
        env.reset()
        _, _, _, _, info = env.step(all_invest(2))
        assert "turn" in info
        assert "crisis" in info
        assert "districts" in info
        assert "collapsed" in info
        assert "actions_taken" in info

    def test_flat_obs_shape_after_step(self):
        env = make_env(flatten=True)
        env.reset()
        obs, *_ = env.step(all_invest(2))
        for agent_obs in obs.values():
            assert agent_obs["flat"].shape == (env.observation_shape,)


# ---------------------------------------------------------------------------
# Reward logic
# ---------------------------------------------------------------------------

class TestRewards:
    def test_survival_reward_positive_when_alive(self):
        env = make_env(seed=0)
        env.reset()
        _, rewards, _, _, _ = env.step(all_invest(2))
        for r in rewards.values():
            assert r > 0, f"Expected positive survival reward, got {r}"

    def test_zero_reward_for_previously_collapsed(self):
        """
        Force a district to collapse by draining its stability,
        then check reward is 0 in subsequent turns.
        """
        env = make_env(seed=99)
        env.reset()
        # Manually collapse district 1.
        env._districts[1].stability = 0.0
        env._collapsed[1] = True
        # Step: district 1 was already collapsed before this turn.
        _, rewards, _, _, _ = env.step({0: DiscreteAction.INVEST})
        assert rewards[1] == pytest.approx(0.0)

    def test_collapse_penalty_on_newly_collapsed(self):
        """
        Force district 0 to collapse this turn by zeroing its stability
        before the step.  _update_collapse_status detects it and applies
        the collapse_penalty (negative), not the survival_reward (positive).
        """
        env = make_env(seed=99)
        env.reset()
        # Pin stability to 0.0 — guaranteed collapse detection this turn.
        env._districts[0].stability = 0.0
        _, rewards, _, _, _ = env.step(all_invest(2))
        assert rewards[0] == pytest.approx(env.config.collapse_penalty)


# ---------------------------------------------------------------------------
# Terminal conditions
# ---------------------------------------------------------------------------

class TestTerminal:
    def test_truncation_after_max_turns(self):
        env = make_env(max_turns=5, seed=0)
        env.reset()
        done = truncated = False
        for _ in range(5):
            _, _, done, truncated, _ = env.step(all_invest(2))
            if done or truncated:
                break
        assert done or truncated

    def test_turn_counter_matches_steps(self):
        env = make_env(max_turns=10, seed=0)
        env.reset()
        for t in range(10):
            _, _, done, trunc, _ = env.step(all_invest(2))
            if done or trunc:
                break
        assert env.turn <= 10

    def test_step_after_done_raises(self):
        env = make_env(max_turns=3, seed=0)
        env.reset()
        for _ in range(3):
            _, _, done, trunc, _ = env.step(all_invest(2))
            if done or trunc:
                break
        with pytest.raises(AssertionError, match="reset()"):
            env.step(all_invest(2))

    def test_done_when_all_collapsed(self):
        env = make_env(seed=0)
        env.reset()
        # Force both districts to the collapse threshold.
        for district in env._districts.values():
            district.stability = env.config.stability_threshold - 0.001
        _, _, done, _, _ = env.step(all_ignore(2))
        assert done is True


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_missing_action_raises_value_error(self):
        env = make_env()
        env.reset()
        with pytest.raises(ValueError, match="Missing action"):
            env.step({0: DiscreteAction.INVEST})   # district 1 missing

    def test_wrong_type_raises_type_error(self):
        env = make_env()
        env.reset()
        with pytest.raises(TypeError, match="DiscreteAction"):
            env.step({0: "invest", 1: DiscreteAction.DEFEND})   # type: ignore

    def test_collapsed_district_not_required_in_actions(self):
        """A collapsed district does not need to be in the actions dict."""
        env = make_env(seed=0)
        env.reset()
        env._districts[1].stability = 0.0
        env._collapsed[1] = True
        # Only providing action for district 0 should not raise.
        obs, rewards, done, trunc, info = env.step({0: DiscreteAction.INVEST})
        assert isinstance(rewards, dict)


# ---------------------------------------------------------------------------
# ActionParser integration
# ---------------------------------------------------------------------------

class TestActionParserIntegration:
    def test_parse_and_step(self):
        env = make_env(seed=1)
        parser = ActionParser(env.config)
        env.reset()
        parsed = parser.parse({0: "invest", 1: "defend"})
        obs, rewards, done, trunc, info = env.step(parsed)
        assert rewards[0] > 0
        assert rewards[1] > 0

    def test_parse_safe_used_during_training(self):
        env = make_env(seed=2)
        parser = ActionParser(env.config)
        env.reset()
        # Malformed LLM output — parse_safe should fall back to IGNORE.
        parsed = parser.parse_safe({0: "gibberish", 1: "invest"})
        assert parsed[0] == DiscreteAction.IGNORE
        assert parsed[1] == DiscreteAction.INVEST


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_obs(self):
        """Two envs with the same seed must return identical initial obs."""
        env1 = make_env(seed=77)
        env2 = make_env(seed=77)
        obs1 = env1.reset()
        obs2 = env2.reset()
        for agent_id in obs1:
            np.testing.assert_array_equal(
                obs1[agent_id]["flat"], obs2[agent_id]["flat"]
            )

    def test_same_seed_same_trajectory(self):
        """Same seed + same actions → identical reward trajectory."""
        rng = np.random.default_rng(0)
        env1 = make_env(seed=55)
        env2 = make_env(seed=55)
        env1.reset()
        env2.reset()

        for _ in range(5):
            actions = random_actions(2, np.random.default_rng(rng.integers(0, 9999)))
            _, r1, _, _, _ = env1.step(actions)
            _, r2, _, _, _ = env2.step(actions)
            for aid in r1:
                assert r1[aid] == pytest.approx(r2[aid])


# ---------------------------------------------------------------------------
# Full episode smoke test
# ---------------------------------------------------------------------------

class TestFullEpisode:
    def test_random_policy_completes(self):
        env = make_env(num_districts=2, max_turns=20, seed=42)
        env.reset()
        rng = np.random.default_rng(42)
        for _ in range(env.config.max_turns):
            actions = random_actions(env.num_agents, rng)
            obs, rewards, done, trunc, info = env.step(actions)
            # All rewards must be finite.
            for r in rewards.values():
                assert np.isfinite(r), f"Non-finite reward: {r}"
            # All flat obs must be finite.
            for agent_obs in obs.values():
                assert np.all(np.isfinite(agent_obs["flat"]))
            if done or trunc:
                break
        assert env.turn <= env.config.max_turns
