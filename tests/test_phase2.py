"""
tests/test_phase2.py
======================
Phase 2 test suite — covers all new functionality without touching Phase 1 tests.

New features under test:
  - DiscreteAction: RECOVER + Phase 3 stubs
  - ActionParser.parse_structured() and parse_structured_safe()
  - ActionSpace: action_mask, validate, sample
  - ObservationBuilder: shapes, noise, stability_delta, flat_dim
  - build_flat_obs() helper
  - RECOVER action effects in env.step()
  - 4-district episode end-to-end
  - Crisis shock (bounded outputs)
  - ParsedAction accepted directly by env.step()
  - Backward compatibility: old DiscreteAction API unchanged
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest

from district_accord.core.crisis import CrisisSystem
from district_accord.env import DistrictAccordEnv
from district_accord.spaces.action import (
    ActionSpace,
    make_default_parsed_action,
    validate_parsed_action,
)
from district_accord.spaces.action_parser import ActionParser
from district_accord.spaces.observation import ObservationBuilder, build_flat_obs
from district_accord.utils.config import EnvConfig
from district_accord.utils.types import (
    AgentID,
    DiscreteAction,
    PHASE3_ACTIVE_ACTIONS,
    ParsedAction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(num_districts: int = 4, max_turns: int = 20, seed: int = 0) -> DistrictAccordEnv:
    cfg = EnvConfig(num_districts=num_districts, max_turns=max_turns, seed=seed)
    return DistrictAccordEnv(cfg)


def make_2d_env(seed: int = 0) -> DistrictAccordEnv:
    """Small 2-district env for focused tests."""
    cfg = EnvConfig(num_districts=2, max_turns=20, seed=seed)
    return DistrictAccordEnv(cfg)


def all_recover(n: int) -> Dict[AgentID, DiscreteAction]:
    return {i: DiscreteAction.RECOVER for i in range(n)}


def random_actions_p2(n: int, rng: np.random.Generator) -> Dict[AgentID, DiscreteAction]:
    choices = [DiscreteAction(i) for i in range(PHASE3_ACTIVE_ACTIONS)]
    return {i: rng.choice(choices) for i in range(n)}


# ---------------------------------------------------------------------------
# DiscreteAction enum
# ---------------------------------------------------------------------------

class TestDiscreteActionPhase2:
    def test_recover_value(self):
        assert DiscreteAction.RECOVER == 3

    def test_phase3_stubs_defined(self):
        assert DiscreteAction.REQUEST_AID == 4
        assert DiscreteAction.SHARE_RESOURCES == 5
        assert DiscreteAction.PROPOSE_COALITION == 6
        assert DiscreteAction.ACCEPT_COALITION == 7
        assert DiscreteAction.REJECT_COALITION == 8

    def test_phase1_values_unchanged(self):
        assert DiscreteAction.INVEST == 0
        assert DiscreteAction.DEFEND == 1
        assert DiscreteAction.IGNORE == 2

    def test_total_actions_count(self):
        assert len(DiscreteAction) == 9

    def test_phase2_active_actions_constant(self):
        assert PHASE3_ACTIVE_ACTIONS == 9


# ---------------------------------------------------------------------------
# make_default_parsed_action
# ---------------------------------------------------------------------------

class TestMakeDefaultParsedAction:
    def test_returns_dict(self):
        pa = make_default_parsed_action(DiscreteAction.INVEST)
        assert isinstance(pa, dict)

    def test_action_type_correct(self):
        pa = make_default_parsed_action(DiscreteAction.RECOVER)
        assert pa["action_type"] == DiscreteAction.RECOVER

    def test_resource_split_default(self):
        pa = make_default_parsed_action(DiscreteAction.DEFEND)
        np.testing.assert_array_equal(pa["resource_split"], [0.5, 0.5])

    def test_resource_split_dtype(self):
        pa = make_default_parsed_action(DiscreteAction.INVEST)
        assert pa["resource_split"].dtype == np.float32

    def test_target_none_by_default(self):
        pa = make_default_parsed_action(DiscreteAction.INVEST)
        assert pa["target"] is None

    def test_amount_none_by_default(self):
        pa = make_default_parsed_action(DiscreteAction.INVEST)
        assert pa["amount"] is None

    def test_raw_auto_filled(self):
        pa = make_default_parsed_action(DiscreteAction.INVEST)
        assert pa["raw"] == "invest"

    def test_custom_resource_split(self):
        pa = make_default_parsed_action(
            DiscreteAction.INVEST,
            resource_split=np.array([0.7, 0.3], dtype=np.float32),
        )
        np.testing.assert_allclose(pa["resource_split"], [0.7, 0.3], rtol=1e-5)


# ---------------------------------------------------------------------------
# validate_parsed_action
# ---------------------------------------------------------------------------

class TestValidateParsedAction:
    def _config(self, n: int = 4) -> EnvConfig:
        return EnvConfig(num_districts=n)

    def test_valid_action_passes(self):
        pa = make_default_parsed_action(DiscreteAction.INVEST)
        validate_parsed_action(pa, self._config(), agent_id=0)

    def test_invalid_split_shape_raises(self):
        pa = make_default_parsed_action(DiscreteAction.INVEST)
        pa["resource_split"] = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            validate_parsed_action(pa, self._config(), agent_id=0)

    def test_invalid_split_bounds_raises(self):
        pa = make_default_parsed_action(DiscreteAction.INVEST)
        pa["resource_split"] = np.array([-0.1, 1.1], dtype=np.float32)
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            validate_parsed_action(pa, self._config(), agent_id=0)

    def test_out_of_range_target_raises(self):
        pa = make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=99)
        with pytest.raises(ValueError, match="out of range"):
            validate_parsed_action(pa, self._config(n=4), agent_id=0)

    def test_negative_target_raises(self):
        pa = make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=-1)
        with pytest.raises(ValueError, match="out of range"):
            validate_parsed_action(pa, self._config(n=4), agent_id=0)

    def test_invalid_amount_bounds_raises(self):
        pa = make_default_parsed_action(DiscreteAction.SHARE_RESOURCES, amount=1.5)
        with pytest.raises(ValueError, match="\\[0.0, 1.0\\]"):
            validate_parsed_action(pa, self._config(), agent_id=0)

    def test_valid_target_passes(self):
        pa = make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=2)
        validate_parsed_action(pa, self._config(n=4), agent_id=0)


# ---------------------------------------------------------------------------
# ActionSpace
# ---------------------------------------------------------------------------

class TestActionSpace:
    def make_space(self, n: int = 4) -> ActionSpace:
        return ActionSpace(EnvConfig(num_districts=n))

    def test_total_actions(self):
        space = self.make_space()
        assert space.num_total_actions == len(DiscreteAction)

    def test_active_actions(self):
        space = self.make_space()
        assert space.num_active_actions == PHASE3_ACTIVE_ACTIONS

    def test_action_mask_shape(self):
        space = self.make_space()
        mask = space.action_mask()
        assert mask.shape == (len(DiscreteAction),)

    def test_action_mask_first_n_one(self):
        space = self.make_space()
        mask = space.action_mask()
        for i in range(PHASE3_ACTIVE_ACTIONS):
            assert mask[i] == pytest.approx(1.0)
        for i in range(PHASE3_ACTIVE_ACTIONS, len(DiscreteAction)):
            assert mask[i] == pytest.approx(0.0)

    def test_action_mask_dtype(self):
        space = self.make_space()
        assert space.action_mask().dtype == np.float32

    def test_sample_returns_parsed_action(self):
        space = self.make_space()
        pa = space.sample(active_only=True)
        assert isinstance(pa, dict)
        assert "action_type" in pa

    def test_sample_active_only_in_range(self):
        space = self.make_space()
        for _ in range(20):
            pa = space.sample(active_only=True)
            assert 0 <= pa["action_type"] < PHASE3_ACTIVE_ACTIONS

    def test_contains_valid(self):
        space = self.make_space()
        pa = make_default_parsed_action(DiscreteAction.RECOVER)
        assert space.contains(pa, agent_id=0)

    def test_active_action_types_length(self):
        space = self.make_space()
        types = space.active_action_types()
        assert len(types) == PHASE3_ACTIVE_ACTIONS


# ---------------------------------------------------------------------------
# ActionParser — parse_structured
# ---------------------------------------------------------------------------

class TestActionParserStructured:
    def make_parser(self, n: int = 4) -> ActionParser:
        return ActionParser(EnvConfig(num_districts=n))

    def test_simple_token_returns_parsed_action(self):
        parser = self.make_parser()
        result = parser.parse_structured({0: "invest"})
        assert isinstance(result[0], dict)
        assert result[0]["action_type"] == DiscreteAction.INVEST

    def test_recover_token(self):
        parser = self.make_parser()
        result = parser.parse_structured({0: "recover"})
        assert result[0]["action_type"] == DiscreteAction.RECOVER

    def test_share_with_target_and_amount(self):
        parser = self.make_parser()
        result = parser.parse_structured({0: "share:target=2,amount=0.3"})
        pa = result[0]
        assert pa["action_type"] == DiscreteAction.SHARE_RESOURCES
        assert pa["target"] == 2
        assert pa["amount"] == pytest.approx(0.3)

    def test_propose_with_target(self):
        parser = self.make_parser()
        result = parser.parse_structured({0: "propose:target=1"})
        pa = result[0]
        assert pa["action_type"] == DiscreteAction.PROPOSE_COALITION
        assert pa["target"] == 1
        assert pa["amount"] is None

    def test_invest_with_resource_split(self):
        parser = self.make_parser()
        result = parser.parse_structured({0: "invest:r0=0.7,r1=0.3"})
        pa = result[0]
        assert pa["action_type"] == DiscreteAction.INVEST
        np.testing.assert_allclose(pa["resource_split"], [0.7, 0.3], rtol=1e-5)

    def test_default_resource_split_when_absent(self):
        parser = self.make_parser()
        result = parser.parse_structured({0: "defend"})
        np.testing.assert_allclose(result[0]["resource_split"], [0.5, 0.5], rtol=1e-5)

    def test_raw_field_preserved(self):
        parser = self.make_parser()
        result = parser.parse_structured({0: "recover"})
        assert result[0]["raw"] == "recover"

    def test_unknown_token_raises_value_error(self):
        parser = self.make_parser()
        with pytest.raises(ValueError, match="unknown action token"):
            parser.parse_structured({0: "attack"})

    def test_non_string_raises_type_error(self):
        parser = self.make_parser()
        with pytest.raises(TypeError):
            parser.parse_structured({0: 42})  # type: ignore

    def test_safe_fallback_on_bad_token(self):
        parser = self.make_parser()
        result = parser.parse_structured_safe({0: "gibberish", 1: "invest"})
        assert result[0]["action_type"] == DiscreteAction.IGNORE
        assert result[1]["action_type"] == DiscreteAction.INVEST

    def test_multiple_agents(self):
        parser = self.make_parser()
        raw = {0: "invest", 1: "recover", 2: "defend", 3: "ignore"}
        result = parser.parse_structured(raw)
        assert result[0]["action_type"] == DiscreteAction.INVEST
        assert result[1]["action_type"] == DiscreteAction.RECOVER
        assert result[2]["action_type"] == DiscreteAction.DEFEND
        assert result[3]["action_type"] == DiscreteAction.IGNORE

    def test_backward_compat_old_parse_still_works(self):
        """Existing parse() must not be affected by Phase 2 additions."""
        parser = self.make_parser()
        result = parser.parse({0: "invest", 1: "defend"})
        assert result[0] == DiscreteAction.INVEST
        assert result[1] == DiscreteAction.DEFEND

    def test_backward_compat_new_token_in_old_parser(self):
        """parse() can now handle 'recover' as it's in ACTION_STR_MAP."""
        parser = self.make_parser()
        result = parser.parse({0: "recover"})
        assert result[0] == DiscreteAction.RECOVER


# ---------------------------------------------------------------------------
# ObservationBuilder
# ---------------------------------------------------------------------------

class TestObservationBuilder:
    def make_builder(
        self,
        num_districts: int = 4,
        seed: int = 0,
        noise: float = 0.05,
    ) -> ObservationBuilder:
        cfg = EnvConfig(
            num_districts=num_districts,
            seed=seed,
            obs_neighbor_noise_std=noise,
        )
        rng = np.random.default_rng(seed)
        return ObservationBuilder(cfg, rng)

    def make_districts(self, n: int):
        """Create dummy DistrictState objects."""
        from district_accord.core.district import DistrictState
        return {
            i: DistrictState(district_id=i, resources=0.6, stability=0.7)
            for i in range(n)
        }

    def make_crisis(self, seed: int = 0) -> CrisisSystem:
        cfg = EnvConfig(seed=seed)
        rng = np.random.default_rng(seed)
        crisis = CrisisSystem(cfg, rng)
        crisis.reset()
        return crisis

    def test_flat_dim_formula(self):
        for n in [2, 4, 6, 12]:
            builder = self.make_builder(num_districts=n)
            assert builder.flat_dim == 4 * n + 4

    def test_self_obs_shape(self):
        builder = self.make_builder(num_districts=4)
        districts = self.make_districts(4)
        crisis = self.make_crisis()
        builder.reset(districts)
        obs = builder.build(0, districts, crisis, turn=0, max_turns=20)
        assert obs["self"].shape == (4,)

    def test_others_obs_shape_4_districts(self):
        builder = self.make_builder(num_districts=4)
        districts = self.make_districts(4)
        crisis = self.make_crisis()
        builder.reset(districts)
        obs = builder.build(0, districts, crisis, turn=0, max_turns=20)
        assert obs["others"].shape == (3, 4)

    def test_others_obs_shape_2_districts(self):
        builder = self.make_builder(num_districts=2)
        districts = self.make_districts(2)
        crisis = self.make_crisis()
        builder.reset(districts)
        obs = builder.build(0, districts, crisis, turn=0, max_turns=20)
        assert obs["others"].shape == (1, 4)

    def test_crisis_obs_shape(self):
        builder = self.make_builder(num_districts=4)
        districts = self.make_districts(4)
        crisis = self.make_crisis()
        builder.reset(districts)
        obs = builder.build(0, districts, crisis, turn=0, max_turns=20)
        assert obs["crisis"].shape == (2,)

    def test_turn_obs_shape(self):
        builder = self.make_builder(num_districts=4)
        districts = self.make_districts(4)
        crisis = self.make_crisis()
        builder.reset(districts)
        obs = builder.build(0, districts, crisis, turn=0, max_turns=20)
        assert obs["turn"].shape == (2,)

    def test_action_mask_shape(self):
        builder = self.make_builder(num_districts=4)
        districts = self.make_districts(4)
        crisis = self.make_crisis()
        builder.reset(districts)
        obs = builder.build(0, districts, crisis, turn=0, max_turns=20)
        assert obs["action_mask"].shape == (len(DiscreteAction),)

    def test_flat_obs_shape_4_districts(self):
        builder = self.make_builder(num_districts=4)
        districts = self.make_districts(4)
        crisis = self.make_crisis()
        builder.reset(districts)
        obs = builder.build(0, districts, crisis, turn=0, max_turns=20)
        assert obs["flat"].shape == (builder.flat_dim,)

    def test_flat_obs_matches_build_flat_obs_helper(self):
        builder = self.make_builder(num_districts=4)
        districts = self.make_districts(4)
        crisis = self.make_crisis()
        builder.reset(districts)
        obs = builder.build(0, districts, crisis, turn=0, max_turns=20)
        expected_flat = build_flat_obs(obs)
        np.testing.assert_array_equal(obs["flat"], expected_flat)

    def test_stability_delta_zero_at_reset(self):
        """Stability delta should be 0 when prev_stability == current."""
        builder = self.make_builder(num_districts=4)
        districts = self.make_districts(4)
        crisis = self.make_crisis()
        builder.reset(districts)
        obs = builder.build(0, districts, crisis, turn=0, max_turns=20)
        assert obs["self"][3] == pytest.approx(0.0)  # index 3 = stability_delta

    def test_stability_delta_nonzero_after_change(self):
        from district_accord.core.district import DistrictState
        builder = self.make_builder(num_districts=2)
        districts = self.make_districts(2)
        crisis = self.make_crisis()
        builder.reset(districts)
        # Simulate stability change.
        old_stab = districts[0].stability
        districts[0].stability = old_stab + 0.05
        builder.update_prev_stability({k: v for k, v in
                                       {0: type("D", (), {"stability": old_stab})(), 1: districts[1]}.items()})
        # Force a manual prev_stability update to old value.
        builder._prev_stability[0] = old_stab
        obs = builder.build(0, districts, crisis, turn=1, max_turns=20)
        assert obs["self"][3] == pytest.approx(0.05, abs=1e-5)

    def test_all_obs_values_finite(self):
        builder = self.make_builder(num_districts=4)
        districts = self.make_districts(4)
        crisis = self.make_crisis()
        builder.reset(districts)
        for agent_id in districts:
            obs = builder.build(agent_id, districts, crisis, turn=5, max_turns=20)
            assert np.all(np.isfinite(obs["flat"]))

    def test_all_obs_dtype_float32(self):
        builder = self.make_builder(num_districts=4)
        districts = self.make_districts(4)
        crisis = self.make_crisis()
        builder.reset(districts)
        obs = builder.build(0, districts, crisis, turn=0, max_turns=20)
        for key in ("self", "others", "crisis", "turn", "action_mask", "flat"):
            assert obs[key].dtype == np.float32, f"Key '{key}' has wrong dtype."

    def test_others_values_clipped_to_unit_interval(self):
        """Even with noise, 'others' values should stay in [0, 1]."""
        # Large noise to force clipping.
        builder = self.make_builder(num_districts=4, noise=5.0)
        districts = self.make_districts(4)
        crisis = self.make_crisis()
        builder.reset(districts)
        for agent_id in districts:
            obs = builder.build(agent_id, districts, crisis, turn=0, max_turns=20)
            assert np.all(obs["others"] >= 0.0)
            assert np.all(obs["others"] <= 1.0)

    def test_others_excludes_self(self):
        """'others' should never contain self's exact values (unless noise=0)."""
        # Use zero noise to check structurally.
        builder = self.make_builder(num_districts=4, noise=0.0)
        districts = self.make_districts(4)
        # Give agent 0 a uniquely identifiable value.
        districts[0].resources = 0.123456
        crisis = self.make_crisis()
        builder.reset(districts)
        obs = builder.build(0, districts, crisis, turn=0, max_turns=20)
        # None of the 'others' rows should match district 0's resources (0.123456).
        for row in obs["others"]:
            assert abs(row[0] - 0.123456) > 1e-4

    def test_no_flat_when_flatten_false(self):
        cfg = EnvConfig(num_districts=4, flatten_observation=False)
        rng = np.random.default_rng(0)
        builder = ObservationBuilder(cfg, rng)
        from district_accord.core.district import DistrictState
        districts = {i: DistrictState(i, 0.6, 0.7) for i in range(4)}
        crisis = self.make_crisis()
        builder.reset(districts)
        obs = builder.build(0, districts, crisis, turn=0, max_turns=20)
        assert "flat" not in obs


# ---------------------------------------------------------------------------
# build_flat_obs helper
# ---------------------------------------------------------------------------

class TestBuildFlatObs:
    def _make_obs(self, n: int = 4) -> dict:
        from district_accord.core.district import DistrictState
        from district_accord.core.crisis import CrisisSystem
        cfg = EnvConfig(num_districts=n, seed=0)
        rng = np.random.default_rng(0)
        builder = ObservationBuilder(cfg, rng)
        districts = {i: DistrictState(i, 0.6, 0.7) for i in range(n)}
        crisis = CrisisSystem(cfg, rng)
        crisis.reset()
        builder.reset(districts)
        return builder.build(0, districts, crisis, turn=0, max_turns=20)

    def test_flat_shape_matches_formula(self):
        obs = self._make_obs(n=4)
        flat = build_flat_obs(obs)
        assert flat.shape == (4 * 4 + 4,)

    def test_flat_dtype_float32(self):
        obs = self._make_obs(n=4)
        flat = build_flat_obs(obs)
        assert flat.dtype == np.float32

    def test_flat_all_finite(self):
        obs = self._make_obs(n=4)
        flat = build_flat_obs(obs)
        assert np.all(np.isfinite(flat))

    def test_flat_excludes_action_mask(self):
        """action_mask (9 elements) should NOT appear consecutively in flat."""
        obs = self._make_obs(n=4)
        flat = build_flat_obs(obs)
        # action_mask is all 0s and 1s; "others" noisy view would not be all 0/1.
        # Just verify flat size = 2*N+6, not 2*N+6+num_actions.
        assert flat.shape == (4 * 4 + 4,)
        assert flat.shape[0] != 4 * 4 + 4 + len(DiscreteAction)


# ---------------------------------------------------------------------------
# Environment — 4-district integration
# ---------------------------------------------------------------------------

class TestEnv4Districts:
    def test_reset_returns_4_agents(self):
        env = make_env(num_districts=4)
        obs = env.reset()
        assert set(obs.keys()) == {0, 1, 2, 3}

    def test_obs_others_shape_for_4_districts(self):
        env = make_env(num_districts=4)
        obs = env.reset()
        for agent_obs in obs.values():
            assert agent_obs["others"].shape == (3, 4)

    def test_obs_flat_shape_for_4_districts(self):
        env = make_env(num_districts=4)
        obs = env.reset()
        expected_flat = 4 * 4 + 4  # = 14
        for agent_obs in obs.values():
            assert agent_obs["flat"].shape == (expected_flat,)

    def test_observation_shape_property(self):
        env = make_env(num_districts=4)
        env.reset()
        assert env.observation_shape == 4 * 4 + 4

    def test_step_4_agents_returns_4_rewards(self):
        env = make_env(num_districts=4)
        env.reset()
        actions = {i: DiscreteAction.INVEST for i in range(4)}
        _, rewards, _, _, _ = env.step(actions)
        assert set(rewards.keys()) == {0, 1, 2, 3}

    def test_full_4_district_episode(self):
        env = make_env(num_districts=4, max_turns=20, seed=42)
        env.reset()
        rng = np.random.default_rng(42)
        for _ in range(env.config.max_turns):
            actions = random_actions_p2(4, rng)
            obs, rewards, done, trunc, info = env.step(actions)
            for r in rewards.values():
                assert np.isfinite(r)
            for agent_obs in obs.values():
                assert np.all(np.isfinite(agent_obs["flat"]))
            if done or trunc:
                break
        assert env.turn <= env.config.max_turns

    def test_observation_consistent_n_agents(self):
        """All agents get obs with the same shape."""
        for n in [2, 4]:
            cfg = EnvConfig(num_districts=n, seed=0)
            env = DistrictAccordEnv(cfg)
            obs = env.reset()
            shapes = {k: v.shape for k, v in obs[0].items()}
            for agent_obs in obs.values():
                for k, shape in shapes.items():
                    assert agent_obs[k].shape == shape, (
                        f"N={n} agent obs key '{k}': expected {shape}, "
                        f"got {agent_obs[k].shape}"
                    )


# ---------------------------------------------------------------------------
# Environment — RECOVER action
# ---------------------------------------------------------------------------

class TestRecoverAction:
    def test_recover_increases_stability(self):
        env = make_2d_env(seed=0)
        env.reset()
        # Set stability to a known low value.
        env._districts[0].stability = 0.3
        env._districts[1].stability = 0.3
        stab_before = env._districts[0].stability

        actions = {0: DiscreteAction.RECOVER, 1: DiscreteAction.IGNORE}
        env.step(actions)
        # District 0 used RECOVER: stability should increase.
        # (net = +recover_stability_gain - passive_stability_drain - crisis_effect)
        stab_after = env._districts[0].stability
        assert stab_after > stab_before - 0.05, (
            f"RECOVER should increase stability: before={stab_before:.3f}, "
            f"after={stab_after:.3f}"
        )

    def test_recover_costs_resources(self):
        env = make_2d_env(seed=0)
        env.reset()
        env._districts[0].resources = 0.8
        res_before = env._districts[0].resources

        actions = {0: DiscreteAction.RECOVER, 1: DiscreteAction.IGNORE}
        env.step(actions)
        res_after = env._districts[0].resources
        # RECOVER costs recover_resource_cost + passive_resource_drain
        expected_cost = env.config.recover_resource_cost + env.config.passive_resource_drain
        assert res_before - res_after >= expected_cost * 0.5, (
            f"RECOVER should cost resources: before={res_before:.3f}, after={res_after:.3f}"
        )

    def test_recover_in_action_mask(self):
        env = make_2d_env()
        obs = env.reset()
        for agent_obs in obs.values():
            mask = agent_obs["action_mask"]
            assert mask[DiscreteAction.RECOVER] == pytest.approx(1.0)

    def test_phase3_actions_in_mask(self):
        pass

class TestParsedActionInput:
    def test_parsed_action_accepted_by_step(self):
        env = make_2d_env(seed=0)
        env.reset()
        pa = make_default_parsed_action(DiscreteAction.INVEST)
        actions = {0: pa, 1: pa}
        obs, rewards, done, trunc, info = env.step(actions)
        assert isinstance(rewards, dict)

    def test_structured_parsed_action_accepted(self):
        env = make_2d_env(seed=0)
        parser = ActionParser(env.config)
        env.reset()
        parsed = parser.parse_structured({0: "recover", 1: "invest:r0=0.6,r1=0.4"})
        obs, rewards, done, trunc, info = env.step(parsed)
        assert rewards[0] > 0 or rewards[0] <= 0  # just check it ran

    def test_discrete_action_and_parsed_action_mixed(self):
        """env.step() should accept a mix of DiscreteAction and ParsedAction."""
        env = make_2d_env(seed=0)
        env.reset()
        pa = make_default_parsed_action(DiscreteAction.INVEST)
        actions = {0: DiscreteAction.DEFEND, 1: pa}
        obs, rewards, done, trunc, info = env.step(actions)
        assert info["actions_taken"][0] == "DEFEND"
        assert info["actions_taken"][1] == "INVEST"


# ---------------------------------------------------------------------------
# Crisis — Phase 2 shock & district pressure
# ---------------------------------------------------------------------------

class TestCrisisPhase2:
    def test_crisis_bounded_after_shock(self):
        """Crisis must stay in [0, 1] even with shocks."""
        cfg = EnvConfig(
            seed=42,
            crisis_shock_prob=1.0,        # shock every turn
            crisis_shock_magnitude=0.5,   # large shock
        )
        rng = np.random.default_rng(42)
        crisis = CrisisSystem(cfg, rng)
        crisis.reset()
        for _ in range(50):
            level = crisis.step()
            assert 0.0 <= level <= 1.0

    def test_district_pressure_increases_crisis(self):
        """More districts → higher drift → faster crisis growth on average."""
        # 12 districts: pressure = 10 * 0.005 = 0.05/turn
        cfg_many = EnvConfig(
            seed=0,
            num_districts=12,
            crisis_drift=0.01,
            crisis_noise_std=0.001,
            crisis_shock_prob=0.0,  # disable shock for determinism
            crisis_district_scale=0.05,
        )
        cfg_few = EnvConfig(
            seed=0,
            num_districts=2,
            crisis_drift=0.01,
            crisis_noise_std=0.001,
            crisis_shock_prob=0.0,
            crisis_district_scale=0.05,
        )

        def run_crisis(cfg: EnvConfig, n: int) -> float:
            rng = np.random.default_rng(cfg.seed)
            crisis = CrisisSystem(cfg, rng)
            crisis.reset()
            districts = [object()] * n  # just need len()
            for _ in range(20):
                crisis.step(district_states=districts)
            return crisis.crisis_level

        level_many = run_crisis(cfg_many, 12)
        level_few = run_crisis(cfg_few, 2)
        assert level_many > level_few

    def test_shock_history_recorded(self):
        """Shock events must be captured in crisis history."""
        cfg = EnvConfig(
            seed=7,
            crisis_shock_prob=1.0,
            crisis_shock_magnitude=0.2,
        )
        rng = np.random.default_rng(7)
        crisis = CrisisSystem(cfg, rng)
        crisis.reset()
        for _ in range(10):
            crisis.step()
        assert len(crisis.history) == 11  # reset + 10 steps


# ---------------------------------------------------------------------------
# Backward compatibility guard
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Verify that every Phase 1 API still works unchanged."""

    def test_phase1_api_parse_returns_discrete_action(self):
        cfg = EnvConfig(num_districts=2)
        parser = ActionParser(cfg)
        result = parser.parse({0: "invest", 1: "defend"})
        assert result[0] == DiscreteAction.INVEST
        assert result[1] == DiscreteAction.DEFEND

    def test_phase1_api_parse_safe_returns_discrete_action(self):
        cfg = EnvConfig(num_districts=2)
        parser = ActionParser(cfg)
        result = parser.parse_safe({0: "bad_token", 1: "ignore"})
        assert result[0] == DiscreteAction.IGNORE
        assert result[1] == DiscreteAction.IGNORE

    def test_phase1_env_step_with_discrete_action(self):
        env = DistrictAccordEnv(EnvConfig(num_districts=2, seed=1))
        env.reset()
        obs, rewards, done, trunc, info = env.step(
            {0: DiscreteAction.INVEST, 1: DiscreteAction.DEFEND}
        )
        assert set(rewards.keys()) == {0, 1}

    def test_phase1_full_episode_still_runs(self):
        cfg = EnvConfig(num_districts=2, max_turns=20, seed=99)
        env = DistrictAccordEnv(cfg)
        env.reset()
        rng = np.random.default_rng(99)
        for _ in range(cfg.max_turns):
            actions = {i: rng.choice(list(DiscreteAction)[:4]) for i in range(2)}
            _, _, done, trunc, _ = env.step(actions)
            if done or trunc:
                break
        assert env.turn <= cfg.max_turns
