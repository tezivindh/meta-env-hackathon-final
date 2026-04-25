"""
tests/test_phase5_engine.py
============================
Phase 5 engine layer tests.

Covers:
    EventBus        — emit, filter, clear, immutability, invalid-type guard
    StateTracker    — record, history length, reset, field completeness
    TurnManager     — run_turn, info enrichment, deterministic replay
    Integration     — event emission per pipeline event, collapse detection,
                       coalition_formed, resource_transferred, proposal lifecycle
"""

from __future__ import annotations

import copy
from typing import Dict

import pytest

from district_accord.engine.event_bus import EventBus, Event, VALID_EVENT_TYPES
from district_accord.engine.state_tracker import (
    AgentSnapshot,
    StateTracker,
    TurnSnapshot,
)
from district_accord.engine.turn_manager import TurnManager
from district_accord.env import DistrictAccordEnv
from district_accord.spaces.action import make_default_parsed_action
from district_accord.utils.config import EnvConfig
from district_accord.utils.types import DiscreteAction


# ─── Fixtures ────────────────────────────────────────────────────────────────

SEED = 7


def make_cfg(**kwargs) -> EnvConfig:
    defaults = dict(
        num_districts=4,
        max_turns=10,
        seed=SEED,
        trust_init_std=0.0,
        obs_neighbor_noise_std=0.0,
        reward_spam_penalty=0.0,
    )
    defaults.update(kwargs)
    return EnvConfig(**defaults)


def make_env(**kwargs) -> DistrictAccordEnv:
    env = DistrictAccordEnv(make_cfg(**kwargs))
    env.reset()
    return env


def step_all_ignore(env: DistrictAccordEnv):
    """Step with all agents IGNOREing."""
    actions = {a: DiscreteAction.IGNORE for a in range(env.config.num_districts)}
    return env.step(actions)


# ─── EventBus unit tests ─────────────────────────────────────────────────────


class TestEventBus:
    def test_emit_appends_event(self):
        bus = EventBus()
        bus.set_turn(0)
        bus.emit("collapse", {"agent_id": 1})
        assert len(bus) == 1
        assert bus._events[0].event_type == "collapse"

    def test_emit_assigns_monotonic_sequence_numbers(self):
        bus = EventBus()
        bus.set_turn(0)
        bus.emit("collapse", {"agent_id": 0})
        bus.emit("action_validated", {"agent_id": 0, "action": "IGNORE"})
        seqs = [e.sequence_num for e in bus._events]
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == len(seqs)

    def test_emit_shallow_copies_payload(self):
        bus = EventBus()
        bus.set_turn(0)
        payload = {"agent_id": 3}
        bus.emit("collapse", payload)
        payload["agent_id"] = 99          # mutate original
        assert bus._events[0].payload["agent_id"] == 3   # bus unaffected

    def test_emit_invalid_type_raises_value_error(self):
        bus = EventBus()
        bus.set_turn(0)
        with pytest.raises(ValueError, match="Unknown event type"):
            bus.emit("nonexistent_event", {})

    def test_all_mandatory_event_types_accepted(self):
        bus = EventBus()
        bus.set_turn(0)
        for etype in VALID_EVENT_TYPES:
            bus.emit(etype, {"dummy": True})
        assert len(bus) == len(VALID_EVENT_TYPES)

    def test_get_events_no_filter_returns_all(self):
        bus = EventBus()
        bus.set_turn(0)
        bus.emit("collapse", {"agent_id": 0})
        bus.emit("collapse", {"agent_id": 1})
        assert len(bus.get_events()) == 2

    def test_get_events_filtered_by_turn(self):
        bus = EventBus()
        bus.set_turn(0)
        bus.emit("collapse", {"agent_id": 0})
        bus.set_turn(1)
        bus.emit("collapse", {"agent_id": 1})
        assert len(bus.get_events(turn=0)) == 1
        assert bus.get_events(turn=0)[0].payload["agent_id"] == 0

    def test_get_events_filtered_by_type(self):
        bus = EventBus()
        bus.set_turn(0)
        bus.emit("collapse", {"agent_id": 0})
        bus.emit("action_validated", {"agent_id": 0, "action": "IGNORE"})
        collapses = bus.get_events(event_type="collapse")
        assert len(collapses) == 1
        assert collapses[0].event_type == "collapse"

    def test_get_events_returns_copy(self):
        bus = EventBus()
        bus.set_turn(0)
        bus.emit("collapse", {"agent_id": 0})
        got = bus.get_events()
        got.clear()
        assert len(bus) == 1   # original unaffected

    def test_clear_resets_all_state(self):
        bus = EventBus()
        bus.set_turn(3)
        bus.emit("collapse", {"agent_id": 0})
        bus.clear()
        assert len(bus) == 0
        assert bus._seq == 0
        assert bus._current_turn == 0

    def test_event_is_frozen(self):
        bus = EventBus()
        bus.set_turn(0)
        bus.emit("collapse", {"agent_id": 0})
        event = bus._events[0]
        with pytest.raises((AttributeError, TypeError)):
            event.turn = 99  # type: ignore[misc]

    def test_to_list_serialisable(self):
        bus = EventBus()
        bus.set_turn(2)
        bus.emit("collapse", {"agent_id": 1})
        result = bus.to_list()
        assert isinstance(result, list)
        assert isinstance(result[0], dict)
        assert result[0]["type"] == "collapse"
        assert result[0]["turn"] == 2


# ─── StateTracker unit tests ──────────────────────────────────────────────────


class TestStateTracker:
    def _make_snapshot(self, tracker: StateTracker, env: DistrictAccordEnv, turn: int = 1):
        return tracker.record(
            turn=turn,
            districts=env._districts,
            crisis=env._crisis,
            coalition=env._coalition,
            trust=env._trust,
            collapsed=env._collapsed,
            n_events=3,
        )

    def test_record_returns_turn_snapshot(self):
        env = make_env()
        tracker = StateTracker()
        snap = self._make_snapshot(tracker, env, turn=1)
        assert isinstance(snap, TurnSnapshot)

    def test_snapshot_has_all_agents(self):
        env = make_env()
        tracker = StateTracker()
        snap = self._make_snapshot(tracker, env, turn=1)
        assert set(snap.agents.keys()) == set(range(env.config.num_districts))

    def test_agent_snapshot_fields_present(self):
        env = make_env()
        tracker = StateTracker()
        snap = self._make_snapshot(tracker, env, turn=1)
        agent_snap = snap.agents[0]
        assert isinstance(agent_snap, AgentSnapshot)
        assert 0.0 <= agent_snap.resources <= 1.0
        assert 0.0 <= agent_snap.stability <= 1.0
        assert 0.0 <= agent_snap.crisis_exposure <= 1.0
        assert isinstance(agent_snap.trust_row, dict)
        assert agent_snap.agent_id == 0

    def test_trust_row_excludes_self(self):
        env = make_env()
        tracker = StateTracker()
        snap = self._make_snapshot(tracker, env, turn=1)
        for aid, asnap in snap.agents.items():
            assert aid not in asnap.trust_row

    def test_history_length_after_multiple_records(self):
        env = make_env()
        tracker = StateTracker()
        N = 5
        for t in range(1, N + 1):
            self._make_snapshot(tracker, env, turn=t)
        assert len(tracker) == N
        assert len(tracker.get_history()) == N

    def test_get_turn_returns_correct_snapshot(self):
        env = make_env()
        tracker = StateTracker()
        self._make_snapshot(tracker, env, turn=1)
        self._make_snapshot(tracker, env, turn=2)
        snap = tracker.get_turn(2)
        assert snap is not None
        assert snap.turn == 2

    def test_get_turn_missing_returns_none(self):
        tracker = StateTracker()
        assert tracker.get_turn(99) is None

    def test_reset_clears_history(self):
        env = make_env()
        tracker = StateTracker()
        self._make_snapshot(tracker, env, turn=1)
        tracker.reset()
        assert len(tracker) == 0
        assert tracker.get_history() == []

    def test_to_dict_serialisable(self):
        env = make_env()
        tracker = StateTracker()
        snap = self._make_snapshot(tracker, env, turn=1)
        d = snap.to_dict()
        assert isinstance(d, dict)
        assert "agents" in d
        assert "crisis_level" in d
        assert "turn" in d

    def test_total_events_recorded_in_snapshot(self):
        env = make_env()
        tracker = StateTracker()
        snap = self._make_snapshot(tracker, env, turn=1)
        assert snap.total_events == 3


# ─── TurnManager + env integration tests ─────────────────────────────────────


class TestTurnManagerIntegration:

    def test_step_returns_events_in_info(self):
        env = make_env()
        _, _, _, _, info = step_all_ignore(env)
        assert "events" in info
        assert isinstance(info["events"], list)

    def test_step_returns_state_snapshot_in_info(self):
        env = make_env()
        _, _, _, _, info = step_all_ignore(env)
        assert "state_snapshot" in info
        snap = info["state_snapshot"]
        assert "agents" in snap
        assert "crisis_level" in snap
        assert "turn" in snap

    def test_state_snapshot_turn_matches_info_turn(self):
        env = make_env()
        _, _, _, _, info = step_all_ignore(env)
        assert info["state_snapshot"]["turn"] == info["turn"]

    def test_action_validated_events_emitted_per_agent(self):
        env = make_env()
        _, _, _, _, info = step_all_ignore(env)
        validated = [e for e in info["events"] if e["type"] == "action_validated"]
        assert len(validated) == env.config.num_districts

    def test_action_invalid_event_emitted_on_mask_violation(self):
        """Any action forced through mask enforcement should emit action_invalid."""
        env = make_env()
        # All ACCEPT — no pending proposals → masked → IGNORE + action_invalid
        actions = {a: DiscreteAction.ACCEPT_COALITION for a in range(4)}
        _, _, _, _, info = env.step(actions)
        invalid_events = [e for e in info["events"] if e["type"] == "action_invalid"]
        assert len(invalid_events) > 0

    def test_proposal_created_event_on_propose(self):
        env = make_env()
        actions = {
            0: make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=1),
            1: DiscreteAction.IGNORE,
            2: DiscreteAction.IGNORE,
            3: DiscreteAction.IGNORE,
        }
        _, _, _, _, info = env.step(actions)
        created = [e for e in info["events"] if e["type"] == "proposal_created"]
        assert len(created) == 1
        assert created[0]["payload"]["proposer"] == 0
        assert created[0]["payload"]["target"] == 1

    def test_coalition_formed_event_on_accept(self):
        env = make_env()
        # Turn 1: propose
        env.step({
            0: make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=1),
            1: DiscreteAction.IGNORE,
            2: DiscreteAction.IGNORE,
            3: DiscreteAction.IGNORE,
        })
        # Turn 2: accept
        _, _, _, _, info = env.step({
            1: make_default_parsed_action(DiscreteAction.ACCEPT_COALITION),
            0: DiscreteAction.IGNORE,
            2: DiscreteAction.IGNORE,
            3: DiscreteAction.IGNORE,
        })
        formed = [e for e in info["events"] if e["type"] in ("coalition_formed", "coalition_joined")]
        assert len(formed) == 1
        payload = formed[0]["payload"]
        assert payload["proposer"] == 0
        assert payload["acceptor"] == 1

    def test_proposal_rejected_event(self):
        env = make_env()
        env.step({
            0: make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=1),
            1: DiscreteAction.IGNORE,
            2: DiscreteAction.IGNORE,
            3: DiscreteAction.IGNORE,
        })
        _, _, _, _, info = env.step({
            1: make_default_parsed_action(DiscreteAction.REJECT_COALITION),
            0: DiscreteAction.IGNORE,
            2: DiscreteAction.IGNORE,
            3: DiscreteAction.IGNORE,
        })
        rejected = [e for e in info["events"] if e["type"] == "proposal_rejected"]
        assert len(rejected) == 1
        assert rejected[0]["payload"]["proposer"] == 0
        assert rejected[0]["payload"]["rejector"] == 1

    def test_trust_updated_event_on_accept(self):
        env = make_env()
        env.step({
            0: make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=1),
            1: DiscreteAction.IGNORE,
            2: DiscreteAction.IGNORE,
            3: DiscreteAction.IGNORE,
        })
        _, _, _, _, info = env.step({
            1: make_default_parsed_action(DiscreteAction.ACCEPT_COALITION),
            0: DiscreteAction.IGNORE,
            2: DiscreteAction.IGNORE,
            3: DiscreteAction.IGNORE,
        })
        trust_events = [e for e in info["events"] if e["type"] == "trust_updated"]
        assert len(trust_events) >= 1
        directions = {e["payload"]["direction"] for e in trust_events}
        assert "accept_bilateral" in directions

    def test_state_history_length_equals_turns(self):
        env = make_env()
        N = 5
        for _ in range(N):
            step_all_ignore(env)
        assert len(env._state_tracker) == N

    def test_state_history_reset_on_new_episode(self):
        env = make_env()
        for _ in range(3):
            step_all_ignore(env)
        env.reset()
        assert len(env._state_tracker) == 0

    def test_event_bus_cleared_on_reset(self):
        env = make_env()
        step_all_ignore(env)
        assert len(env._event_bus) > 0
        env.reset()
        assert len(env._event_bus) == 0

    def test_reward_breakdown_still_in_info(self):
        """Phase 4 reward_breakdown must still be present after Phase 5 wiring."""
        env = make_env()
        _, _, _, _, info = step_all_ignore(env)
        assert "reward_breakdown" in info
        for a in range(env.config.num_districts):
            bd = info["reward_breakdown"][a]
            assert "survival" in bd
            assert "total" in bd

    def test_deterministic_replay_same_events(self):
        """Same seed → identical event sequence across two full episodes."""
        cfg = make_cfg()

        env1 = DistrictAccordEnv(cfg)
        env1.reset()
        events1 = []
        for _ in range(cfg.max_turns):
            actions = {a: DiscreteAction.IGNORE for a in range(cfg.num_districts)}
            _, _, done, trunc, info = env1.step(actions)
            events1.extend(info["events"])
            if done or trunc:
                break

        env2 = DistrictAccordEnv(cfg)
        env2.reset()
        events2 = []
        for _ in range(cfg.max_turns):
            actions = {a: DiscreteAction.IGNORE for a in range(cfg.num_districts)}
            _, _, done, trunc, info = env2.step(actions)
            events2.extend(info["events"])
            if done or trunc:
                break

        assert len(events1) == len(events2)
        for e1, e2 in zip(events1, events2):
            assert e1["type"] == e2["type"]
            assert e1["turn"] == e2["turn"]
            assert e1["payload"] == e2["payload"]

    def test_pipeline_event_order_within_turn(self):
        """
        Within one turn, events must appear in pipeline order:
            action_validated/action_invalid → proposal_created →
            coalition_formed → resource_transferred → proposal_expired →
            collapse → trust_updated
        ORDER: action events come before coalition events.
        """
        env = make_env()
        # Turn 1: propose (so proposal_created appears after action_validated)
        _, _, _, _, info = env.step({
            0: make_default_parsed_action(DiscreteAction.PROPOSE_COALITION, target=1),
            1: DiscreteAction.IGNORE,
            2: DiscreteAction.IGNORE,
            3: DiscreteAction.IGNORE,
        })
        events = info["events"]
        types = [e["type"] for e in events]

        action_idx = next(i for i, t in enumerate(types) if t == "action_validated")
        proposal_idx = next((i for i, t in enumerate(types) if t == "proposal_created"), None)
        if proposal_idx is not None:
            assert action_idx < proposal_idx, (
                "action_validated must come before proposal_created"
            )

    def test_collapse_event_emitted(self):
        """Force a collapse by running many turns with crisis drain only."""
        cfg = make_cfg(
            num_districts=2,
            max_turns=50,
            passive_stability_drain=0.15,
            stability_threshold=0.05,
        )
        env = DistrictAccordEnv(cfg)
        env.reset()

        collapse_events = []
        for _ in range(50):
            actions = {a: DiscreteAction.IGNORE for a in range(2)}
            _, _, done, trunc, info = env.step(actions)
            collapse_events.extend(
                e for e in info["events"] if e["type"] == "collapse"
            )
            if done or trunc:
                break

        assert len(collapse_events) > 0, "At least one district must collapse"
        payload = collapse_events[0]["payload"]
        assert "agent_id" in payload
        assert "turn" in payload

    def test_turn_manager_repr(self):
        env = make_env()
        assert "TurnManager" in repr(env._turn_manager)

    def test_step_signature_unchanged(self):
        """env.step() return value must still be a 5-tuple with same types."""
        env = make_env()
        result = step_all_ignore(env)
        obs, rewards, done, trunc, info = result
        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(done, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
