"""
district_accord/engine/turn_manager.py
========================================
TurnManager — coordinator for the Phase 5 engine layer.

Design
------
TurnManager does NOT own or replicate env state.  It wraps env.step()'s
existing private-method pipeline, adding two orthogonal responsibilities:

    1. Event emission   — calls self._event_bus.emit() at each pipeline
                          boundary, tagging events with the current turn.
    2. State recording  — calls self._state_tracker.record() at the end
                          of every turn for full time-series history.

env._execute_step_pipeline(actions) does the real work (unchanged logic);
TurnManager is purely an observation / coordination layer above it.

Pipeline order (locked, deterministic):
    1.  normalize + validate actions (mask enforcement)
    2.  emit action_validated / action_invalid
    3.  deduct proposal cost + create proposals
    4.  emit proposal_created
    5.  resolve proposals (accept / reject)
    6.  emit coalition_formed / coalition_joined / proposal_rejected
    7.  apply action effects (resources) + emit resource_transferred
    8.  tick TTL → emit proposal_expired
    9.  crisis step
    10. detect collapse → emit collapse
    11. compute rewards  (RewardEngine — unchanged)
    12. update trust → emit trust_updated
    13. record state (StateTracker)
    14. finalize info dict (events + snapshot injected here)

The run_turn() return value is the full (obs, rewards, done, trunc, info)
tuple, identical to what env.step() previously returned, with the
additions of info["events"] and info["state_snapshot"].
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from district_accord.engine.event_bus import EventBus
from district_accord.engine.state_tracker import StateTracker
from district_accord.utils.config import EnvConfig


class TurnManager:
    """
    Thin coordinator that wraps env._execute_step_pipeline() with
    event emission and state recording.

    Attributes:
        event_bus:     Shared EventBus instance (same object as env._event_bus).
        state_tracker: Shared StateTracker (same object as env._state_tracker).
        config:        EnvConfig reference for any config reads needed here.

    Usage (from env.step):
        return self._turn_manager.run_turn(actions, env=self)
    """

    def __init__(
        self,
        config:        EnvConfig,
        event_bus:     EventBus,
        state_tracker: StateTracker,
    ) -> None:
        self.config        = config
        self.event_bus     = event_bus
        self.state_tracker = state_tracker

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def run_turn(self, raw_actions: Dict, env: Any) -> Tuple:
        """
        Execute one environment turn with full observability.

        Args:
            raw_actions: Agent action dict passed by the caller of env.step().
            env:         The DistrictAccordEnv instance (passed by reference).

        Returns:
            (obs, rewards, done, truncated, info) — standard env.step() tuple,
            enriched with info["events"] and info["state_snapshot"].
        """
        # ── Tag the event bus with the current turn ───────────────────
        self.event_bus.set_turn(env._turn)
        events_before = len(self.event_bus)

        # ── Run the full step pipeline (all 19 sub-steps) ─────────────
        # Pipeline is in env._execute_step_pipeline; it emits events
        # via env._event_bus (same reference as self.event_bus) at each
        # key pipeline boundary.
        obs, rewards, done, truncated, info = env._execute_step_pipeline(
            raw_actions
        )

        # ── Record state AFTER pipeline completes ─────────────────────
        # env._turn has already been incremented by _execute_step_pipeline.
        n_events_this_turn = len(self.event_bus) - events_before
        snapshot = self.state_tracker.record(
            turn=env._turn,
            districts=env._districts,
            crisis=env._crisis,
            coalition=env._coalition,
            trust=env._trust,
            collapsed=env._collapsed,
            n_events=n_events_this_turn,
        )

        # ── Inject engine layer data into info ────────────────────────
        this_turn_events = self.event_bus._events[events_before:]
        info["events"]        = [e.to_dict() for e in this_turn_events]
        info["state_snapshot"] = snapshot.to_dict()

        return obs, rewards, done, truncated, info

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TurnManager("
            f"events={len(self.event_bus)}, "
            f"turns_recorded={len(self.state_tracker)})"
        )
