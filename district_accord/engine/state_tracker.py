"""
district_accord/engine/state_tracker.py
=========================================
Per-turn state snapshot history for the DistrictAccord environment.

Records a complete, serialisable snapshot of every agent and global
state at the END of each step.  Enables:
    - Time-series replay for debugging
    - Training-curve analysis
    - Deterministic reproducibility verification

Cleared on env.reset().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Snapshot primitives
# ---------------------------------------------------------------------------

@dataclass
class AgentSnapshot:
    """
    Complete per-agent state at the end of one turn.

    All float fields are rounded to 4 decimal places for stable equality
    comparison across deterministic replays.
    """

    agent_id:       int
    resources:      float
    stability:      float
    crisis_exposure: float
    coalition_id:   Optional[int]           # None if not in a coalition
    trust_row:      Dict[int, float]        # {other_id: trust_value}
    collapsed:      bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id":        self.agent_id,
            "resources":       round(self.resources, 4),
            "stability":       round(self.stability, 4),
            "crisis_exposure": round(self.crisis_exposure, 4),
            "coalition_id":    self.coalition_id,
            "trust_row":       {k: round(float(v), 4) for k, v in self.trust_row.items()},
            "collapsed":       self.collapsed,
        }


@dataclass
class TurnSnapshot:
    """
    Complete environment state at the end of one turn.

    Attributes:
        turn:              Turn number AFTER increment (matches info["turn"]).
        agents:            Per-agent snapshots, keyed by agent_id.
        crisis_level:      Global crisis system level.
        active_coalitions: Number of distinct coalitions with ≥ 2 members.
        total_events:      Number of EventBus events emitted this turn.
    """

    turn:              int
    agents:            Dict[int, AgentSnapshot]
    crisis_level:      float
    active_coalitions: int
    total_events:      int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn":              self.turn,
            "crisis_level":      round(self.crisis_level, 4),
            "active_coalitions": self.active_coalitions,
            "total_events":      self.total_events,
            "agents": {
                str(a): snap.to_dict()
                for a, snap in sorted(self.agents.items())
            },
        }


# ---------------------------------------------------------------------------
# StateTracker
# ---------------------------------------------------------------------------

class StateTracker:
    """
    Time-series recorder of TurnSnapshots.

    Usage:
        tracker = StateTracker()
        snapshot = tracker.record(turn, districts, crisis, coalition, trust, n_events)
        history  = tracker.get_history()
        tracker.reset()            # called by env.reset()
    """

    def __init__(self) -> None:
        self._history: List[TurnSnapshot] = []

    # ------------------------------------------------------------------
    # Record
    # ------------------------------------------------------------------

    def record(
        self,
        turn:      int,
        districts: Any,          # Dict[AgentID, DistrictState]
        crisis:    Any,          # CrisisSystem
        coalition: Any,          # CoalitionSystem
        trust:     Any,          # TrustSystem
        collapsed: Dict[int, bool],
        n_events:  int = 0,
    ) -> TurnSnapshot:
        """
        Build and store a TurnSnapshot for `turn`.

        This is called at the END of each step, after all state mutations
        have completed.

        Args:
            turn:      Turn counter value (post-increment).
            districts: Dict[AgentID, DistrictState] — live env._districts.
            crisis:    CrisisSystem — live env._crisis.
            coalition: CoalitionSystem — live env._coalition.
            trust:     TrustSystem — live env._trust.
            collapsed: Dict[AgentID, bool] — live env._collapsed.
            n_events:  Number of EventBus events emitted this turn.

        Returns:
            The stored TurnSnapshot.
        """
        trust_mat = trust.as_matrix()

        agents: Dict[int, AgentSnapshot] = {}
        for agent_id, d in sorted(districts.items()):
            row = trust_mat.get(agent_id, {})
            trust_row = {
                k: round(float(v), 4)
                for k, v in row.items()
                if k != agent_id
            }
            agents[agent_id] = AgentSnapshot(
                agent_id=agent_id,
                resources=round(float(d.resources), 4),
                stability=round(float(d.stability), 4),
                crisis_exposure=round(float(d.crisis_exposure), 4),
                coalition_id=coalition.get_coalition(agent_id),
                trust_row=trust_row,
                collapsed=collapsed.get(agent_id, False),
            )

        # Crisis level — support both .level attribute and .to_dict()
        try:
            crisis_level = float(crisis.level)
        except AttributeError:
            crisis_level = float(crisis.to_dict().get("level", 0.0))

        # Count coalitions with ≥ 2 members
        try:
            active_coalitions = len([
                cid for cid in set(agents[a].coalition_id for a in agents)
                if cid is not None
                and sum(1 for b in agents if agents[b].coalition_id == cid) >= 2
            ])
        except Exception:
            active_coalitions = 0

        snapshot = TurnSnapshot(
            turn=turn,
            agents=agents,
            crisis_level=round(crisis_level, 4),
            active_coalitions=active_coalitions,
            total_events=n_events,
        )
        self._history.append(snapshot)
        return snapshot

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_history(self) -> List[TurnSnapshot]:
        """Full time-series of TurnSnapshots (copy)."""
        return list(self._history)

    def get_turn(self, turn: int) -> Optional[TurnSnapshot]:
        """Return snapshot for a specific turn, or None if not recorded."""
        return next((s for s in self._history if s.turn == turn), None)

    def to_list(self) -> List[Dict[str, Any]]:
        """Serialisable list of all snapshots."""
        return [s.to_dict() for s in self._history]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all history for a new episode."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return f"StateTracker(turns_recorded={len(self._history)})"
