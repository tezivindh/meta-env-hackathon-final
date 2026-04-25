"""
district_accord/engine/event_bus.py
=====================================
Centralised, append-only event log for one DistrictAccord episode.

All state-changing events are emitted here during env.step(), in
deterministic insertion order (sorted agent-id processing).
The bus is cleared on env.reset() and does NOT persist across episodes.

Event types (mandatory set):
    action_validated    — action passed mask check
    action_invalid      — action was masked; converted to IGNORE
    proposal_created    — PROPOSE_COALITION or REQUEST_AID succeeded
    proposal_rejected   — REJECT_COALITION resolved a pending proposal
    proposal_expired    — proposal TTL hit zero with no response
    coalition_formed    — new coalition created (proposer + first acceptor)
    coalition_joined    — subsequent agent joined an existing coalition
    resource_transferred — SHARE_RESOURCES executed
    trust_updated       — accept/reject event changed trust values
    collapse            — district stability ≤ threshold; district collapses
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Closed set of valid event types — emit() rejects unknown types.
VALID_EVENT_TYPES: frozenset = frozenset({
    "action_validated",
    "action_invalid",
    "proposal_created",
    "proposal_rejected",
    "proposal_expired",
    "coalition_formed",
    "coalition_joined",
    "resource_transferred",
    "trust_updated",
    "collapse",
})


@dataclass(frozen=True)
class Event:
    """
    Immutable record of one state-changing event.

    Attributes:
        turn:         Environment turn during which the event occurred
                      (value of env._turn BEFORE the turn-counter increment).
        sequence_num: Monotonically increasing insertion index within
                      the current episode.  Used to reconstruct exact ordering.
        event_type:   One of VALID_EVENT_TYPES.
        payload:      Read-only dict with event-specific fields.
    """

    turn:         int
    sequence_num: int
    event_type:   str
    payload:      Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable representation (compatible with JSON / info logging)."""
        return {
            "turn":     self.turn,
            "seq":      self.sequence_num,
            "type":     self.event_type,
            "payload":  dict(self.payload),
        }


class EventBus:
    """
    Append-only event log for one episode.

    Usage:
        bus = EventBus()
        bus.set_turn(env._turn)          # called at start of each step
        bus.emit("action_validated", {"agent_id": 0, "action": "DEFEND"})
        events_this_turn = bus.get_events(turn=0)
        bus.clear()                      # called by env.reset()
    """

    def __init__(self) -> None:
        self._events:      List[Event] = []
        self._seq:         int         = 0
        self._current_turn: int        = 0

    # ------------------------------------------------------------------
    # Turn boundary
    # ------------------------------------------------------------------

    def set_turn(self, turn: int) -> None:
        """Tag subsequent events with this turn number."""
        self._current_turn = turn

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        """
        Append one event.

        Args:
            event_type: Must be in VALID_EVENT_TYPES.
            payload:    Shallow-copied on insertion for immutability.

        Raises:
            ValueError: Unknown event_type.
        """
        if event_type not in VALID_EVENT_TYPES:
            raise ValueError(
                f"Unknown event type {event_type!r}. "
                f"Valid: {sorted(VALID_EVENT_TYPES)}"
            )
        self._events.append(
            Event(
                turn=self._current_turn,
                sequence_num=self._seq,
                event_type=event_type,
                payload=dict(payload),   # shallow copy
            )
        )
        self._seq += 1

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_events(
        self,
        turn:       Optional[int] = None,
        event_type: Optional[str] = None,
    ) -> List[Event]:
        """
        Return events, optionally filtered.

        Args:
            turn:       If given, only events from that turn.
            event_type: If given, only events of that type.

        Returns:
            New list (safe to mutate); events in insertion order.
        """
        result: List[Event] = self._events
        if turn is not None:
            result = [e for e in result if e.turn == turn]
        if event_type is not None:
            result = [e for e in result if e.event_type == event_type]
        return list(result)

    def get_current_turn_events(self) -> List[Event]:
        """Shorthand for get_events(turn=current_turn)."""
        return self.get_events(turn=self._current_turn)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all events and reset counters for a new episode."""
        self._events.clear()
        self._seq          = 0
        self._current_turn = 0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_list(self) -> List[Dict[str, Any]]:
        """Return all events as serialisable dicts."""
        return [e.to_dict() for e in self._events]

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._events)

    def __repr__(self) -> str:
        return (
            f"EventBus(total={len(self._events)}, "
            f"turn={self._current_turn})"
        )
