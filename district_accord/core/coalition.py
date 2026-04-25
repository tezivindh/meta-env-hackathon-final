"""
district_accord/core/coalition.py
====================================
CoalitionSystem — Phase 3 membership state manager.

Responsibilities:
    - Coalition ID assignment
    - Agent membership tracking
    - Max size enforcement
    - Membership queries (same_coalition, coalition_size, is_full)

NOT responsible for:
    - Proposal lifecycle (→ NegotiationSystem)
    - Trust updates (→ TrustSystem)
    - Resource effects (→ env._apply_coalition_damping)

Coalition ID convention:
    The coalition_id equals the founding agent's ID by default (stable
    across the episode even if the founder later leaves).
    This keeps the ID assignment deterministic and inspectable.

Design:
    - All agents start solo (membership = None).
    - new_coalition(founder) → creates a coalition, founder joins, returns coalition_id.
    - join(agent_id, coalition_id) → adds agent; enforces max_coalition_size.
    - leave(agent_id) → removes agent; coalition disbanded if last member.
"""

from __future__ import annotations

from typing import Dict, Optional, Set

from district_accord.utils.config import EnvConfig
from district_accord.utils.types import AgentID


class CoalitionSystem:
    """
    Centralized coalition membership tracker.

    Attributes:
        memberships:    Dict[AgentID, Optional[int]] — agent → coalition_id or None.
        _members:       Dict[int, Set[AgentID]] — coalition_id → set of member IDs.
        _next_id:       Next coalition ID to assign (monotonically increasing).
        config:         EnvConfig (for max_coalition_size).
    """

    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        self.memberships: Dict[AgentID, Optional[int]] = {}
        self._members: Dict[int, Set[AgentID]] = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, num_districts: int) -> None:
        """
        Reset coalition state for a new episode.

        All agents start unaffiliated (memberships[i] = None).
        """
        self.memberships = {i: None for i in range(num_districts)}
        self._members = {}
        self._next_id = 0

    # ------------------------------------------------------------------
    # Formation
    # ------------------------------------------------------------------

    def new_coalition(self, founder: AgentID) -> int:
        """
        Create a new coalition and add `founder` as its first member.

        If founder is already in a coalition, they leave it first.

        Returns:
            The new coalition_id.
        """
        if self.memberships.get(founder) is not None:
            self.leave(founder)

        coalition_id = self._next_id
        self._next_id += 1
        self._members[coalition_id] = set()
        # Join (size check: founder always fits in a new coalition)
        self._members[coalition_id].add(founder)
        self.memberships[founder] = coalition_id
        return coalition_id

    def join(self, agent_id: AgentID, coalition_id: int) -> bool:
        """
        Add `agent_id` to `coalition_id`.

        Enforces max_coalition_size.  If agent is already in this coalition,
        no-ops and returns True.  If agent is in a *different* coalition, they
        leave it first.

        Returns:
            True if join succeeded; False if coalition is full.
        """
        if coalition_id not in self._members:
            # Coalition doesn't exist; caller should use new_coalition first
            return False

        current = self.memberships.get(agent_id)
        if current == coalition_id:
            return True  # already in this coalition, no-op

        # Enforce size limit (count existing members + this new one)
        if len(self._members[coalition_id]) >= self.config.max_coalition_size:
            return False

        # Leave current coalition if any
        if current is not None:
            self.leave(agent_id)

        self._members[coalition_id].add(agent_id)
        self.memberships[agent_id] = coalition_id
        return True

    def leave(self, agent_id: AgentID) -> bool:
        """
        Remove `agent_id` from their current coalition.

        If they were the last member, the coalition is disbanded.

        Returns:
            True if agent was in a coalition and left; False if they were solo.
        """
        coalition_id = self.memberships.get(agent_id)
        if coalition_id is None:
            return False

        self._members[coalition_id].discard(agent_id)
        self.memberships[agent_id] = None

        # Disband if empty
        if not self._members.get(coalition_id):
            del self._members[coalition_id]

        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_coalition(self, agent_id: AgentID) -> Optional[int]:
        """Return the coalition_id of agent, or None if unaffiliated."""
        return self.memberships.get(agent_id)

    def coalition_members(self, coalition_id: int) -> Set[AgentID]:
        """Return the set of member IDs for a coalition (empty set if not found)."""
        return set(self._members.get(coalition_id, set()))

    def same_coalition(self, a: AgentID, b: AgentID) -> bool:
        """
        True if a and b are in the same coalition (and both in one).
        """
        ca = self.memberships.get(a)
        cb = self.memberships.get(b)
        return ca is not None and ca == cb

    def coalition_size(self, agent_id: AgentID) -> int:
        """
        Return the number of members in agent's coalition.

        Returns 0 if agent is not in any coalition.
        """
        coalition_id = self.memberships.get(agent_id)
        if coalition_id is None:
            return 0
        return len(self._members.get(coalition_id, set()))

    def is_full(self, coalition_id: int) -> bool:
        """True if coalition has reached max_coalition_size."""
        return len(self._members.get(coalition_id, set())) >= self.config.max_coalition_size

    def active_coalitions(self) -> Dict[int, Set[AgentID]]:
        """Return a copy of the active coalition membership dict."""
        return {cid: set(members) for cid, members in self._members.items()}

    def to_dict(self) -> dict:
        """Serialisable snapshot for info/logging."""
        return {
            "memberships": dict(self.memberships),
            "coalitions": {
                str(cid): sorted(members)
                for cid, members in self._members.items()
            },
        }

    def __repr__(self) -> str:  # pragma: no cover
        n_coalitions = len(self._members)
        n_affiliated = sum(1 for v in self.memberships.values() if v is not None)
        return (
            f"CoalitionSystem("
            f"coalitions={n_coalitions}, "
            f"affiliated={n_affiliated}/{len(self.memberships)})"
        )
