"""
district_accord/core/negotiation.py
=====================================
NegotiationSystem — Phase 3 centralized proposal lifecycle manager.

Handles ALL inter-district proposals:
    "coalition" — invitation to form/join a coalition
    "aid"       — request for resource support
    "share"     — direct resource transfer offer

Design principles:
    - Single source of truth for all pending proposals.
    - CoalitionSystem handles membership state ONLY; this module handles
      the proposal lifecycle (create → accept/reject/expire).
    - Deterministic: all iteration over proposals uses sorted(proposal_id).
    - No randomness.

Proposal lifecycle:
    create()  →  [PENDING, ttl=T]
                      ↓              ↓            ↓
                  accept()       reject()     tick() × T
                      ↓              ↓            ↓
                [RESOLVED]     [RESOLVED]    [EXPIRED]

Anti-spam constraints:
    - proposal_cost:         resource penalty per proposal (applied by env)
    - max_pending_proposals: per-proposer outgoing cap
    - proposal_cooldown:     minimum turns between proposals to same target
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from district_accord.utils.config import EnvConfig
from district_accord.utils.types import AgentID


# ---------------------------------------------------------------------------
# Proposal data structure
# ---------------------------------------------------------------------------

@dataclass
class Proposal:
    """
    Immutable-ish record of a single inter-district proposal.

    Fields:
        proposal_id:  Monotonically increasing integer (unique within episode).
        proposer:     Agent that created the proposal.
        target:       Agent that must accept/reject.
        kind:         "coalition" | "aid" | "share"
        terms:        Arbitrary dict; e.g. {"amount": 0.1} for share proposals.
        ttl:          Turns remaining before auto-expiry.  Decremented by tick().
        turn_created: Turn index when proposal was created (for cooldown calc).
    """
    proposal_id: int
    proposer: AgentID
    target: AgentID
    kind: str
    terms: dict = field(default_factory=dict)
    ttl: int = 3
    turn_created: int = 0


# ---------------------------------------------------------------------------
# NegotiationSystem
# ---------------------------------------------------------------------------

class NegotiationSystem:
    """
    Centralized proposal lifecycle manager.

    Attributes:
        _proposals:  Dict[int, Proposal] — all currently-pending proposals,
                     keyed by proposal_id.
        _next_id:    Monotonically increasing proposal counter.
        _cooldowns:  Dict[(proposer, target), turn_expires] — prevents re-spam.
    """

    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        self._proposals: Dict[int, Proposal] = {}
        self._next_id: int = 0
        self._cooldowns: Dict[Tuple[int, int], int] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, num_districts: int) -> None:  # noqa: ARG002
        """Clear all proposals and cooldowns for a new episode."""
        self._proposals = {}
        self._next_id = 0
        self._cooldowns = {}

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(
        self,
        proposer: AgentID,
        target: AgentID,
        kind: str,
        terms: dict,
        current_turn: int,
    ) -> Optional[Proposal]:
        """
        Create a new proposal.

        Returns:
            The new Proposal if created.
            None if any constraint blocks creation:
                - proposer == target
                - cooldown still active for (proposer, target)
                - proposer has reached max_pending_proposals

        The env is responsible for deducting proposal_cost AFTER this returns
        a non-None result.

        Cooldown is set immediately on creation (blocks re-proposal while
        one is already pending to the same target).
        """
        if proposer == target:
            return None

        # Cooldown check
        key = (proposer, target)
        cooldown_end = self._cooldowns.get(key, 0)
        if current_turn < cooldown_end:
            return None

        # Max outgoing proposals check
        if len(self.pending_from(proposer)) >= self.config.max_pending_proposals:
            return None

        # Max incoming proposals check (prevents target being flooded)
        if len(self.pending_for(target)) >= self.config.max_pending_proposals:
            return None

        # Create
        proposal = Proposal(
            proposal_id=self._next_id,
            proposer=proposer,
            target=target,
            kind=kind,
            terms=dict(terms),
            ttl=self.config.proposal_ttl,
            turn_created=current_turn,
        )
        self._proposals[self._next_id] = proposal
        self._next_id += 1

        # Set cooldown so proposer can't spam same target while proposal is pending
        self._cooldowns[key] = current_turn + self.config.proposal_cooldown

        return proposal

    # ------------------------------------------------------------------
    # Tick (per-turn TTL expiration)
    # ------------------------------------------------------------------

    def tick(self) -> List[Proposal]:
        """
        Advance one turn: decrement TTL, remove expired proposals.

        Returns:
            List of Proposal objects that expired this tick (for logging/trust).
        """
        expired: List[Proposal] = []
        for pid in sorted(self._proposals):  # deterministic order
            proposal = self._proposals[pid]
            proposal.ttl -= 1
            if proposal.ttl <= 0:
                expired.append(proposal)

        for p in expired:
            del self._proposals[p.proposal_id]

        return expired

    # ------------------------------------------------------------------
    # Accept / Reject
    # ------------------------------------------------------------------

    def accept(self, proposal_id: int, target: AgentID) -> Optional[Proposal]:
        """
        Accept a specific proposal.

        Args:
            proposal_id: ID of the proposal to accept.
            target:      Must match proposal.target (security check).

        Returns:
            The accepted Proposal (now removed from pending), or None if not found
            or target mismatch.
        """
        proposal = self._proposals.get(proposal_id)
        if proposal is None or proposal.target != target:
            return None
        del self._proposals[proposal_id]
        return proposal

    def reject(self, proposal_id: int, target: AgentID) -> bool:
        """
        Reject a specific proposal.

        Args:
            proposal_id: ID of the proposal to reject.
            target:      Must match proposal.target.

        Returns:
            True if found and removed; False otherwise.
        """
        proposal = self._proposals.get(proposal_id)
        if proposal is None or proposal.target != target:
            return False
        del self._proposals[proposal_id]
        return True

    def accept_first(self, target: AgentID, kind: str) -> Optional[Proposal]:
        """
        Accept the first pending proposal of the given kind addressed to target.

        Used by env when agent takes ACCEPT_COALITION without specifying
        a proposal_id (common in RL where the agent just says "accept").

        Returns:
            The accepted Proposal, or None if no matching proposal exists.
        """
        candidates = [p for p in self.pending_for(target) if p.kind == kind]
        if not candidates:
            return None
        return self.accept(candidates[0].proposal_id, target)

    def reject_first(self, target: AgentID, kind: str) -> Optional[Proposal]:
        """
        Reject the first pending proposal of the given kind addressed to target.
        """
        candidates = [p for p in self.pending_for(target) if p.kind == kind]
        if not candidates:
            return None
        success = self.reject(candidates[0].proposal_id, target)
        return candidates[0] if success else None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def pending_for(self, target: AgentID) -> List[Proposal]:
        """
        All proposals currently addressed to `target`, sorted by proposal_id.
        """
        return sorted(
            [p for p in self._proposals.values() if p.target == target],
            key=lambda p: p.proposal_id,
        )

    def pending_from(self, proposer: AgentID) -> List[Proposal]:
        """
        All proposals currently sent by `proposer`, sorted by proposal_id.
        """
        return sorted(
            [p for p in self._proposals.values() if p.proposer == proposer],
            key=lambda p: p.proposal_id,
        )

    def has_active_request(self, agent_id: AgentID) -> bool:
        """True if agent_id has any active outgoing aid request."""
        return any(
            p.proposer == agent_id and p.kind == "aid"
            for p in self._proposals.values()
        )

    def active_requesters(self) -> Set[AgentID]:
        """Set of agent IDs that have at least one active outgoing aid request."""
        return {
            p.proposer for p in self._proposals.values() if p.kind == "aid"
        }

    def coalition_proposers_for(self, target: AgentID) -> Set[AgentID]:
        """
        Set of proposer IDs who have sent a coalition invite to `target`.

        Used to build the dynamic action mask (ACCEPT/REJECT valid iff non-empty).
        """
        return {
            p.proposer for p in self._proposals.values()
            if p.target == target and p.kind == "coalition"
        }

    def all_pending(self) -> List[Proposal]:
        """All pending proposals, sorted by proposal_id (deterministic)."""
        return sorted(self._proposals.values(), key=lambda p: p.proposal_id)

    def to_dict(self) -> dict:
        """Serialisable snapshot for info/logging."""
        return {
            "pending_count": len(self._proposals),
            "proposals": [
                {
                    "id": p.proposal_id,
                    "proposer": p.proposer,
                    "target": p.target,
                    "kind": p.kind,
                    "terms": p.terms,
                    "ttl": p.ttl,
                }
                for p in self.all_pending()
            ],
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"NegotiationSystem("
            f"pending={len(self._proposals)}, "
            f"next_id={self._next_id})"
        )
