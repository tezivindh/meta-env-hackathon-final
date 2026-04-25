"""
district_accord/spaces/action.py
==================================
Structured action space definition and validation for District Accord.

Provides:
  - ActionSpace: validates, samples, and introspects the action space.
  - make_default_parsed_action: fills in all defaults for a given action type.
  - validate_parsed_action: standalone validator (used by ActionParser).

Design:
  - ParsedAction (from utils/types.py) is the canonical structured action.
  - env.step() accepts both DiscreteAction and ParsedAction; normalisation
    happens via _normalize_actions() in the env.
  - ActionSpace does NOT depend on the environment; it is stateless.

Phase 3 extension:
  - validate() will enforce target/amount constraints for coalition actions.
  - sample() will optionally bias sampling towards active actions only.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np

from district_accord.utils.config import EnvConfig
from district_accord.utils.types import (
    AgentID,
    DiscreteAction,
    ParsedAction,
    PHASE3_ACTIVE_ACTIONS,
)

if TYPE_CHECKING:
    from district_accord.core.district import DistrictState


# ---------------------------------------------------------------------------
# Default constructor
# ---------------------------------------------------------------------------

def make_default_parsed_action(
    action_type: DiscreteAction,
    raw: str = "",
    target: Optional[int] = None,
    amount: Optional[float] = None,
    resource_split: Optional[np.ndarray] = None,
) -> ParsedAction:
    """
    Construct a ParsedAction with sensible defaults.

    Args:
        action_type:    The discrete action type.
        raw:            Original text string (empty = auto-filled from action name).
        target:         Optional target district ID.
        amount:         Optional resource amount.
        resource_split: Optional (2,) split vector.  Defaults to [0.5, 0.5].

    Returns:
        ParsedAction dict (is a plain dict at runtime).
    """
    if resource_split is None:
        resource_split = np.array([0.5, 0.5], dtype=np.float32)
    else:
        resource_split = np.asarray(resource_split, dtype=np.float32)

    return ParsedAction(
        action_type=action_type,
        resource_split=resource_split,
        target=target,
        amount=amount,
        raw=raw or action_type.name.lower(),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_parsed_action(
    action: ParsedAction,
    config: EnvConfig,
    agent_id: AgentID,
) -> None:
    """
    Validate a ParsedAction.  Raises ValueError or TypeError on invalid input.

    Checks:
        - action_type is a valid DiscreteAction
        - resource_split has shape (2,), dtype float32, values in [0, 1]
        - target (if present) is a valid agent id in [0, num_districts)
        - amount (if present) is a float in [0.0, 1.0]

    Phase 3: will add coalition-specific constraints here (e.g. can't propose
    to yourself, amount must be ≤ own resources).
    """
    if not isinstance(action.get("action_type"), DiscreteAction):
        try:
            action["action_type"] = DiscreteAction(int(action["action_type"]))
        except (ValueError, TypeError, KeyError):
            raise ValueError(
                f"District {agent_id}: 'action_type' must be a DiscreteAction. "
                f"Got {action.get('action_type')!r}."
            )

    split = action.get("resource_split")
    if split is None:
        action["resource_split"] = np.array([0.5, 0.5], dtype=np.float32)
    else:
        split = np.asarray(split, dtype=np.float32)
        if split.shape != (2,):
            raise ValueError(
                f"District {agent_id}: 'resource_split' must have shape (2,), "
                f"got {split.shape}."
            )
        if np.any(split < 0) or np.any(split > 1):
            raise ValueError(
                f"District {agent_id}: 'resource_split' values must be in [0, 1], "
                f"got {split}."
            )
        action["resource_split"] = split

    target = action.get("target")
    if target is not None:
        if not isinstance(target, (int, np.integer)):
            raise TypeError(
                f"District {agent_id}: 'target' must be int, got {type(target).__name__!r}."
            )
        if not (0 <= int(target) < config.num_districts):
            raise ValueError(
                f"District {agent_id}: 'target' {target} out of range "
                f"[0, {config.num_districts})."
            )

    amount = action.get("amount")
    if amount is not None:
        if not isinstance(amount, (int, float, np.floating)):
            raise TypeError(
                f"District {agent_id}: 'amount' must be float, got {type(amount).__name__!r}."
            )
        if not (0.0 <= float(amount) <= 1.0):
            raise ValueError(
                f"District {agent_id}: 'amount' must be in [0.0, 1.0], got {amount}."
            )


# ---------------------------------------------------------------------------
# Dynamic action mask
# ---------------------------------------------------------------------------

def build_action_mask(
    agent_id: AgentID,
    district: "DistrictState",
    all_districts: Dict[AgentID, "DistrictState"],
    collapsed: Optional[Dict[AgentID, bool]],
    config: EnvConfig,
    *,
    pending_proposals: Optional[Set[int]] = None,
    active_request_agents: Optional[Set[int]] = None,
    coalition_membership: Optional[Dict[int, Optional[int]]] = None,
) -> np.ndarray:
    """
    Compute the state-dependent action validity mask for one agent.

    Returns a float32 array of shape ``(len(DiscreteAction),)``.
    1.0 = valid;  0.0 = invalid  (attempting an invalid action triggers
    mask_violation_penalty and is silently converted to IGNORE).

    Mask rules
    ----------
    INVEST, DEFEND, IGNORE
        Always valid (1.0).

    RECOVER
        Valid iff ``district.resources >= config.recover_resource_cost``.
        Prevents agents from attempting recovery they cannot afford.

    REQUEST_AID
        Valid iff ``district.stability < config.aid_request_stability_threshold``
        AND the agent is not already in ``active_request_agents``.
        Phase 3+: additional coalition membership checks.

    SHARE_RESOURCES
        Valid iff ``district.resources > config.min_share_threshold``
        AND at least one non-self, non-collapsed target exists.
        Phase 3+: coalition filters, max share amount enforcement.

    PROPOSE_COALITION
        Valid iff at least one proposable target exists:
            - target != agent_id
            - target is not collapsed
            - Phase 3+: target not in same coalition as agent

    ACCEPT_COALITION
        Valid iff ``pending_proposals`` is non-empty (someone proposed TO
        this agent and the proposal is still outstanding).
        Phase 2: always 0 (no negotiation system yet).

    REJECT_COALITION
        Same condition as ACCEPT_COALITION.
        Phase 2: always 0.

    Args:
        agent_id:              Target agent.
        district:              Agent's own DistrictState.
        all_districts:         All district states (for target validation).
        collapsed:             Per-agent collapse status.  None = none collapsed.
        config:                EnvConfig.
        pending_proposals:     Set of AgentIDs that have proposed to this agent.
                               Phase 3+ arg; None = empty in Phase 2.
        active_request_agents: Set of AgentIDs already requesting aid.
                               Phase 3+ arg; None = empty in Phase 2.
        coalition_membership:  Dict mapping agent → coalition ID (int) or None.
                               Phase 3+ arg; None = all agents solo.

    Returns:
        np.ndarray of shape ``(len(DiscreteAction),)``, dtype float32.
    """
    _collapsed = collapsed or {}
    _pending = pending_proposals or set()
    _active_req = active_request_agents or set()

    mask = np.zeros(len(DiscreteAction), dtype=np.float32)

    # ── Phase 1/2 base actions: always valid ────────────────────────────
    mask[DiscreteAction.INVEST] = 1.0
    mask[DiscreteAction.DEFEND] = 1.0
    mask[DiscreteAction.IGNORE] = 1.0

    # ── RECOVER ─────────────────────────────────────────────────────────
    if district.resources >= config.recover_resource_cost:
        mask[DiscreteAction.RECOVER] = 1.0

    # ── Pre-compute valid targets (shared by several rules below) ───────
    valid_targets: List[int] = [
        i for i in all_districts
        if i != agent_id and not _collapsed.get(i, False)
    ]
    has_valid_target = len(valid_targets) > 0

    # ── REQUEST_AID ─────────────────────────────────────────────────────
    if (
        district.stability < config.aid_request_stability_threshold
        and agent_id not in _active_req
    ):
        mask[DiscreteAction.REQUEST_AID] = 1.0

    # ── SHARE_RESOURCES ─────────────────────────────────────────────────
    if district.resources > config.min_share_threshold and has_valid_target:
        mask[DiscreteAction.SHARE_RESOURCES] = 1.0

    # ── PROPOSE_COALITION ───────────────────────────────────────────────
    if has_valid_target:
        if coalition_membership is not None:
            # Phase 3+: can only propose to districts outside own coalition.
            own_id = coalition_membership.get(agent_id)
            proposable = [
                i for i in valid_targets
                if (own_id is None or coalition_membership.get(i) != own_id)
            ]
            if proposable:
                mask[DiscreteAction.PROPOSE_COALITION] = 1.0
        else:
            # Phase 2: no coalition system → can propose to any valid target.
            mask[DiscreteAction.PROPOSE_COALITION] = 1.0

    # ── ACCEPT_COALITION ────────────────────────────────────────────────
    # Requires at least one pending proposal addressed to this agent.
    if _pending:
        mask[DiscreteAction.ACCEPT_COALITION] = 1.0

    # ── REJECT_COALITION ────────────────────────────────────────────────
    # Same condition as ACCEPT.
    if _pending:
        mask[DiscreteAction.REJECT_COALITION] = 1.0

    return mask


# ---------------------------------------------------------------------------
# ActionSpace
# ---------------------------------------------------------------------------

class ActionSpace:
    """
    Defines, validates, and introspects the structured action space.

    Attributes:
        config:             EnvConfig reference.
        num_total_actions:  Total actions defined (inc. Phase 3+ stubs).
        num_active_actions: Actions with implemented effects in current phase.

    Usage:
        space = ActionSpace(config)
        mask = space.action_mask(district_id)   # (num_total_actions,) float32
        space.validate(parsed_action, district_id)
        sampled = space.sample(active_only=True)
    """

    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        self.num_total_actions: int = len(DiscreteAction)
        self.num_active_actions: int = PHASE3_ACTIVE_ACTIONS

    def action_mask(
        self,
        district_id: Optional[AgentID] = None,
    ) -> np.ndarray:
        """
        **Static** base action mask — which actions have implemented effects.

        This is the *theoretical* mask ("what is available this phase?"), NOT
        the runtime state-dependent mask ("is this valid right now?").
        For the runtime mask use ``build_action_mask()`` from this module.

        Phase 2: first PHASE2_ACTIVE_ACTIONS entries = 1.0, rest = 0.0.
        Phase 3+: will also enable coalition actions.

        Args:
            district_id: Reserved for future per-agent static defaults.

        Returns:
            np.ndarray of shape (num_total_actions,), dtype float32.
        """
        _ = district_id  # unused in Phase 2
        mask = np.zeros(self.num_total_actions, dtype=np.float32)
        mask[: self.num_active_actions] = 1.0
        return mask

    def validate(self, action: ParsedAction, agent_id: AgentID) -> None:
        """Delegate to validate_parsed_action."""
        validate_parsed_action(action, self.config, agent_id)

    def contains(self, action: ParsedAction, agent_id: AgentID) -> bool:
        """True if action passes validation; does not raise."""
        try:
            self.validate(action, agent_id)
            return True
        except (ValueError, TypeError):
            return False

    def sample(self, active_only: bool = True) -> ParsedAction:
        """
        Sample a random valid action uniformly.

        Args:
            active_only: If True, sample only from PHASE2_ACTIVE_ACTIONS.

        Returns:
            Random ParsedAction with default parameters.
        """
        n = self.num_active_actions if active_only else self.num_total_actions
        action_type = DiscreteAction(np.random.randint(0, n))
        return make_default_parsed_action(action_type)

    def active_action_types(self) -> List[DiscreteAction]:
        """Return list of currently implemented DiscreteAction types."""
        return [DiscreteAction(i) for i in range(self.num_active_actions)]

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ActionSpace("
            f"total={self.num_total_actions}, "
            f"active={self.num_active_actions})"
        )
