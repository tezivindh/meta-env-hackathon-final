"""
district_accord/utils/types.py
==============================
Shared type aliases, enums, and constants — Phase 2 extended.

Phase 2 additions:
  - DiscreteAction: RECOVER (active) + 5 Phase-3 stubs
  - PHASE2_ACTIVE_ACTIONS: count of currently implemented actions
  - ACTION_STR_MAP: extended with all new token strings
  - ParsedAction: structured action TypedDict returned by parse_structured()

Design decisions encoded here:
    ACTION_FORMAT  = "structured_text"  → RawAction = str
    ACTION_PARSER  = "external"         → env.step() receives DiscreteAction/ParsedAction
    OBSERVATION_MODE = "dict"           → ObsDict = Dict[str, np.ndarray]
    STATE_AUTHORITY = "centralized_env" → AgentID keys are ints owned by env
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Optional, TypedDict

import numpy as np


# ---------------------------------------------------------------------------
# Agent identity
# ---------------------------------------------------------------------------

AgentID = int
RawAction = str
ActionMap = Dict[AgentID, "DiscreteAction"]


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

class DiscreteAction(IntEnum):
    """
    Full action space across all phases.

    Phase 2 active (have implemented effects):
        INVEST, DEFEND, IGNORE, RECOVER

    Phase 3+ defined (parsed + validated, but map to IGNORE in Phase 2):
        REQUEST_AID, SHARE_RESOURCES, PROPOSE_COALITION,
        ACCEPT_COALITION, REJECT_COALITION

    IntEnum values are STABLE — adding new actions appends at the end.
    """

    INVEST = 0           # Grow resources; mild stability boost.
    DEFEND = 1           # Shore up stability; costs resources; reduces exposure.
    IGNORE = 2           # No active action; passive drains still apply.
    RECOVER = 3          # Phase 2: emergency stability recovery; costs resources.
    REQUEST_AID = 4      # Phase 3+: request support from coalition peers.
    SHARE_RESOURCES = 5  # Phase 3+: transfer resources to a target district.
    PROPOSE_COALITION = 6  # Phase 3+: invite a district to join a coalition.
    ACCEPT_COALITION = 7   # Phase 3+: accept a pending coalition invitation.
    REJECT_COALITION = 8   # Phase 3+: decline a coalition invitation.


# Number of actions with implemented effects per phase.
# PHASE2_ACTIVE_ACTIONS is retained for backward compatibility with Phase 2 tests.
PHASE2_ACTIVE_ACTIONS: int = 4   # INVEST, DEFEND, IGNORE, RECOVER
PHASE3_ACTIVE_ACTIONS: int = 9   # all nine actions

# Locked column ordering for the "others" observation tensor (N-1, 4).
# This list must NEVER be reordered — doing so silently breaks trained policies.
# To add columns, append here and update ObservationBuilder._build_others().
OTHERS_SCHEMA: List[str] = ["resources", "stability", "trust", "coalition_flag"]

# Canonical set of all action token strings (lowercase).
# Phase 3 strings are registered here so the parser is forward-compatible.
ACTION_STR_MAP: Dict[str, DiscreteAction] = {
    "invest": DiscreteAction.INVEST,
    "defend": DiscreteAction.DEFEND,
    "ignore": DiscreteAction.IGNORE,
    "recover": DiscreteAction.RECOVER,
    "request_aid": DiscreteAction.REQUEST_AID,
    "share": DiscreteAction.SHARE_RESOURCES,
    "propose": DiscreteAction.PROPOSE_COALITION,
    "accept": DiscreteAction.ACCEPT_COALITION,
    "reject": DiscreteAction.REJECT_COALITION,
}


# ---------------------------------------------------------------------------
# Structured action type (Phase 2)
# ---------------------------------------------------------------------------

class ParsedAction(TypedDict):
    """
    Structured action dict returned by ActionParser.parse_structured().

    All fields are required; defaults are filled in by make_default_parsed_action()
    and ActionParser._parse_single_structured().

    Fields:
        action_type:    DiscreteAction enum (always present).
        resource_split: (2,) float32 array summing to ≤ 1.0.
                        Represents how the agent allocates available resources
                        across two priority buckets.  Defaults to [0.5, 0.5].
        target:         Optional AgentID for directed actions (SHARE, PROPOSE, etc.).
                        None for non-targeted actions.
        amount:         Optional resource amount (0.0–1.0) for SHARE_RESOURCES.
                        None when not applicable.
        raw:            Original text string, kept for debugging / logging.

    Note: ParsedAction is a TypedDict → instances are plain dicts at runtime.
    Use `isinstance(x, dict) and "action_type" in x` to detect at runtime.
    """

    action_type: DiscreteAction
    resource_split: np.ndarray   # shape (2,), dtype float32
    target: Optional[int]
    amount: Optional[float]
    raw: str


# ---------------------------------------------------------------------------
# Observation types
# ---------------------------------------------------------------------------

ObsDict = Dict[str, np.ndarray]
MultiAgentObs = Dict[AgentID, ObsDict]
RewardDict = Dict[AgentID, float]


# ---------------------------------------------------------------------------
# Crisis tiers (unchanged from Phase 1)
# ---------------------------------------------------------------------------

class CrisisTier(IntEnum):
    """
    Five-tier crisis severity classification.

    Thresholds (crisis_level):
        CALM      → [0.0, 0.2)
        ELEVATED  → [0.2, 0.4)
        CRITICAL  → [0.4, 0.6)
        EMERGENCY → [0.6, 0.8)
        COLLAPSE  → [0.8, 1.0]
    """

    CALM = 0
    ELEVATED = 1
    CRITICAL = 2
    EMERGENCY = 3
    COLLAPSE = 4


CRISIS_TIER_THRESHOLDS: Dict[CrisisTier, float] = {
    CrisisTier.CALM: 0.0,
    CrisisTier.ELEVATED: 0.2,
    CrisisTier.CRITICAL: 0.4,
    CrisisTier.EMERGENCY: 0.6,
    CrisisTier.COLLAPSE: 0.8,
}


def crisis_level_to_tier(level: float) -> CrisisTier:
    """Map a scalar crisis level in [0, 1] to its CrisisTier."""
    if level >= 0.8:
        return CrisisTier.COLLAPSE
    elif level >= 0.6:
        return CrisisTier.EMERGENCY
    elif level >= 0.4:
        return CrisisTier.CRITICAL
    elif level >= 0.2:
        return CrisisTier.ELEVATED
    else:
        return CrisisTier.CALM
