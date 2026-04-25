"""
district_accord/spaces/action_parser.py
=========================================
External action parser: converts raw LLM/human text to DiscreteAction enums
or fully structured ParsedAction dicts.

Design decisions enforced here:
    ACTION_FORMAT = "structured_text"  → input is always a string
    ACTION_PARSER = "external"         → env.step() never receives raw strings

Phase 1 API (unchanged):
    parser.parse(raw)       → Dict[AgentID, DiscreteAction]
    parser.parse_safe(raw)  → Dict[AgentID, DiscreteAction]

Phase 2 API (new):
    parser.parse_structured(raw)        → Dict[AgentID, ParsedAction]
    parser.parse_structured_safe(raw)   → Dict[AgentID, ParsedAction]

Structured text format (Phase 2):
    Simple:     "invest", "defend", "ignore", "recover"
    With params: "share:target=1,amount=0.1"
                 "propose:target=2"
                 "invest:r0=0.7,r1=0.3"

    Format: <action>[:<key>=<value>[,<key>=<value>...]]
    All param values are scalars (int or float).

Phase 3+ extension:
    When negotiation sub-actions require structured JSON, parse_structured
    will be extended to handle e.g. '{"policy": "invest", "target": 1}'.
    The ParsedAction return type absorbs this naturally.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from district_accord.spaces.action import make_default_parsed_action
from district_accord.utils.config import EnvConfig
from district_accord.utils.types import (
    AgentID,
    DiscreteAction,
    ParsedAction,
    RawAction,
    ACTION_STR_MAP,
)


class ActionParser:
    """
    Converts raw text actions (str) to DiscreteAction enums.

    Attributes:
        config:       EnvConfig reference (for num_districts validation).
        valid_tokens: Frozenset of accepted lowercase strings.
    """

    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        self.valid_tokens: frozenset = frozenset(ACTION_STR_MAP.keys())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        raw_actions: Dict[AgentID, RawAction],
    ) -> Dict[AgentID, DiscreteAction]:
        """
        Strict parser: raise ValueError on any unrecognised token.

        Args:
            raw_actions: Dict mapping AgentID → raw action string.

        Returns:
            Dict mapping AgentID → DiscreteAction.

        Raises:
            ValueError: If any action string is not in ACTION_STR_MAP.
            TypeError:  If any action value is not a string.
        """
        parsed: Dict[AgentID, DiscreteAction] = {}
        for agent_id, raw in raw_actions.items():
            if not isinstance(raw, str):
                raise TypeError(
                    f"District {agent_id}: expected str action, "
                    f"got {type(raw).__name__!r}.  "
                    f"Did you forget to convert the LLM output to a string?"
                )
            token = raw.strip().lower()
            if token not in ACTION_STR_MAP:
                raise ValueError(
                    f"District {agent_id}: unknown action {raw!r}.  "
                    f"Valid tokens: {sorted(self.valid_tokens)}"
                )
            parsed[agent_id] = ACTION_STR_MAP[token]
        return parsed

    def parse_safe(
        self,
        raw_actions: Dict[AgentID, RawAction],
        default: DiscreteAction = DiscreteAction.IGNORE,
    ) -> Dict[AgentID, DiscreteAction]:
        """
        Lenient parser: fall back to `default` on unrecognised tokens.

        Use this during training when LLM output may be malformed.
        The returned dict will always have an entry for every key in
        raw_actions.

        Args:
            raw_actions: Dict mapping AgentID → raw action string.
            default:     DiscreteAction to use when parsing fails.

        Returns:
            Dict mapping AgentID → DiscreteAction (never raises).
        """
        parsed: Dict[AgentID, DiscreteAction] = {}
        for agent_id, raw in raw_actions.items():
            try:
                token = str(raw).strip().lower()
                parsed[agent_id] = ACTION_STR_MAP.get(token, default)
            except Exception:  # noqa: BLE001
                parsed[agent_id] = default
        return parsed

    def valid_action_strings(self) -> List[str]:
        """Return sorted list of valid action strings (for prompting LLMs)."""
        return sorted(self.valid_tokens)

    # ------------------------------------------------------------------
    # Phase 2: structured text parsing
    # ------------------------------------------------------------------

    def parse_structured(
        self,
        raw_actions: Dict[AgentID, RawAction],
    ) -> Dict[AgentID, ParsedAction]:
        """
        Strict structured parser: converts text to ParsedAction dicts.

        Format:  "<action>" or "<action>:<key>=<val>[,<key>=<val>...]"

        Supported param keys:
            target    int   - target district ID (SHARE, PROPOSE, etc.)
            amount    float - resource amount for SHARE_RESOURCES
            r0        float - first component of resource_split
            r1        float - second component of resource_split

        Examples:
            "invest"                   → INVEST, split=[0.5,0.5], no target
            "recover"                  → RECOVER, defaults
            "share:target=2,amount=0.1" → SHARE_RESOURCES, target=2, amount=0.1
            "propose:target=1"         → PROPOSE_COALITION, target=1
            "invest:r0=0.7,r1=0.3"    → INVEST, split=[0.7, 0.3]

        Args:
            raw_actions: Dict mapping AgentID → raw action string.

        Returns:
            Dict mapping AgentID → ParsedAction.

        Raises:
            ValueError: On unknown action token or invalid param values.
            TypeError:  If any action value is not a string.
        """
        return {
            agent_id: self._parse_single_structured(agent_id, raw)
            for agent_id, raw in raw_actions.items()
        }

    def parse_structured_safe(
        self,
        raw_actions: Dict[AgentID, RawAction],
        default: DiscreteAction = DiscreteAction.IGNORE,
    ) -> Dict[AgentID, ParsedAction]:
        """
        Lenient structured parser: never raises.

        Falls back to a default ParsedAction on any parse error.
        Use during RL training when LLM output may be malformed.

        Args:
            raw_actions: Dict mapping AgentID → raw action string.
            default:     DiscreteAction to use when parsing fails.

        Returns:
            Dict mapping AgentID → ParsedAction (never raises).
        """
        result: Dict[AgentID, ParsedAction] = {}
        for agent_id, raw in raw_actions.items():
            try:
                result[agent_id] = self._parse_single_structured(agent_id, raw)
            except Exception:  # noqa: BLE001
                result[agent_id] = make_default_parsed_action(default, raw=str(raw))
        return result

    # ------------------------------------------------------------------
    # Private: single-action structured parser
    # ------------------------------------------------------------------

    def _parse_single_structured(
        self, agent_id: AgentID, raw: RawAction
    ) -> ParsedAction:
        """
        Parse one raw text action into a ParsedAction.

        Args:
            agent_id: Used only for error messages.
            raw:      Raw text string from LLM/policy.

        Returns:
            ParsedAction dict.

        Raises:
            ValueError: Unknown action token or invalid param.
            TypeError:  Input is not a string.
        """
        if not isinstance(raw, str):
            raise TypeError(
                f"District {agent_id}: expected str, got {type(raw).__name__!r}."
            )

        raw_stripped = str(raw).strip()

        # Split on the FIRST colon only: "action:params"
        if ":" in raw_stripped:
            action_str, params_str = raw_stripped.split(":", 1)
        else:
            action_str = raw_stripped
            params_str = ""

        action_str = action_str.strip().lower()
        if action_str not in ACTION_STR_MAP:
            raise ValueError(
                f"District {agent_id}: unknown action token {raw_stripped!r}.  "
                f"Valid tokens: {sorted(self.valid_tokens)}"
            )
        action_type = ACTION_STR_MAP[action_str]

        # Parse key=value pairs separated by commas.
        params = self._parse_params(params_str, agent_id)

        # Extract resource_split from optional r0/r1 params.
        resource_split = self._extract_resource_split(params)

        # Extract optional target.
        target: Optional[int] = None
        if "target" in params:
            try:
                target = int(params["target"])
            except (ValueError, TypeError):
                raise ValueError(
                    f"District {agent_id}: 'target' must be an integer, "
                    f"got {params['target']!r}."
                )

        # Extract optional amount.
        amount: Optional[float] = None
        if "amount" in params:
            try:
                amount = float(params["amount"])
            except (ValueError, TypeError):
                raise ValueError(
                    f"District {agent_id}: 'amount' must be a float, "
                    f"got {params['amount']!r}."
                )
            if not (0.0 <= amount <= 1.0):
                raise ValueError(
                    f"District {agent_id}: 'amount' must be in [0.0, 1.0], "
                    f"got {amount}."
                )

        return make_default_parsed_action(
            action_type=action_type,
            raw=raw_stripped,
            target=target,
            amount=amount,
            resource_split=resource_split,
        )

    def _parse_params(
        self, params_str: str, agent_id: AgentID = 0
    ) -> Dict[str, str]:
        """
        Parse "key=val,key=val,..." into a string dict.

        Silently skips malformed pairs (lenient; error raised per-key downstream).
        """
        params: Dict[str, str] = {}
        if not params_str.strip():
            return params
        for pair in params_str.split(","):
            pair = pair.strip()
            if "=" not in pair:
                continue  # skip malformed param
            key, _, val = pair.partition("=")
            params[key.strip().lower()] = val.strip()
        return params

    def _extract_resource_split(
        self, params: Dict[str, str]
    ) -> "np.ndarray":
        """
        Extract resource_split from params dict.

        Keys: r0 (first component), r1 (second component).
        Both default to 0.5 if absent.
        """
        import numpy as np  # local import to avoid module-level dep
        r0 = float(params.get("r0", 0.5))
        r1 = float(params.get("r1", 0.5))
        return np.array([r0, r1], dtype=np.float32)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ActionParser("
            f"valid={sorted(self.valid_tokens)}, "
            f"num_districts={self.config.num_districts})"
        )
