"""
district_accord/policy/self_play.py
=====================================
SelfPlayPolicy — simple, mask-aware policy for self-play training.

Three modes:
    "random"              — uniform random over ALL 9 actions (mask ignored;
                            env will enforce the mask via IGNORE fallback).
    "mask_aware_random"   — uniform random over VALID actions only (respects mask).
    "rule_based"          — heuristic mixed strategy:
                              * DEFEND if crisis_exposure high
                              * INVEST if resources high + stability low
                              * PROPOSE_COALITION at start of game
                              * ACCEPT_COALITION when a proposal is pending
                              * RECOVER if stability critical
                              * IGNORE otherwise

Design principles:
    - Always O(N) per act() call.
    - Deterministic with a given seed.
    - Policies live outside the env — zero env state mutation.
    - Intended as a stable training baseline, NOT a hand-tuned optimal policy.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from district_accord.spaces.action import make_default_parsed_action
from district_accord.utils.types import AgentID, DiscreteAction, ParsedAction

# All 9 discrete action indices
_ALL_ACTIONS: List[int] = list(range(len(DiscreteAction)))


class SelfPlayPolicy:
    """
    Configurable self-play policy for DistrictAccordEnv.

    Args:
        mode:  "random" | "mask_aware_random" | "rule_based"
        seed:  RNG seed for reproducibility (None = non-deterministic).

    Usage:
        policy = SelfPlayPolicy(mode="mask_aware_random", seed=42)
        obs = env.reset()
        actions = policy.act(obs, env)
        obs, rewards, done, trunc, info = env.step(actions)
    """

    MODES = frozenset({"random", "mask_aware_random", "rule_based"})

    def __init__(self, mode: str = "mask_aware_random", seed: Optional[int] = None) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode {mode!r}. Valid: {sorted(self.MODES)}")
        self.mode = mode
        self._rng: np.random.Generator = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def act(
        self,
        obs_dict: Dict[AgentID, dict],
        env: "DistrictAccordEnv",  # type: ignore[name-defined]
    ) -> Dict[AgentID, ParsedAction]:
        """
        Produce one action per non-collapsed agent.

        Args:
            obs_dict: Per-agent observations from env.reset() or env.step().
            env:      Live environment reference (read-only in policies).

        Returns:
            Dict[AgentID, ParsedAction] — one entry per non-collapsed agent.
        """
        actions: Dict[AgentID, ParsedAction] = {}

        for agent_id in sorted(obs_dict):
            if env._collapsed.get(agent_id, False):
                continue

            obs = obs_dict[agent_id]
            mask: np.ndarray = obs["action_mask"]  # shape (9,)

            if self.mode == "random":
                chosen = self._random_action(agent_id, env)
            elif self.mode == "mask_aware_random":
                chosen = self._mask_aware_random(mask, agent_id, env)
            else:
                chosen = self._rule_based(obs, mask, agent_id, env)

            actions[agent_id] = chosen

        return actions

    # ------------------------------------------------------------------
    # Mode implementations
    # ------------------------------------------------------------------

    def _random_action(
        self,
        agent_id: AgentID,
        env: "DistrictAccordEnv",  # type: ignore[name-defined]
    ) -> ParsedAction:
        """Uniform random from all 9 actions. Env will IGNORE masked ones."""
        action_type = DiscreteAction(self._rng.integers(0, len(DiscreteAction)))
        return self._build_parsed(action_type, agent_id, env)

    def _mask_aware_random(
        self,
        mask: np.ndarray,
        agent_id: AgentID,
        env: "DistrictAccordEnv",  # type: ignore[name-defined]
    ) -> ParsedAction:
        """Uniform random from VALID (mask=1) actions only."""
        valid = [i for i, m in enumerate(mask) if m == 1]
        if not valid:
            return make_default_parsed_action(DiscreteAction.IGNORE)
        idx = int(self._rng.choice(valid))
        action_type = DiscreteAction(idx)
        return self._build_parsed(action_type, agent_id, env)

    def _rule_based(
        self,
        obs:      dict,
        mask:     np.ndarray,
        agent_id: AgentID,
        env:      "DistrictAccordEnv",  # type: ignore[name-defined]
    ) -> ParsedAction:
        """
        Heuristic mixed strategy.  Priority cascade:

        1. If ACCEPT_COALITION valid → accept (trust signals encourage cooperation)
        2. If stability critical (<0.25) → RECOVER (resource permitting)
        3. If crisis_exposure high (>0.15) → DEFEND
        4. If turn < 5 and PROPOSE valid → PROPOSE to nearest neighbour
        5. If resources high and stability low → INVEST
        6. If resources very low → IGNORE (conserve)
        7. Default → DEFEND
        """
        d = env._districts[agent_id]
        turn = env._turn

        def valid(a: DiscreteAction) -> bool:
            return bool(mask[a.value] == 1)

        # 1. Coalition formation — always beneficial
        if valid(DiscreteAction.ACCEPT_COALITION):
            return make_default_parsed_action(DiscreteAction.ACCEPT_COALITION)

        # 2. Emergency recovery
        if d.stability < 0.25 and valid(DiscreteAction.RECOVER):
            return make_default_parsed_action(DiscreteAction.RECOVER)

        # 3. Crisis defence
        if d.crisis_exposure > 0.15 and valid(DiscreteAction.DEFEND):
            return make_default_parsed_action(DiscreteAction.DEFEND)

        # 4. Early game: propose to an uncollapsed non-coalition neighbour
        if turn < 5 and valid(DiscreteAction.PROPOSE_COALITION):
            target = self._find_propose_target(agent_id, env)
            if target is not None:
                return make_default_parsed_action(
                    DiscreteAction.PROPOSE_COALITION, target=target
                )

        # 5. Invest when resources allow
        if d.resources > 0.35 and d.stability < 0.80 and valid(DiscreteAction.INVEST):
            return make_default_parsed_action(DiscreteAction.INVEST)

        # 6. Resource conservation
        if d.resources < 0.10 and valid(DiscreteAction.IGNORE):
            return make_default_parsed_action(DiscreteAction.IGNORE)

        # 7. Default
        if valid(DiscreteAction.DEFEND):
            return make_default_parsed_action(DiscreteAction.DEFEND)

        return make_default_parsed_action(DiscreteAction.IGNORE)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_parsed(
        self,
        action_type: DiscreteAction,
        agent_id:    AgentID,
        env:         "DistrictAccordEnv",  # type: ignore[name-defined]
    ) -> ParsedAction:
        """Build a ParsedAction with sensible defaults and a valid target if needed."""
        target: Optional[int] = None
        if action_type in (
            DiscreteAction.PROPOSE_COALITION,
            DiscreteAction.REQUEST_AID,
            DiscreteAction.SHARE_RESOURCES,
        ):
            target = self._find_propose_target(agent_id, env)
            if target is None:
                # Fallback: can't act → IGNORE
                return make_default_parsed_action(DiscreteAction.IGNORE)
        return make_default_parsed_action(action_type, target=target)

    def _find_propose_target(
        self,
        agent_id: AgentID,
        env:      "DistrictAccordEnv",  # type: ignore[name-defined]
    ) -> Optional[AgentID]:
        """
        Choose a target for coalition/aid proposals.

        Prefers agents:
            - Not collapsed
            - Not already in the same coalition
            - With lowest stability (most in need of help)

        Returns None if no valid target exists.
        """
        candidates = [
            a for a in sorted(env._districts)
            if a != agent_id
            and not env._collapsed.get(a, False)
        ]
        if not candidates:
            return None
        # Sort by stability ascending (help those who need it most)
        candidates.sort(key=lambda a: env._districts[a].stability)
        return candidates[0]

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"SelfPlayPolicy(mode={self.mode!r})"
