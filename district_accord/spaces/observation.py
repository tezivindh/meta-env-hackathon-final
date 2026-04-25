"""
district_accord/spaces/observation.py
=======================================
ObservationBuilder — constructs per-agent observations for DistrictAccordEnv.

Encapsulates:
  - Observation dict structure (all keys, shapes, dtypes)
  - Neighbor noise injection (information asymmetry)
  - Stability-delta tracking across turns
  - Flat vector construction (dynamic, not hardcoded)
  - Action mask computation

This module was split from env._build_agent_obs() to keep the env focused
on the episode lifecycle and make obs construction independently testable.

Phase 2 observation layout per agent (N = num_districts):

    Key             Shape           Description
    ─────────────────────────────────────────────────────────────────
    "self"          (4,)            [resources, stability,
                                     crisis_exposure, stability_delta]
    "others"        (N-1, 2)        Noisy peer view: [resources, stability]
    "crisis"        (2,)            [crisis_level, normalized_tier]
    "turn"          (2,)            [progress, remaining] ∈ [0, 1]
    "action_mask"   (num_actions,)  Binary validity mask (NOT in flat)
    "flat"          (2N+6,)         Concat: self‖others.flatten()‖crisis‖turn

Phase 3 extensions (add here):
  - "public_state": coalition proposals, trust signals
  - "neighbors" → richer (N-1, 8) with influence, policy_vector partial views
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np

from district_accord.spaces.action import build_action_mask
from district_accord.utils.config import EnvConfig
from district_accord.utils.types import (
    AgentID,
    DiscreteAction,
    ObsDict,
    PHASE3_ACTIVE_ACTIONS,
)

if TYPE_CHECKING:
    from district_accord.core.crisis import CrisisSystem
    from district_accord.core.district import DistrictState


# ---------------------------------------------------------------------------
# Helper: standalone flat builder (usable outside the class)
# ---------------------------------------------------------------------------

def build_flat_obs(obs_dict: ObsDict) -> np.ndarray:
    """
    Build a flat float32 vector from a structured obs dict.

    Concatenation order: self ‖ others.flatten() ‖ crisis ‖ turn
    "action_mask" is intentionally excluded (it is semantic, not numeric state).

    Args:
        obs_dict: An obs dict as returned by ObservationBuilder.build().

    Returns:
        np.ndarray of shape (K,), dtype float32.
    """
    parts = [
        obs_dict["self"],
        obs_dict["others"].flatten(),
        obs_dict["crisis"],
        obs_dict["turn"],
    ]
    return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------------------
# ObservationBuilder
# ---------------------------------------------------------------------------

class ObservationBuilder:
    """
    Stateful per-episode observation constructor.

    State:
        _prev_stability: Dict[AgentID, float]
            Stability values from the previous turn, used to compute
            stability_delta in the "self" observation component.

    Lifecycle:
        builder = ObservationBuilder(config, rng)
        builder.reset(districts)                 # start of episode
        for each turn:
            obs = builder.build(agent_id, districts, crisis, turn, max_turns)
        builder.update_prev_stability(districts)  # call AFTER build(), BEFORE next step
    """

    def __init__(
        self,
        config: EnvConfig,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self._rng = rng
        self._prev_stability: Dict[AgentID, float] = {}
        self._num_actions: int = len(DiscreteAction)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, districts: Dict[AgentID, "DistrictState"]) -> None:
        """
        Initialise stability tracking at the start of a new episode.

        Sets prev_stability = current stability for all districts so that
        stability_delta = 0 in the first observation (correct: no prior turn).
        """
        self._prev_stability = {i: d.stability for i, d in districts.items()}

    def update_prev_stability(
        self, districts: Dict[AgentID, "DistrictState"]
    ) -> None:
        """
        Update the stability baseline for next turn's delta computation.

        Must be called AFTER build() in each step:
            obs = builder.build(...)
            builder.update_prev_stability(districts)   ← here
        """
        self._prev_stability = {i: d.stability for i, d in districts.items()}

    # ------------------------------------------------------------------
    # Public: build one agent's observation
    # ------------------------------------------------------------------

    def build(
        self,
        agent_id: AgentID,
        districts: Dict[AgentID, "DistrictState"],
        crisis: "CrisisSystem",
        turn: int,
        max_turns: int,
        collapsed: Optional[Dict[AgentID, bool]] = None,
        pending_proposals: Optional[Set[int]] = None,
        active_request_agents: Optional[Set[int]] = None,
        coalition_membership: Optional[Dict[int, Optional[int]]] = None,
        trust_matrix: Optional[Dict[int, Dict[int, float]]] = None,
    ) -> ObsDict:
        """
        Construct the full observation dict for a single agent.

        Args:
            agent_id:              ID of the agent receiving the observation.
            districts:             Full district state dict (env is authoritative).
            crisis:                Current CrisisSystem.
            turn:                  Current turn index.
            max_turns:             Episode length (from config).
            collapsed:             Per-agent collapse booleans.  None = none collapsed.
            pending_proposals:     Set of AgentIDs that proposed TO this agent
                                   (Phase 3+).  None = empty set.
            active_request_agents: Set of agents with active aid requests
                                   (Phase 3+).  None = empty set.
            coalition_membership:  Dict mapping agent → coalition ID
                                   (Phase 3+).  None = all solo.
            trust_matrix:          Dict mapping agent → Dict of [other_id → trust]
                                   (Phase 3+). None = uniform trust.

        Returns:
            ObsDict with keys: "self", "others", "crisis", "turn",
            "action_mask", and optionally "flat".
        """
        district = districts[agent_id]

        # ── "self": (4,) ─────────────────────────────────────────────
        stability_delta = district.stability - self._prev_stability.get(
            agent_id, district.stability
        )
        self_vec = np.array(
            [
                district.resources,
                district.stability,
                district.crisis_exposure,
                stability_delta,
            ],
            dtype=np.float32,
        )

        # ── "others": (N-1, 4) noisy peer view ───────────────────────
        others = self._build_others(agent_id, districts, trust_matrix, coalition_membership)

        # ── "crisis": (2,) ───────────────────────────────────────────
        normalized_tier = crisis.tier.value / 4.0  # maps [0, 4] → [0.0, 1.0]
        crisis_vec = np.array(
            [crisis.crisis_level, normalized_tier],
            dtype=np.float32,
        )

        # ── "turn": (2,) ─────────────────────────────────────────────
        turn_progress = turn / max_turns
        turns_remaining = (max_turns - turn) / max_turns
        turn_vec = np.array([turn_progress, turns_remaining], dtype=np.float32)

        # ── "action_mask": (num_actions,) ────────────────────────────
        action_mask = self._build_action_mask(
            agent_id,
            districts,
            collapsed,
            pending_proposals,
            active_request_agents,
            coalition_membership,
        )

        obs: ObsDict = {
            "self": self_vec,
            "others": others,
            "crisis": crisis_vec,
            "turn": turn_vec,
            "action_mask": action_mask,
        }

        # ── "flat": dynamic concat (excluded: action_mask) ───────────
        if self.config.flatten_observation:
            obs["flat"] = build_flat_obs(obs)

        return obs

    # ------------------------------------------------------------------
    # Public: shape introspection
    # ------------------------------------------------------------------

    @property
    def flat_dim(self) -> int:
        """
        Total flat observation dimension per agent.

        Formula: 4 (self) + (N-1)*4 (others) + 2 (crisis) + 2 (turn)
               = 4N + 4
        """
        n = self.config.num_districts
        return 4 + (n - 1) * 4 + 2 + 2

    def obs_shapes(self) -> Dict[str, tuple]:
        """
        Return expected shapes for all obs keys.
        Useful for space registration (Phase 6 Gym integration).
        """
        n = self.config.num_districts
        shapes = {
            "self": (4,),
            "others": (n - 1, 4),
            "crisis": (2,),
            "turn": (2,),
            "action_mask": (self._num_actions,),
        }
        if self.config.flatten_observation:
            shapes["flat"] = (self.flat_dim,)
        return shapes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_others(
        self,
        agent_id: AgentID,
        districts: Dict[AgentID, "DistrictState"],
        trust_matrix: Optional[Dict[int, Dict[int, float]]] = None,
        coalition_membership: Optional[Dict[int, Optional[int]]] = None,
    ) -> np.ndarray:
        """
        Build the noisy (N-1, 4) peer observation.

        Each row = [resources + ε, stability + ε, trust + ε, same_coalition] for one other district.
        Noise ε ~ N(0, obs_neighbor_noise_std), clipped to [0, 1] post-addition.
        """
        rows: List[np.ndarray] = []
        own_coalition = None if coalition_membership is None else coalition_membership.get(agent_id)
        # Safe default trust
        agent_trusts = trust_matrix.get(agent_id, {}) if trust_matrix else {}

        for other_id in sorted(districts.keys()):
            if other_id == agent_id:
                continue
            other = districts[other_id]
            
            # Coalition check
            same_coalition = 0.0
            if own_coalition is not None and coalition_membership is not None:
                other_coalition = coalition_membership.get(other_id)
                if own_coalition == other_coalition and own_coalition is not None:
                    same_coalition = 1.0

            # Trust lookup
            base_trust = agent_trusts.get(other_id, self.config.trust_init_mean)
            
            base = np.array([other.resources, other.stability, base_trust], dtype=np.float32)
            noise = self._rng.normal(
                0.0, self.config.obs_neighbor_noise_std, size=3
            ).astype(np.float32)
            noisy_base = np.clip(base + noise, 0.0, 1.0)
            
            row = np.array([noisy_base[0], noisy_base[1], noisy_base[2], same_coalition], dtype=np.float32)
            rows.append(row)

        if not rows:
            return np.zeros((0, 4), dtype=np.float32)
        return np.stack(rows, axis=0)  # (N-1, 4)

    def _build_action_mask(
        self,
        agent_id: AgentID,
        districts: Dict[AgentID, "DistrictState"],
        collapsed: Optional[Dict[AgentID, bool]] = None,
        pending_proposals: Optional[Set[int]] = None,
        active_request_agents: Optional[Set[int]] = None,
        coalition_membership: Optional[Dict[int, Optional[int]]] = None,
    ) -> np.ndarray:
        """
        Delegate dynamic mask computation to ``build_action_mask()``.

        All Phase 3+ arguments are forwarded transparently; passing None
        results in the safe Phase 2 defaults (shown in build_action_mask docstring).
        """
        return build_action_mask(
            agent_id=agent_id,
            district=districts[agent_id],
            all_districts=districts,
            collapsed=collapsed,
            config=self.config,
            pending_proposals=pending_proposals,
            active_request_agents=active_request_agents,
            coalition_membership=coalition_membership,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ObservationBuilder("
            f"num_districts={self.config.num_districts}, "
            f"flat_dim={self.flat_dim})"
        )
