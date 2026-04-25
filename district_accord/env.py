"""
district_accord/env.py
========================
DistrictAccordEnv — Phase 3 Gym-style multi-agent environment.

Phase 3 additions over Phase 2:
    - CoalitionSystem: deterministic membership management
    - NegotiationSystem: full proposal lifecycle (create/tick/accept/reject)
    - TrustSystem: per-pair trust matrix with update rules and decay
    - Dynamic action mask enforcement (mask violations → IGNORE + penalty)
    - Coalition crisis exposure damping
    - Coalition survival bonus reward signal

Gym-style interface (UNCHANGED from Phase 1):
    obs                              = env.reset()
    obs, rewards, done, trunc, info  = env.step(actions)

Actions (UNCHANGED):
    Dict[AgentID, DiscreteAction]  — Phase 1 input, still fully accepted
    Dict[AgentID, ParsedAction]    — Phase 2/3 structured input

Step processing order (Phase 3 — 19 steps):
     1.  validate_actions
     2.  normalize_actions
     3.  enforce_action_masks  ← converts violations to IGNORE, records penalty set
     4.  process_negotiation   ← PROPOSE/REQUEST_AID create proposals;
                                  ACCEPT/REJECT resolve proposals + update coalition
     5.  apply_resource_actions ← INVEST/DEFEND/RECOVER/SHARE/IGNORE + passive drains
     6.  negotiation.tick()    ← decrement TTL, expire old proposals
     7.  crisis.step()
     8.  update_exposures
     9.  apply_coalition_damping ← exposure *= (1 - damping) for coalition members
    10.  apply_crisis_effects
    11.  clip_all
    12.  advance turn
    13.  update_collapse_status
    14.  check_terminal
    15.  compute_rewards        ← survival + coalition_bonus - mask_penalty
    16.  trust.decay()
    17.  get_obs               ← with trust_matrix + coalition + pending
    18.  update_prev_stability
    19.  build_info

Design decisions (Phase 1 locks honoured):
    STATE_AUTHORITY     = "centralized_env"
    OBSERVATION_MODE    = "dict"
    FLATTEN_OBSERVATION = True (configurable)
    ACTION_FORMAT       = "structured_text"
    ACTION_PARSER       = "external"
    VOTING_MODE         = "deterministic"
    Resolution order    = sorted(agent_ids)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from district_accord.core.coalition import CoalitionSystem
from district_accord.core.crisis import CrisisSystem
from district_accord.core.district import DistrictState
from district_accord.core.negotiation import NegotiationSystem, Proposal
from district_accord.core.trust import TrustSystem
from district_accord.engine.event_bus import EventBus
from district_accord.engine.reward import RewardBreakdown, RewardEngine
from district_accord.engine.state_tracker import StateTracker
from district_accord.engine.turn_manager import TurnManager
from district_accord.spaces.action import build_action_mask, make_default_parsed_action
from district_accord.spaces.observation import ObservationBuilder
from district_accord.utils.config import EnvConfig
from district_accord.utils.types import (
    AgentID,
    DiscreteAction,
    MultiAgentObs,
    ObsDict,
    ParsedAction,
    RewardDict,
)


class DistrictAccordEnv:
    """
    Multi-agent RL environment: District Accord (Phase 3).

    Attributes:
        config:            Frozen EnvConfig.
        turn:              Current turn index [0, max_turns].
        districts:         Read-only view of DistrictState objects.
        crisis:            Read-only access to CrisisSystem.
        num_agents:        Number of active districts.
        observation_shape: Flat obs dimension per agent (4N+4).
    """

    metadata: Dict[str, Any] = {"version": "0.5.0", "phase": 5}

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        """
        Args:
            config: EnvConfig instance.  Default = EnvConfig() (12 districts, 100 turns).
        """
        self.config: EnvConfig = config or EnvConfig()

        self._rng: np.random.Generator = np.random.default_rng(self.config.seed)
        self._crisis: CrisisSystem = CrisisSystem(self.config, self._rng)
        self._obs_builder: ObservationBuilder = ObservationBuilder(self.config, self._rng)
        self._coalition: CoalitionSystem = CoalitionSystem(self.config)
        self._negotiation: NegotiationSystem = NegotiationSystem(self.config)
        self._trust: TrustSystem = TrustSystem(self.config, self._rng)
        self._reward_engine: RewardEngine = RewardEngine(self.config)

        # Phase 5 engine layer
        self._event_bus:     EventBus     = EventBus()
        self._state_tracker: StateTracker = StateTracker()
        self._turn_manager:  TurnManager  = TurnManager(
            config=self.config,
            event_bus=self._event_bus,
            state_tracker=self._state_tracker,
        )

        self._districts: Dict[AgentID, DistrictState] = {}
        self._collapsed: Dict[AgentID, bool] = {}
        self._turn: int = 0
        self._done: bool = False

    # ------------------------------------------------------------------
    # Gym API — reset
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
    ) -> MultiAgentObs:
        """
        Initialise a new episode.

        Args:
            seed: Optional RNG seed for this episode.

        Returns:
            obs: Dict[AgentID, ObsDict] — initial observation for each agent.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._crisis = CrisisSystem(self.config, self._rng)
            self._obs_builder = ObservationBuilder(self.config, self._rng)
            self._trust = TrustSystem(self.config, self._rng)

        self._turn = 0
        self._done = False
        self._districts = self._init_districts()
        self._collapsed = {i: False for i in self._districts}

        self._crisis.reset()
        self._coalition.reset(self.config.num_districts)
        self._negotiation.reset(self.config.num_districts)
        self._trust.reset(self.config.num_districts)
        self._obs_builder.reset(self._districts)

        # Phase 5: clear episode-scoped engine state
        self._event_bus.clear()
        self._state_tracker.reset()

        return self._get_obs()

    # ------------------------------------------------------------------
    # Gym API — step
    # ------------------------------------------------------------------

    def step(
        self,
        actions: Dict[AgentID, Union[DiscreteAction, ParsedAction]],
    ) -> Tuple[MultiAgentObs, RewardDict, bool, bool, Dict[str, Any]]:
        """
        Advance the environment by one turn.

        Args:
            actions: One action per non-collapsed agent.
                     Accepts DiscreteAction, numpy integer, or ParsedAction dict.
                     Text → DiscreteAction conversion must be done externally
                     via ActionParser (ACTION_PARSER = "external").

        Returns:
            obs:       Dict[AgentID, ObsDict]
            rewards:   Dict[AgentID, float]
            done:      True if all districts have collapsed.
            truncated: True if max_turns reached (and not all collapsed).
            info:      Diagnostic dict.

        Raises:
            AssertionError: If called after episode has ended.
            ValueError:     If an active agent's action is missing or invalid.
            TypeError:      If any action has an unsupported type.
        """
        assert not self._done, (
            f"Episode has ended (turn={self._turn}, "
            f"max_turns={self.config.max_turns}). "
            f"Call reset() to start a new episode."
        )
        return self._turn_manager.run_turn(actions, env=self)

    # ------------------------------------------------------------------
    # Internal pipeline (called by TurnManager.run_turn)
    # ------------------------------------------------------------------

    def _execute_step_pipeline(
        self,
        actions: Dict[AgentID, Any],
    ) -> Tuple[MultiAgentObs, RewardDict, bool, bool, Dict[str, Any]]:
        """
        Full 19-step deterministic step pipeline.

        Called exclusively by TurnManager.run_turn().
        Emits events on self._event_bus at each pipeline boundary.
        Returns (obs, rewards, done, truncated, info) WITHOUT the
        engine-layer additions (events/state_snapshot injected by TurnManager).
        """
        # Snapshot pre-step state for reward delta computation (Phase 4/6)
        prev_stability: Dict[AgentID, float] = {
            i: d.stability for i, d in self._districts.items()
        }
        prev_exposure: Dict[AgentID, float] = {
            i: d.crisis_exposure for i, d in self._districts.items()
        }
        # avg positive trust per agent before trust updates this turn
        _trust_mat_pre = self._trust.as_matrix()
        prev_avg_trust: Dict[AgentID, float] = {
            i: (
                sum(max(v, 0.0) for k, v in row.items() if k != i) / max(len(row) - 1, 1)
                if row else 0.0
            )
            for i, row in _trust_mat_pre.items()
        }

        # Step 1 — Validate (type check only; mask logic below)
        self._validate_actions(actions)

        # Step 2 — Normalise to ParsedAction
        normalized: Dict[AgentID, ParsedAction] = self._normalize_actions(actions)

        # Step 3 — Enforce dynamic action masks
        normalized, mask_violated = self._enforce_action_masks(normalized)

        # Emit: action events (sorted for determinism)
        for agent_id in sorted(normalized):
            if agent_id in mask_violated:
                self._event_bus.emit("action_invalid", {
                    "agent_id": agent_id,
                    "action":   normalized[agent_id]["action_type"].name,
                })
            else:
                self._event_bus.emit("action_validated", {
                    "agent_id": agent_id,
                    "action":   normalized[agent_id]["action_type"].name,
                })

        # Step 4 — Process negotiation actions (PROPOSE / REQUEST_AID / ACCEPT / REJECT)
        accept_events, reject_events, created_proposals = (
            self._process_negotiation_actions(normalized)
        )

        # Emit: proposal lifecycle events
        for p in created_proposals:
            self._event_bus.emit("proposal_created", {
                "proposal_id": p.proposal_id,
                "proposer":    p.proposer,
                "target":      p.target,
                "kind":        p.kind,
            })
        for proposer, acceptor in accept_events:
            # New coalition vs. join existing?
            cid_proposer = self._coalition.get_coalition(proposer)
            cid_acceptor = self._coalition.get_coalition(acceptor)
            if cid_proposer == cid_acceptor and cid_proposer is not None:
                # Both now in same coalition
                csize = self._coalition.coalition_size(acceptor)
                etype = "coalition_joined" if csize > 2 else "coalition_formed"
            else:
                etype = "coalition_formed"
            self._event_bus.emit(etype, {
                "proposer": proposer,
                "acceptor": acceptor,
                "coalition_id": self._coalition.get_coalition(acceptor),
            })
        for proposer, rejector in reject_events:
            self._event_bus.emit("proposal_rejected", {
                "proposer": proposer,
                "rejector": rejector,
            })

        # Step 5 — Apply resource-affecting actions + passive drains
        transfers = self._apply_resource_actions(normalized)
        for from_a, to_a, amount in transfers:
            self._event_bus.emit("resource_transferred", {
                "from":   from_a,
                "to":     to_a,
                "amount": round(amount, 4),
            })

        # Step 6 — TTL expiration
        expired = self._negotiation.tick()
        for p in expired:
            self._event_bus.emit("proposal_expired", {
                "proposal_id": p.proposal_id,
                "proposer":    p.proposer,
                "target":      p.target,
            })

        # Step 7 — Crisis advances
        self._crisis.step(list(self._districts.values()))

        # Step 8 — Update crisis exposures
        self._update_exposures()

        # Step 9 — Coalition exposure damping
        self._apply_coalition_damping()

        # Step 10 — Apply crisis effects
        self._apply_crisis_effects()

        # Step 11 — Clip all state values to [0, 1]
        self._clip_all()

        # Step 12 — Advance turn counter
        self._turn += 1

        # Steps 13–14 — Collapse & terminal
        newly_collapsed: Set[AgentID] = self._update_collapse_status()
        for agent_id in sorted(newly_collapsed):
            self._event_bus.emit("collapse", {
                "agent_id":  agent_id,
                "turn":      self._turn,
                "stability": self._districts[agent_id].stability,
            })
        done, truncated = self._check_terminal()

        # Step 15 — Rewards (Phase 4 RewardEngine — unchanged)
        rewards, reward_breakdowns = self._run_reward_engine(
            prev_stability, newly_collapsed, mask_violated,
            prev_exposure, prev_avg_trust,
        )

        # Step 16 — Trust update + decay
        self._update_trust(accept_events, reject_events)
        self._trust.decay()

        # Emit: trust_updated (one event per accept or reject event)
        for proposer, acceptor in accept_events:
            self._event_bus.emit("trust_updated", {
                "agent_i":   proposer,
                "agent_j":   acceptor,
                "direction": "accept_bilateral",
            })
        for proposer, rejector in reject_events:
            self._event_bus.emit("trust_updated", {
                "agent_i":   rejector,
                "agent_j":   proposer,
                "direction": "reject_unilateral",
            })

        # Step 17 — Observations
        obs: MultiAgentObs = self._get_obs()

        # Step 18 — Update stability baseline for next turn’s delta
        self._obs_builder.update_prev_stability(self._districts)

        # Step 19 — Info (events + state_snapshot injected by TurnManager)
        info: Dict[str, Any] = self._build_info(
            normalized, mask_violated, reward_breakdowns
        )

        if done or truncated:
            self._done = True

        return obs, rewards, done, truncated, info

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def turn(self) -> int:
        """Current turn index."""
        return self._turn

    @property
    def districts(self) -> Dict[AgentID, DistrictState]:
        """Shallow copy of the districts dict (read-only by convention)."""
        return dict(self._districts)

    @property
    def crisis(self) -> CrisisSystem:
        """Direct reference to CrisisSystem (read-only by convention)."""
        return self._crisis

    @property
    def coalition(self) -> CoalitionSystem:
        """Direct reference to CoalitionSystem (read-only by convention)."""
        return self._coalition

    @property
    def negotiation(self) -> NegotiationSystem:
        """Direct reference to NegotiationSystem (read-only by convention)."""
        return self._negotiation

    @property
    def trust(self) -> TrustSystem:
        """Direct reference to TrustSystem (read-only by convention)."""
        return self._trust

    @property
    def num_agents(self) -> int:
        return self.config.num_districts

    @property
    def observation_shape(self) -> int:
        """
        Flat observation dimension per agent.  Dynamic: depends on num_districts.
        Formula: 4 * num_districts + 4
        """
        return self._obs_builder.flat_dim

    # ------------------------------------------------------------------
    # Private: episode initialisation
    # ------------------------------------------------------------------

    def _init_districts(self) -> Dict[AgentID, DistrictState]:
        cfg = self.config
        districts: Dict[AgentID, DistrictState] = {}
        for i in range(cfg.num_districts):
            resources = float(np.clip(
                self._rng.normal(cfg.resource_init_mean, cfg.resource_init_std),
                0.1, 1.0,
            ))
            stability = float(np.clip(
                self._rng.normal(cfg.stability_init_mean, cfg.stability_init_std),
                0.1, 1.0,
            ))
            districts[i] = DistrictState(
                district_id=i,
                resources=resources,
                stability=stability,
                crisis_exposure=0.0,
            )
        return districts

    # ------------------------------------------------------------------
    # Private: action validation & normalisation
    # ------------------------------------------------------------------

    def _validate_actions(
        self,
        actions: Dict[AgentID, Union[DiscreteAction, ParsedAction]],
    ) -> None:
        """
        Validate that every non-collapsed agent has a legal action type.

        Checks type only — mask enforcement happens in _enforce_action_masks().
        Collapsed agents are silently skipped.
        """
        for agent_id in self._districts:
            if self._collapsed[agent_id]:
                continue
            if agent_id not in actions:
                raise ValueError(
                    f"Missing action for active district {agent_id}.  "
                    f"Provide a DiscreteAction or ParsedAction for all non-collapsed agents."
                )
            action = actions[agent_id]
            if isinstance(action, dict):
                if "action_type" not in action:
                    raise ValueError(
                        f"District {agent_id}: ParsedAction dict is missing 'action_type' key."
                    )
            elif isinstance(action, (DiscreteAction, int, np.integer)):
                pass  # coercible; handled in _normalize_actions
            else:
                raise TypeError(
                    f"District {agent_id}: action must be DiscreteAction, int, or ParsedAction dict. "
                    f"Got {type(action).__name__!r}.  "
                    f"Pass raw text through ActionParser.parse() first."
                )

    def _normalize_actions(
        self,
        actions: Dict[AgentID, Union[DiscreteAction, ParsedAction]],
    ) -> Dict[AgentID, ParsedAction]:
        """
        Convert every action to a canonical ParsedAction dict.

        DiscreteAction (or coercible int) → make_default_parsed_action().
        ParsedAction dict                 → passed through unchanged.
        """
        normalized: Dict[AgentID, ParsedAction] = {}
        for agent_id, action in actions.items():
            if isinstance(action, dict) and "action_type" in action:
                normalized[agent_id] = action  # type: ignore[assignment]
            else:
                normalized[agent_id] = make_default_parsed_action(
                    DiscreteAction(int(action)),
                    raw=getattr(action, "name", str(action)).lower(),
                )
        return normalized

    # ------------------------------------------------------------------
    # Private: Step 3 — action mask enforcement
    # ------------------------------------------------------------------

    def _enforce_action_masks(
        self,
        normalized: Dict[AgentID, ParsedAction],
    ) -> Tuple[Dict[AgentID, ParsedAction], Set[AgentID]]:
        """
        Validate each action against the current dynamic state-dependent mask.

        For each non-collapsed agent (in sorted order):
            1. Compute per-agent mask via build_action_mask().
            2. If action is masked (mask[action_type] == 0.0):
               → Replace with IGNORE; add to mask_violated set.
            3. Additionally: PROPOSE/SHARE targeting self or a collapsed
               district → also treated as violation.

        Returns:
            (validated_actions, mask_violated_set)
        """
        mask_violated: Set[AgentID] = set()
        result: Dict[AgentID, ParsedAction] = {}

        for agent_id in sorted(self._districts):
            if self._collapsed.get(agent_id, False):
                # Collapsed agents' actions are irrelevant; keep as-is (not applied)
                if agent_id in normalized:
                    result[agent_id] = normalized[agent_id]
                continue

            if agent_id not in normalized:
                continue

            parsed = normalized[agent_id]
            action_type = parsed["action_type"]

            # Build current dynamic mask
            coalition_proposers = self._negotiation.coalition_proposers_for(agent_id)
            active_req = self._negotiation.active_requesters()
            mask = build_action_mask(
                agent_id=agent_id,
                district=self._districts[agent_id],
                all_districts=self._districts,
                collapsed=self._collapsed,
                config=self.config,
                pending_proposals=coalition_proposers,
                active_request_agents=active_req,
                coalition_membership=self._coalition.memberships,
            )

            violated = bool(mask[action_type] == 0.0)

            # Extra: validate target for directed actions
            if not violated and action_type in (
                DiscreteAction.PROPOSE_COALITION,
                DiscreteAction.SHARE_RESOURCES,
            ):
                target = parsed.get("target")
                if target is not None:
                    if target == agent_id:
                        violated = True   # cannot target self
                    elif self._collapsed.get(target, False):
                        violated = True   # cannot target collapsed district

            if violated:
                result[agent_id] = make_default_parsed_action(
                    DiscreteAction.IGNORE,
                    raw=f"[masked:{parsed['raw']}→ignore]",
                )
                mask_violated.add(agent_id)
            else:
                result[agent_id] = parsed

        return result, mask_violated

    # ------------------------------------------------------------------
    # Private: Step 4 — negotiation actions
    # ------------------------------------------------------------------

    def _process_negotiation_actions(
        self,
        normalized: Dict[AgentID, ParsedAction],
    ) -> Tuple[
        List[Tuple[int, int]],
        List[Tuple[int, int]],
        List["Proposal"],
    ]:
        """
        Process PROPOSE_COALITION, REQUEST_AID, ACCEPT_COALITION, REJECT_COALITION.

        Resolution order: sorted(agent_ids) — deterministic.

        Returns:
            accept_events:     List of (proposer, acceptor) tuples.
            reject_events:     List of (proposer, rejector) tuples.
            created_proposals: Proposal objects created this turn (for EventBus).
        """
        accept_events:     List[Tuple[int, int]] = []
        reject_events:     List[Tuple[int, int]] = []
        created_proposals: List[Proposal]        = []

        for agent_id in sorted(self._districts):
            if self._collapsed.get(agent_id, False):
                continue

            parsed = normalized.get(agent_id)
            if parsed is None:
                continue

            action_type = parsed["action_type"]
            district = self._districts[agent_id]

            # ── PROPOSE_COALITION ────────────────────────────────────────
            if action_type == DiscreteAction.PROPOSE_COALITION:
                target = parsed.get("target")
                if (
                    target is not None
                    and target != agent_id
                    and not self._collapsed.get(target, False)
                ):
                    proposal = self._negotiation.create(
                        proposer=agent_id,
                        target=target,
                        kind="coalition",
                        terms={},
                        current_turn=self._turn,
                    )
                    if proposal is not None:
                        district.resources -= self.config.proposal_cost
                        created_proposals.append(proposal)

            # ── REQUEST_AID ──────────────────────────────────────────────
            elif action_type == DiscreteAction.REQUEST_AID:
                target = parsed.get("target")
                if (
                    target is not None
                    and target != agent_id
                    and not self._collapsed.get(target, False)
                ):
                    proposal = self._negotiation.create(
                        proposer=agent_id,
                        target=target,
                        kind="aid",
                        terms={},
                        current_turn=self._turn,
                    )
                    if proposal is not None:
                        district.resources -= self.config.proposal_cost
                        created_proposals.append(proposal)

            # ── ACCEPT_COALITION ─────────────────────────────────────────
            elif action_type == DiscreteAction.ACCEPT_COALITION:
                accepted_proposal = self._negotiation.accept_first(
                    target=agent_id, kind="coalition"
                )
                if accepted_proposal is not None:
                    proposer = accepted_proposal.proposer
                    # Resolve coalition membership
                    proposer_coalition = self._coalition.get_coalition(proposer)
                    if proposer_coalition is None:
                        # Proposer not in coalition yet; form a new one
                        coalition_id = self._coalition.new_coalition(proposer)
                    else:
                        coalition_id = proposer_coalition

                    self._coalition.join(agent_id, coalition_id)
                    accept_events.append((proposer, agent_id))

            # ── REJECT_COALITION ─────────────────────────────────────────
            elif action_type == DiscreteAction.REJECT_COALITION:
                rejected_proposal = self._negotiation.reject_first(
                    target=agent_id, kind="coalition"
                )
                if rejected_proposal is not None:
                    reject_events.append((rejected_proposal.proposer, agent_id))

        return accept_events, reject_events, created_proposals


    # ------------------------------------------------------------------
    # Private: Step 5 — resource-affecting actions
    # ------------------------------------------------------------------

    def _apply_resource_actions(
        self, normalized: Dict[AgentID, ParsedAction]
    ) -> List[Tuple[int, int, float]]:
        """
        Apply resource-affecting actions and per-turn passive drains.

        Resolution order: sorted(agent_ids) — deterministic.

        Returns:
            transfers: List of (from_agent, to_agent, amount) for SHARE_RESOURCES.
                       Empty list if no transfers occurred.  Used by EventBus.

        Phase 3 action effects:
            INVEST:          resources  += gain - cost;  stability += effect
            DEFEND:          resources  -= cost;  stability += gain;
                             exposure   -= reduction
            RECOVER:         resources  -= cost;  stability += gain (larger)
            SHARE_RESOURCES: resources  -= amount;  target.resources += amount*(1-loss)
            IGNORE/others:   no resource effect

        Passive per-turn drains applied to all living districts:
            resources -= passive_resource_drain
            stability -= passive_stability_drain
        """
        cfg = self.config
        transfers: List[Tuple[int, int, float]] = []   # Phase 5 addition

        for agent_id in sorted(self._districts):
            if self._collapsed.get(agent_id, False):
                continue

            district = self._districts[agent_id]
            parsed = normalized.get(agent_id)
            if parsed is None:
                district.resources -= cfg.passive_resource_drain
                district.stability -= cfg.passive_stability_drain
                continue

            action_type = parsed["action_type"]

            if action_type == DiscreteAction.INVEST:
                district.resources += cfg.invest_resource_gain - cfg.invest_resource_cost
                district.stability += cfg.invest_stability_effect

            elif action_type == DiscreteAction.DEFEND:
                district.resources -= cfg.defend_resource_cost
                district.stability += cfg.defend_stability_gain
                district.crisis_exposure -= cfg.defend_crisis_exposure_reduction

            elif action_type == DiscreteAction.RECOVER:
                district.resources -= cfg.recover_resource_cost
                district.stability += cfg.recover_stability_gain

            elif action_type == DiscreteAction.SHARE_RESOURCES:
                target = parsed.get("target")
                amount = float(parsed.get("amount") or 0.0)
                if (
                    target is not None
                    and target != agent_id
                    and not self._collapsed.get(target, False)
                    and amount > 0.0
                ):
                    received = amount * (1.0 - cfg.transfer_loss_ratio)
                    district.resources -= amount
                    self._districts[target].resources += received
                    transfers.append((agent_id, target, amount))

            # IGNORE, PROPOSE_COALITION, REQUEST_AID, ACCEPT_COALITION,
            # REJECT_COALITION: no direct resource effect here.

            # Passive drains (applied to all living districts regardless of action)
            district.resources -= cfg.passive_resource_drain
            district.stability -= cfg.passive_stability_drain

        return transfers

    # ------------------------------------------------------------------
    # Private: crisis integration
    # ------------------------------------------------------------------

    def _update_exposures(self) -> None:
        """Refresh each district's crisis_exposure from CrisisSystem."""
        for agent_id, district in self._districts.items():
            if not self._collapsed[agent_id]:
                district.crisis_exposure = self._crisis.compute_exposure(agent_id)

    def _apply_coalition_damping(self) -> None:
        """
        Reduce crisis_exposure for coalition members.

        exposure *= (1 - coalition_exposure_damping)
        Applied after update_exposures(), before apply_crisis_effects().
        """
        damping = self.config.coalition_exposure_damping
        for agent_id, district in self._districts.items():
            if self._collapsed[agent_id]:
                continue
            if self._coalition.get_coalition(agent_id) is not None:
                district.crisis_exposure = float(
                    np.clip(district.crisis_exposure * (1.0 - damping), 0.0, 1.0)
                )

    def _apply_crisis_effects(self) -> None:
        """Drain resources and stability proportional to crisis_exposure."""
        cfg = self.config
        for agent_id, district in self._districts.items():
            if self._collapsed[agent_id]:
                continue
            district.resources -= district.crisis_exposure * cfg.exposure_resource_drain
            district.stability -= district.crisis_exposure * cfg.exposure_stability_drain

    def _clip_all(self) -> None:
        """Clip all district state values to [0.0, 1.0]."""
        for district in self._districts.values():
            district.clip_values()

    # ------------------------------------------------------------------
    # Private: collapse & terminal
    # ------------------------------------------------------------------

    def _update_collapse_status(self) -> Set[AgentID]:
        """Mark newly collapsed districts; return set of IDs."""
        newly_collapsed: Set[AgentID] = set()
        for agent_id, district in self._districts.items():
            if not self._collapsed[agent_id]:
                if district.stability <= self.config.stability_threshold:
                    district.stability = 0.0
                    self._collapsed[agent_id] = True
                    newly_collapsed.add(agent_id)
                    # Auto-leave coalition on collapse
                    self._coalition.leave(agent_id)
        return newly_collapsed

    def _check_terminal(self) -> Tuple[bool, bool]:
        """Returns (done, truncated)."""
        all_collapsed = all(self._collapsed.values())
        turn_limit = self._turn >= self.config.max_turns
        done = all_collapsed
        truncated = turn_limit and not all_collapsed
        return done, truncated

    # ------------------------------------------------------------------
    # Private: rewards (Phase 4 — RewardEngine)
    # ------------------------------------------------------------------

    def _run_reward_engine(
        self,
        prev_stability:  Dict[AgentID, float],
        newly_collapsed: Set[AgentID],
        mask_violated:   Set[AgentID],
        prev_exposure:   Dict[AgentID, float],
        prev_avg_trust:  Dict[AgentID, float],
    ) -> Tuple["RewardDict", Dict[AgentID, dict]]:
        """
        Delegate per-agent reward computation to RewardEngine (Phase 4/6).

        Returns:
            (rewards, reward_breakdowns)
            rewards:           Dict[AgentID, float]  — what env.step() returns
            reward_breakdowns: Dict[AgentID, dict]   — added to info for logging
        """
        trust_mat = self._trust.as_matrix()
        agents    = list(self._districts.keys())

        rewards, breakdowns = self._reward_engine.compute_all(
            agents=agents,
            newly_collapsed=newly_collapsed,
            collapsed=self._collapsed,
            prev_stability=prev_stability,
            curr_stability={i: d.stability        for i, d in self._districts.items()},
            prev_exposure=prev_exposure,
            curr_exposure={i: d.crisis_exposure   for i, d in self._districts.items()},
            prev_avg_trust=prev_avg_trust,
            coalition_sizes={
                i: self._coalition.coalition_size(i) for i in agents
            },
            trust_matrix=trust_mat,
            mask_violated=mask_violated,
            pending_outgoing={
                i: len(self._negotiation.pending_from(i)) for i in agents
            },
        )
        return rewards, {i: bd.to_dict() for i, bd in breakdowns.items()}

    # ------------------------------------------------------------------
    # Private: trust
    # ------------------------------------------------------------------

    def _update_trust(
        self,
        accept_events: List[Tuple[int, int]],
        reject_events: List[Tuple[int, int]],
    ) -> None:
        """
        Apply trust updates from this turn's accept/reject events.

        accept_events: List of (proposer, acceptor) — bilateral trust increase.
        reject_events: List of (proposer, rejector) — rejector trusts proposer less.
        """
        for proposer, acceptor in accept_events:
            self._trust.update_accept(proposer, acceptor)
        for proposer, rejector in reject_events:
            self._trust.update_reject(rejector, proposer)

    # ------------------------------------------------------------------
    # Private: observations
    # ------------------------------------------------------------------

    def _get_obs(self) -> MultiAgentObs:
        """Build per-agent observation dicts via ObservationBuilder."""
        return {
            agent_id: self._build_agent_obs(agent_id)
            for agent_id in self._districts
        }

    def _build_agent_obs(self, agent_id: AgentID) -> ObsDict:
        """Delegate to ObservationBuilder.build() with full Phase 3 context."""
        coalition_proposers = self._negotiation.coalition_proposers_for(agent_id)
        active_req = self._negotiation.active_requesters()
        return self._obs_builder.build(
            agent_id=agent_id,
            districts=self._districts,
            crisis=self._crisis,
            turn=self._turn,
            max_turns=self.config.max_turns,
            collapsed=self._collapsed,
            pending_proposals=coalition_proposers,
            active_request_agents=active_req,
            coalition_membership=self._coalition.memberships,
            trust_matrix=self._trust.as_matrix(),
        )

    # ------------------------------------------------------------------
    # Private: info
    # ------------------------------------------------------------------

    def _build_info(
        self,
        normalized: Dict[AgentID, ParsedAction],
        mask_violated: Optional[Set[AgentID]] = None,
        reward_breakdowns: Optional[Dict[AgentID, dict]] = None,
    ) -> Dict[str, Any]:
        """Build diagnostic info dict including Phase 4 reward breakdowns."""
        mask_violated = mask_violated or set()
        return {
            "turn":             self._turn,
            "crisis":           self._crisis.to_dict(),
            "districts":        {i: d.to_dict() for i, d in self._districts.items()},
            "collapsed":        dict(self._collapsed),
            "actions_taken":    {
                i: a["action_type"].name for i, a in normalized.items()
            },
            "mask_violations":  sorted(mask_violated),
            "coalition":        self._coalition.to_dict(),
            "negotiation":      self._negotiation.to_dict(),
            "trust":            self._trust.to_dict(),
            "reward_breakdown": reward_breakdowns or {},
        }


    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DistrictAccordEnv("
            f"districts={self.config.num_districts}, "
            f"turn={self._turn}/{self.config.max_turns}, "
            f"crisis={self._crisis.crisis_level:.3f}, "
            f"done={self._done})"
        )
