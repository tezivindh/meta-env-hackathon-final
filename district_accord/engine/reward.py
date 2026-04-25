"""
district_accord/engine/reward.py
==================================
RewardEngine — Phase 4 multi-signal, anti-exploit reward computation.

Replaces env._compute_rewards() with a modular, deterministic, stateless
function suitable for TRL + GRPO training.

Reward components (per agent per turn):
    A. Survival          +survival_reward  per turn alive           (~+1.0)
    B. Stability delta   +weight * (stability_now - stability_prev) (~±0.05)
    C. Crisis mitigation -weight * (exposure_now - exposure_prev)   (delta-based, Phase 6)
    D. Cooperation       +per_peer * (coalition_size − 1)           (≤ +0.15)
    E. Trust alignment   +weight * (avg_pos_now - avg_pos_prev)     (delta-based, Phase 6)
    F. Mask penalty      config.mask_violation_penalty (if violated)(~−0.50)
    G. Spam penalty      −weight × excess_pending  (off by default) (  0.00)
    H. Collapse penalty  config.collapse_penalty (one-time)         (~−10.0)

Scale target:
    Typical alive agent per-step: ∈ [0.4, 1.2]
    Collapse turn:                 collapse_penalty (−10.0)
    No component dominates.

Anti-exploit guarantees:
    - D (cooperation) hard-capped at +0.15 total.
    - E (trust)       hard-capped at +0.05 total.
    - G (spam)        defaults to 0.0 weight — off unless enabled.
    - Proposal cost is a resource drain only; NOT double-penalised here.

TRL/GRPO compatibility:
    - Deterministic: same envstate → same reward, every time.
    - Stateless: RewardEngine holds no mutable episode state.
    - Modular: each component is an Override-friendly method.
    - Transparency: RewardBreakdown.to_dict() exposes all parts to logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from district_accord.utils.config import EnvConfig
from district_accord.utils.types import AgentID


# ---------------------------------------------------------------------------
# RewardBreakdown — transparent itemised record
# ---------------------------------------------------------------------------

@dataclass
class RewardBreakdown:
    """
    Itemised reward record for one agent on one turn.

    Attributes mirror the lettered components in the module docstring.
    `total` is a property that sums all components — it is NOT stored separately
    to avoid any risk of total ≠ sum(parts).

    Usage:
        r, bd = engine.compute(agent_id=0, ...)
        assert r == bd.total
        print(bd.to_dict())   # → {"survival": 1.0, "stability_delta": 0.03, ...}
    """

    agent_id: int
    survival:          float = 0.0
    stability_delta:   float = 0.0
    crisis_mitigation: float = 0.0
    cooperation:       float = 0.0
    trust_alignment:   float = 0.0
    mask_penalty:      float = 0.0
    spam_penalty:      float = 0.0
    collapse_penalty:  float = 0.0

    @property
    def total(self) -> float:
        """Sum of all components — the value returned by RewardEngine.compute()."""
        return (
            self.survival
            + self.stability_delta
            + self.crisis_mitigation
            + self.cooperation
            + self.trust_alignment
            + self.mask_penalty
            + self.spam_penalty
            + self.collapse_penalty
        )

    def to_dict(self) -> Dict[str, float]:
        """Serialisable flat dict for info logging / debugging."""
        return {
            "survival":          round(self.survival, 4),
            "stability_delta":   round(self.stability_delta, 4),
            "crisis_mitigation": round(self.crisis_mitigation, 4),
            "cooperation":       round(self.cooperation, 4),
            "trust_alignment":   round(self.trust_alignment, 4),
            "mask_penalty":      round(self.mask_penalty, 4),
            "spam_penalty":      round(self.spam_penalty, 4),
            "collapse_penalty":  round(self.collapse_penalty, 4),
            "total":             round(self.total, 4),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RewardBreakdown(agent={self.agent_id}, total={self.total:.4f} | "
            f"surv={self.survival:.2f} Δstab={self.stability_delta:+.4f} "
            f"cris={self.crisis_mitigation:+.4f} coop={self.cooperation:.4f} "
            f"trust={self.trust_alignment:.4f} mask={self.mask_penalty:.2f})"
        )


# ---------------------------------------------------------------------------
# RewardEngine
# ---------------------------------------------------------------------------

class RewardEngine:
    """
    Stateless, deterministic multi-signal reward computer (Phase 4).

    Design:
        - Pure function semantics: RewardEngine has no mutable state.
        - Each signal component is a separate method → ablation-friendly.
        - compute() is the single entry point per agent per turn.
        - compute_all() is a convenience wrapper for the full agent set.

    All numeric parameters are drawn from EnvConfig; nothing is hard-coded
    inside this class beyond safety caps to prevent reward hacking.
    """

    # Safety caps — hard bounds enforced AFTER weight multiplication.
    # Prevents a single signal from dominating even if weights are misconfigured.
    _COOPERATION_CAP  = 0.15   # D: max coalition bonus per turn
    _TRUST_CAP        = 0.05   # E: max trust alignment per turn
    _DELTA_CAP        = 0.50   # B: max |stability delta| per turn

    def __init__(self, config: EnvConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def compute(
        self,
        agent_id: AgentID,
        *,
        is_newly_collapsed: bool,
        is_collapsed: bool,
        prev_stability: float,
        curr_stability: float,
        prev_exposure: float,      # Phase 6: delta-based crisis signal
        curr_exposure: float,
        prev_avg_trust: float,     # Phase 6: delta-based trust signal
        coalition_size: int,
        trust_row: Dict[AgentID, float],
        mask_violated: bool,
        pending_outgoing: int,
    ) -> Tuple[float, RewardBreakdown]:
        """
        Compute total reward and component breakdown for agent_id on one step.

        Args:
            agent_id:            Agent identifier.
            is_newly_collapsed:  True if agent just hit stability ≤ threshold.
            is_collapsed:        True if agent was already dead before this step.
            prev_stability:      Stability at START of the turn (before actions).
            curr_stability:      Stability at END of the turn (after all effects).
            prev_exposure:       Crisis exposure at START of turn (Phase 6 delta).
            curr_exposure:       Crisis exposure at END of turn.
            prev_avg_trust:      Mean positive trust toward others at START (Phase 6 delta).
            coalition_size:      Total members in agent's coalition (0 = solo).
            trust_row:           Dict {other_id: trust_value ∈ [-1,1]}.
                                 Contains all OTHER agents, excludes self.
            mask_violated:       True if agent's action was masked → converted to IGNORE.
            pending_outgoing:    Number of proposals this agent has currently pending.

        Returns:
            (total_reward: float, breakdown: RewardBreakdown).
            total_reward == breakdown.total  (always).
        """
        bd = RewardBreakdown(agent_id=agent_id)

        # ── H. Collapse (one-time; overrides all other signals) ────────────
        if is_newly_collapsed:
            bd.collapse_penalty = float(self.config.collapse_penalty)
            return float(bd.total), bd

        # ── Already dead (collapsed in a prior turn) ───────────────────
        if is_collapsed:
            return 0.0, bd

        # ── A. Survival ───────────────────────────────────────────────
        bd.survival = self._survival()

        # ── B. Stability delta ────────────────────────────────────────
        bd.stability_delta = self._stability_delta(prev_stability, curr_stability)

        # ── C. Crisis mitigation (delta-based, Phase 6) ─────────────────
        bd.crisis_mitigation = self._crisis_mitigation(prev_exposure, curr_exposure)

        # ── D. Cooperation ──────────────────────────────────────────
        bd.cooperation = self._cooperation(coalition_size)

        # ── E. Trust alignment (delta-based, Phase 6) ──────────────────
        bd.trust_alignment = self._trust_alignment(trust_row, prev_avg_trust)

        # ── F. Mask penalty ─────────────────────────────────────────
        if mask_violated:
            bd.mask_penalty = self._mask_penalty()

        # ── G. Spam penalty (off by default; reward_spam_penalty=0.0) ────
        bd.spam_penalty = self._spam_penalty(pending_outgoing)

        return float(bd.total), bd

    def compute_all(
        self,
        agents: List[AgentID],
        *,
        newly_collapsed: Set[AgentID],
        collapsed: Dict[AgentID, bool],
        prev_stability:  Dict[AgentID, float],
        curr_stability:  Dict[AgentID, float],
        prev_exposure:   Dict[AgentID, float],   # Phase 6 delta
        curr_exposure:   Dict[AgentID, float],
        prev_avg_trust:  Dict[AgentID, float],   # Phase 6 delta
        coalition_sizes: Dict[AgentID, int],
        trust_matrix:    Dict[AgentID, Dict[AgentID, float]],
        mask_violated:   Set[AgentID],
        pending_outgoing: Dict[AgentID, int],
    ) -> Tuple[Dict[AgentID, float], Dict[AgentID, RewardBreakdown]]:
        """
        Compute rewards for all agents in deterministic (sorted) order.

        Returns:
            rewards:    Dict[AgentID, float]
            breakdowns: Dict[AgentID, RewardBreakdown]
        """
        rewards:    Dict[AgentID, float]          = {}
        breakdowns: Dict[AgentID, RewardBreakdown] = {}

        for agent_id in sorted(agents):   # deterministic order
            trust_row = {
                k: v
                for k, v in trust_matrix.get(agent_id, {}).items()
                if k != agent_id
            }
            r, bd = self.compute(
                agent_id=agent_id,
                is_newly_collapsed=(agent_id in newly_collapsed),
                is_collapsed=collapsed.get(agent_id, False),
                prev_stability=prev_stability.get(
                    agent_id, curr_stability.get(agent_id, 0.5)
                ),
                curr_stability=curr_stability.get(agent_id, 0.0),
                prev_exposure=prev_exposure.get(agent_id, 0.0),
                curr_exposure=curr_exposure.get(agent_id, 0.0),
                prev_avg_trust=prev_avg_trust.get(agent_id, 0.0),
                coalition_size=coalition_sizes.get(agent_id, 0),
                trust_row=trust_row,
                mask_violated=(agent_id in mask_violated),
                pending_outgoing=pending_outgoing.get(agent_id, 0),
            )
            rewards[agent_id]    = r
            breakdowns[agent_id] = bd

        return rewards, breakdowns

    # ------------------------------------------------------------------
    # Component methods — override or ablate individually
    # ------------------------------------------------------------------

    def _survival(self) -> float:
        """A: Base survival reward per turn alive."""
        return float(self.config.survival_reward)

    def _stability_delta(self, prev: float, curr: float) -> float:
        """
        B: Signed, weighted stability change.

        Positive → agent improved (recovered, defended, invested).
        Negative → agent deteriorated (neglect, crisis damage).
        Clipped to ±_DELTA_CAP to prevent a single dramatic turn
        from overshadowing the cumulative signal.
        """
        w = self.config.reward_stability_weight
        raw = float(w * (curr - prev))
        return float(max(-self._DELTA_CAP, min(self._DELTA_CAP, raw)))

    def _crisis_mitigation(self, prev_exposure: float, curr_exposure: float) -> float:
        """
        C: Delta-based crisis signal (Phase 6).

        reward -= weight * (curr_exposure - prev_exposure)

        Interpretation:
            exposure increased  (+delta)  → negative reward (crisis got worse)
            exposure decreased  (-delta)  → positive reward (DEFEND paid off)
            exposure unchanged  (0 delta) → zero           (no chronic drain)

        Removes the chronic negative bias from absolute exposure, making the
        signal purely about how the agent changed its situation this turn.
        Clipped to ±_DELTA_CAP for safety.
        """
        w = self.config.reward_crisis_weight
        delta = float(curr_exposure) - float(prev_exposure)
        raw = -w * delta
        return float(max(-self._DELTA_CAP, min(self._DELTA_CAP, raw)))

    def _cooperation(self, coalition_size: int) -> float:
        """
        D: Small per-peer coalition bonus.

        per_peer * (size − 1),  hard-capped at _COOPERATION_CAP (0.15).
        Signals: "being in a coalition is better than being alone."
        Small enough that solo INVEST is still competitive if coalition
        formation is costly.
        """
        if coalition_size <= 1:
            return 0.0
        per_peer = self.config.reward_cooperation_per_peer
        raw = float(per_peer * (coalition_size - 1))
        return float(min(raw, self._COOPERATION_CAP))

    def _trust_alignment(self, trust_row: Dict[AgentID, float], prev_avg_trust: float) -> float:
        """
        E: Delta-based trust signal (Phase 6).

        signal = weight * (avg_pos_trust_now - avg_pos_trust_prev)
        Hard-capped at _TRUST_CAP (0.05).

        Rewards trust *improvement* (e.g. from accepting a coalition).
        Zero when trust is stable (prevents trust farming / passive accumulation).
        Negative delta is not double-penalised (capped at 0 below).
        """
        if not trust_row:
            return 0.0
        w = self.config.reward_trust_alignment
        curr_avg_pos = sum(max(v, 0.0) for v in trust_row.values()) / len(trust_row)
        delta = curr_avg_pos - float(prev_avg_trust)
        # Only positive improvements contribute; negative deltas give 0
        return float(min(w * max(delta, 0.0), self._TRUST_CAP))

    def _mask_penalty(self) -> float:
        """F: Penalty for attempting a masked (invalid) action."""
        return float(self.config.mask_violation_penalty)  # already negative

    def _spam_penalty(self, pending_outgoing: int) -> float:
        """
        G: Optional tiny penalty for flooding outgoing proposals.

        Off by default (reward_spam_penalty = 0.0).
        Only activates for proposals beyond max_pending_proposals.
        Proposal_cost in resources already disincentivises spam — this is
        an additional signal for trained policies that might exploit the
        attention cost of the negotiation system.
        """
        w = self.config.reward_spam_penalty
        if w == 0.0:
            return 0.0
        excess = max(0, pending_outgoing - self.config.max_pending_proposals)
        return float(-w * excess)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RewardEngine("
            f"surv={self.config.survival_reward}, "
            f"stab_w={self.config.reward_stability_weight}, "
            f"crisis_w={self.config.reward_crisis_weight}, "
            f"coop={self.config.reward_cooperation_per_peer}, "
            f"trust={self.config.reward_trust_alignment})"
        )
