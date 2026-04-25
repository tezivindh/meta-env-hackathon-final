"""
district_accord/utils/config.py
================================
Centralized, immutable configuration for DistrictAccordEnv.

All numeric hyperparameters live here.  Nothing is hard-coded in other modules.
Phase 1 defaults are provided; later phases extend the dataclass without
breaking existing code (new fields have defaults).

Usage:
    from district_accord.utils.config import EnvConfig

    cfg = EnvConfig()                        # Phase 1 defaults
    cfg = EnvConfig(num_districts=12, max_turns=100)  # Full production
    cfg = EnvConfig.phase1()                 # Named constructor → same as defaults
    cfg = EnvConfig.production()             # Named constructor → full 12-agent env
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EnvConfig:
    """
    Frozen configuration dataclass.  Frozen = no mutation after construction,
    which is critical when STATE_AUTHORITY = "centralized_env" — we never want
    a policy or subsystem sneaking a config change mid-episode.

    Naming convention:
        <subsystem>_<parameter>  (e.g. crisis_drift, invest_resource_gain)
    """

    # ------------------------------------------------------------------
    # Agent population
    # ------------------------------------------------------------------
    num_districts: int = 12
    """
    Phase 1: 2.  Phase 6: 12.
    Controls all loops and space shapes — changing this is the primary
    scaling knob.
    """

    # ------------------------------------------------------------------
    # Episode
    # ------------------------------------------------------------------
    max_turns: int = 100
    """Phase 1: 20.  Phase 6: 100."""

    # ------------------------------------------------------------------
    # Crisis system
    # ------------------------------------------------------------------
    crisis_init_mean: float = 0.15
    """Mean of the Gaussian used to sample the initial crisis level."""

    crisis_init_std: float = 0.05
    """Std-dev of the initial crisis level distribution."""

    crisis_drift: float = 0.02
    """
    Deterministic upward pressure applied to crisis_level each turn.
    Models the baseline tendency for situations to deteriorate without
    active intervention.
    """

    crisis_noise_std: float = 0.03
    """
    Std-dev of the Gaussian noise added to crisis each turn.
    Keeps the environment stochastic and prevents trivially deterministic
    policies.
    """

    # Phase 3+: coalition stability damps crisis drift.
    # Not used in Phase 1 but present so CrisisSystem.step() can accept it.
    crisis_coalition_damping: float = 0.05
    """
    [Phase 3+] Multiplied by average coalition stability and subtracted
    from drift.  Zero effect in Phase 1 since there are no coalitions.
    """

    # ------------------------------------------------------------------
    # District initial state
    # ------------------------------------------------------------------
    resource_init_mean: float = 0.60
    resource_init_std: float = 0.10
    stability_init_mean: float = 0.70
    stability_init_std: float = 0.10

    # ------------------------------------------------------------------
    # Action effects  (INVEST)
    # ------------------------------------------------------------------
    invest_resource_cost: float = 0.05
    """Resource spent to execute an INVEST action."""

    invest_resource_gain: float = 0.10
    """Resource gained from an INVEST action (gross; net = gain - cost)."""

    invest_stability_effect: float = 0.02
    """Small stability improvement from investing (economic confidence)."""

    # ------------------------------------------------------------------
    # Action effects  (DEFEND)
    # ------------------------------------------------------------------
    defend_resource_cost: float = 0.05
    """Resource spent to execute a DEFEND action."""

    defend_stability_gain: float = 0.10
    """Stability gained from a DEFEND action."""

    defend_crisis_exposure_reduction: float = 0.05
    """
    Immediate reduction in crisis_exposure from defending.
    Models active crisis mitigation (building defenses, emergency services).
    """

    # ------------------------------------------------------------------
    # Passive per-turn drains (applied to all living districts)
    # ------------------------------------------------------------------
    passive_resource_drain: float = 0.02
    """Base resource consumption per turn (upkeep, entropy)."""

    passive_stability_drain: float = 0.01
    """Base stability erosion per turn without active maintenance."""

    # ------------------------------------------------------------------
    # Crisis exposure effects
    # ------------------------------------------------------------------
    exposure_resource_drain: float = 0.03
    """Resource lost per unit of crisis_exposure per turn."""

    exposure_stability_drain: float = 0.02
    """Stability lost per unit of crisis_exposure per turn."""

    # ------------------------------------------------------------------
    # Survival reward (Phase 1 only reward signal)
    # ------------------------------------------------------------------
    survival_reward: float = 1.0
    """
    Reward granted each turn that a district remains alive (not collapsed).
    This is the only reward signal in Phase 1.  Phase 4 introduces the
    full multi-signal RewardEngine.
    """

    collapse_penalty: float = -10.0
    """
    One-time penalty applied in the turn a district collapses.
    Negative incentive to avoid losing stability entirely.
    """

    stability_threshold: float = 0.05
    """
    Stability at or below which a district is considered collapsed.
    Using a small positive value (not 0.0) avoids floating-point edge cases.
    """

    # ------------------------------------------------------------------
    # Observation settings
    # ------------------------------------------------------------------
    observation_mode: str = "dict"
    """
    "dict": structured dict of named np.ndarrays (OBSERVATION_MODE locked).
    Future: "flat" for baseline RL algorithms.
    """

    flatten_observation: bool = True
    """
    If True, also include a "flat" key in each agent's ObsDict containing
    the concatenated flat vector.  Allows both LLM-style (dict) and
    baseline (flat array) usage from a single env.step() call.
    """

    # ------------------------------------------------------------------
    # Action + parser settings
    # ------------------------------------------------------------------
    action_format: str = "structured_text"
    """ACTION_FORMAT lock: actions originate as text strings."""

    action_parser: str = "external"
    """
    ACTION_PARSER lock: env.step() receives pre-parsed DiscreteAction enums.
    Text → DiscreteAction conversion happens in ActionParser (spaces/).
    """

    # ------------------------------------------------------------------
    # Coalition / voting (Phase 3+)
    # ------------------------------------------------------------------
    voting_mode: str = "deterministic"
    """
    "deterministic": coalition join/leave decisions resolve instantly.
    Phase 3 will implement multi-turn async voting when needed.
    """

    max_coalitions: int = 4
    """Maximum concurrent coalitions.  Not enforced in Phase 1."""

    # ------------------------------------------------------------------
    # State authority
    # ------------------------------------------------------------------
    state_authority: str = "centralized_env"
    """
    "centralized_env": DistrictAccordEnv is the single source of truth for
    all district states.  No agent owns or mutates its own state object.
    """

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    seed: Optional[int] = None
    """
    RNG seed.  None = non-deterministic.  Set for reproducible episodes.
    Passed to np.random.default_rng().
    """

    # ------------------------------------------------------------------
    # Phase 2 additions
    # ------------------------------------------------------------------

    # Action effects — RECOVER
    recover_resource_cost: float = 0.08
    """Resources spent by a RECOVER action."""

    recover_stability_gain: float = 0.12
    """Stability gained from a RECOVER action (larger than DEFEND to justify
    the higher cost; intended as an emergency option)."""

    # Observation noise
    obs_neighbor_noise_std: float = 0.05
    """
    Std-dev of Gaussian noise added to the 'others' observation rows.
    Models realistic information asymmetry: agents observe peers with error.
    Set to 0.0 for fully observable (useful for debugging).
    """

    # Crisis — volatility shock
    crisis_shock_prob: float = 0.05
    """Probability (per turn) of a crisis volatility shock event."""

    crisis_shock_magnitude: float = 0.15
    """Maximum size of a volatility shock (sampled uniformly in
    [magnitude/2, magnitude] when triggered)."""

    crisis_district_scale: float = 0.005
    """Additional crisis drift per district beyond the baseline of 2.
    With 4 districts: +0.01/turn extra drift.
    With 12 districts (Phase 6): +0.05/turn extra drift."""

    # ------------------------------------------------------------------
    # Dynamic action mask thresholds (Phase 2)
    # ------------------------------------------------------------------

    aid_request_stability_threshold: float = 0.4
    """
    Maximum stability at which REQUEST_AID becomes valid.
    Below this threshold an agent is considered distressed enough to
    legitimately request coalition aid.
    """

    min_share_threshold: float = 0.10
    """
    Minimum resource level required to SHARE_RESOURCES.
    Prevents sharing when already resource-poor.
    """

    mask_violation_penalty: float = -0.5
    """
    Reward penalty applied when an agent attempts a masked (invalid) action.
    The action is silently converted to IGNORE so training does not crash.
    Must be negative and less severe than collapse_penalty to create a proper
    signal hierarchy: collapse_penalty < mask_violation_penalty < 0 < survival_reward.
    """

    # ------------------------------------------------------------------
    # Phase 3 additions (Coalition, Negotiation & Trust)
    # ------------------------------------------------------------------

    # Trust network initialisation
    trust_init_mean: float = 0.50
    """Mean of initial trust values (all pairs start near neutral-positive)."""

    trust_init_std: float = 0.10
    """Std-dev of initial trust value distribution."""

    trust_noise_std: float = 0.05
    """Gaussian noise added to trust values in the 'others' observation column."""

    # Trust update rates
    trust_accept_bonus: float = 0.10
    """Bilateral trust increase when a coalition proposal is accepted."""

    trust_reject_penalty: float = 0.05
    """Unilateral trust decrease (rejector → proposer) on rejection."""

    trust_betrayal_penalty: float = 0.20
    """Trust decrease (victim → betrayer) on a betrayal event (Phase 4+)."""

    trust_decay: float = 0.99
    """Per-turn trust decay multiplier.  trust[i][j] *= trust_decay each turn."""

    # Proposal lifecycle constraints (anti-spam)
    proposal_cost: float = 0.04
    """
    Resource cost deducted from proposer when a proposal is successfully created.
    Prevents spam: each PROPOSE_COALITION or REQUEST_AID costs resources.
    """

    max_pending_proposals: int = 2
    """
    Maximum number of outgoing proposals an agent may have pending simultaneously.
    Also applies to incoming: a target with ≥ max_pending proposals incoming rejects new ones.
    """

    proposal_ttl: int = 3
    """
    Turns before an unanswered proposal auto-expires.
    After TTL turns the target can no longer accept/reject it.
    """

    proposal_cooldown: int = 4
    """
    Turns the proposer must wait before re-proposing to the same target.
    Cooldown starts when the proposal is created (prevents juggling proposals).
    Increased from 2 to 4 (Phase 4 spam-fix diagnostic result).
    """

    # Resource transfer
    transfer_loss_ratio: float = 0.10
    """
    Fraction of resources lost in transit during SHARE_RESOURCES.
    Receiver gets amount * (1 - transfer_loss_ratio).
    """

    aid_transfer_amount: float = 0.10
    """
    Fixed resource amount transferred when a REQUEST_AID is fulfilled.
    Phase 3: the target (recipient-of-request) decides whether to fulfil
    by issuing SHARE_RESOURCES toward the requester.
    """

    # Coalition membership benefits
    max_coalition_size: int = 12
    """Hard cap on members per coalition (defaults to unconstrained for N≤12)."""

    coalition_exposure_damping: float = 0.15
    """
    Fractional reduction in crisis_exposure for coalition members.
    Applied after update_exposures() each turn:
        exposure *= (1 - coalition_exposure_damping)
    """

    coalition_stability_bonus: float = 0.5
    """
    [Phase 3 legacy — not used by Phase 4 RewardEngine]
    Per-coalition-peer survival reward bonus (replaced by reward_cooperation_per_peer).
    Kept for backward-compat with any code that reads this field.
    """

    # ------------------------------------------------------------------
    # Phase 4 additions — Reward Engine weights
    # ------------------------------------------------------------------

    reward_stability_weight: float = 1.0
    """
    Weight applied to the per-turn stability delta signal.
    delta = reward_stability_weight * (stability_now - stability_prev)
    Clipped internally to [-0.5, +0.5] to prevent spike dominance.
    """

    reward_crisis_weight: float = 1.0
    """
    Weight applied to the crisis mitigation penalty.
    penalty = -reward_crisis_weight * crisis_exposure   (exposure ∈ [0, 1])
    """

    reward_cooperation_per_peer: float = 0.05
    """
    Cooperation bonus per coalition peer per turn.
    Total capped at 0.15 regardless of coalition size.
    Small by design: coalition should not dominate the signal.
    """

    reward_trust_alignment: float = 0.02
    """
    Weight for the trust alignment component.
    signal = reward_trust_alignment * avg_positive_trust_toward_others
    Capped at 0.05 total.  Light signal only.
    """

    reward_spam_penalty: float = 0.0
    """
    Per-excess-proposal penalty.  Default 0.0 (off).
    Activates only when pending_outgoing > max_pending_proposals.
    Kept off by default since proposal_cost already deters spam.
    """

    # ------------------------------------------------------------------
    # Named constructors
    # ------------------------------------------------------------------

    @classmethod
    def phase1(cls) -> "EnvConfig":
        """Return the Phase 1 minimal config (2 districts, 20 turns)."""
        return cls(num_districts=2, max_turns=20)

    @classmethod
    def production(cls) -> "EnvConfig":
        """Return the full production config (12 districts, 100 turns)."""
        return cls(num_districts=12, max_turns=100)
