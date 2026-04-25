"""
district_accord/core/crisis.py
================================
Phase 1 Crisis System: single scalar crisis level with stochastic drift.

Crisis drives external pressure on all districts uniformly in Phase 1.
Phase 2 will extend this with:
  - 5-tier tier effects (already classified here via CrisisTier)
  - Per-district neighbourhood propagation
  - Coalition stability as a damping feedback signal

Update rule (Phase 1):
    C_{t+1} = clip(C_t + drift + σ·ε,  0, 1)
    where ε ~ N(0, 1)

Every call to step() appends to an internal history list enabling Phase 5's
StateTracker to audit the full crisis trajectory without storing extra state.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np

from district_accord.utils.config import EnvConfig
from district_accord.utils.types import CrisisTier, crisis_level_to_tier

if TYPE_CHECKING:
    # Avoids circular import at runtime; only used for type hints.
    from district_accord.core.district import DistrictState


class CrisisSystem:
    """
    Manages the global crisis level for one episode.

    Lifecycle:
        crisis = CrisisSystem(config, rng)
        crisis.reset()                        # start of episode
        for each turn:
            crisis.step(district_states)      # advance one turn
            exposure = crisis.compute_exposure(district_id)

    Attributes:
        crisis_level: float  Current scalar level in [0.0, 1.0].
        tier:         CrisisTier  Current tier classification (read-only property).
        history:      List[float] Full trajectory since last reset().
    """

    def __init__(
        self,
        config: EnvConfig,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self._rng = rng
        self.crisis_level: float = self.config.crisis_init_mean
        self._history: List[float] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Initialise crisis level at the start of a new episode.

        Samples from N(crisis_init_mean, crisis_init_std) and clips to [0, 1].
        Seeds history with the initial value.
        """
        raw = self._rng.normal(
            self.config.crisis_init_mean,
            self.config.crisis_init_std,
        )
        self.crisis_level = float(np.clip(raw, 0.0, 1.0))
        self._history = [self.crisis_level]

    def step(
        self,
        district_states: Optional[List["DistrictState"]] = None,
    ) -> float:
        """
        Advance the crisis level by one turn.

        Phase 2 update rule:
            C_{t+1} = clip(
                C_t
                + drift
                + district_pressure     ← NEW: scales with num districts
                - coalition_damping     ← Phase 3+
                + σ·ε                   ← Gaussian noise
                + shock                 ← NEW: rare volatility spike
            , 0, 1)

        Args:
            district_states: Passed by the env for forward-compatibility.
                             Phase 2 uses len() for district_pressure.
                             Phase 3+ uses coalition stability for damping.

        Returns:
            Updated crisis_level in [0.0, 1.0].
        """
        # Phase 3+: coalition stability damps crisis drift.
        coalition_damping = 0.0

        # Phase 2: additional drift from having more districts to manage.
        # Baseline of 2 districts has zero extra pressure.
        num_districts = len(district_states) if district_states else 2
        district_pressure = max(
            0.0,
            (num_districts - 2) * self.config.crisis_district_scale,
        )

        # Stochastic base noise.
        noise = self._rng.normal(0.0, self.config.crisis_noise_std)

        # Phase 2: volatility shock — rare large upward jump.
        shock = 0.0
        if self._rng.uniform() < self.config.crisis_shock_prob:
            shock = float(
                self._rng.uniform(
                    self.config.crisis_shock_magnitude * 0.5,
                    self.config.crisis_shock_magnitude,
                )
            )

        new_level = (
            self.crisis_level
            + self.config.crisis_drift
            + district_pressure
            - coalition_damping
            + noise
            + shock
        )
        self.crisis_level = float(np.clip(new_level, 0.0, 1.0))
        self._history.append(self.crisis_level)
        return self.crisis_level


    # ------------------------------------------------------------------
    # Per-district exposure
    # ------------------------------------------------------------------

    def compute_exposure(self, district_id: int) -> float:
        """
        Compute crisis exposure for a specific district.

        Phase 1:  All districts share the same global crisis level.
                  district_id is accepted but ignored.

        Phase 2+: Will incorporate neighbourhood topology and district-
                  specific vulnerability scores.

        Args:
            district_id: The targeted district's ID.

        Returns:
            Exposure value in [0.0, 1.0].
        """
        _ = district_id  # Unused in Phase 1; explicit suppression.
        return self.crisis_level

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tier(self) -> CrisisTier:
        """Current crisis tier classification (derived from crisis_level)."""
        return crisis_level_to_tier(self.crisis_level)

    @property
    def history(self) -> List[float]:
        """
        Full crisis level trajectory since last reset().
        Index 0 = initial value (after reset), index N = after N steps.
        """
        return list(self._history)  # defensive copy

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_vector(self) -> np.ndarray:
        """
        Serialise crisis state to a float32 array.

        Shape: (1,) in Phase 1.
        Phase 2+ will extend to (6,): [level, tier_one_hot × 5].
        """
        return np.array([self.crisis_level], dtype=np.float32)

    def to_dict(self) -> dict:
        """Serialise to plain dict (for info dicts and logging)."""
        return {
            "crisis_level": round(self.crisis_level, 6),
            "tier": self.tier.name,
            "tier_value": int(self.tier),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CrisisSystem("
            f"level={self.crisis_level:.3f}, "
            f"tier={self.tier.name})"
        )
