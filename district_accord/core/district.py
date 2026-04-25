"""
district_accord/core/district.py
=================================
Mutable state model for a single district agent.

Design principles:
- DistrictState is a plain data container; it does NOT make decisions.
- The environment (STATE_AUTHORITY = "centralized_env") owns all mutation.
- All continuous values are normalised to [0.0, 1.0].
- clip_values() must be called by the env after every mutation batch.

Scalability notes (Phase 1 → Phase 6):
- Phase 3: add coalition_id, influence, trust_scores dict.
- Phase 4: add policy_vector (8-dim continuous).
- Phase 5: add history deque for state_tracker integration.
- Phase 6: frozen snapshot variant for self-play pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from district_accord.utils.types import AgentID


@dataclass
class DistrictState:
    """
    Mutable per-turn state for one district.

    Attributes:
        district_id:      Unique integer ID in [0, num_districts).
        resources:        Normalised resource pool in [0.0, 1.0].
                          Represents economic capacity, supply stores, etc.
        stability:        Normalised internal stability in [0.0, 1.0].
                          At or below stability_threshold → district collapses.
        crisis_exposure:  Accumulated crisis pressure from the global crisis
                          system in [0.0, 1.0].  Updated by CrisisSystem each
                          turn; drives passive resource/stability drains.

    Phase 3+ additions (not present in Phase 1):
        coalition_id:     Coalition membership (None = independent).
        influence:        Political influence score [0.0, 1.0].
        policy_vector:    8-dim continuous preference vector.
        trust_scores:     Per-peer trust in [0.0, 1.0].
    """

    district_id: AgentID
    resources: float
    stability: float
    crisis_exposure: float = 0.0

    # ------------------------------------------------------------------
    # Phase 3+ stubs — defined but not populated in Phase 1.
    # Kept here so DistrictState can be passed to Phase 3 code without
    # structural changes.
    # ------------------------------------------------------------------
    coalition_id: Optional[int] = field(default=None, repr=False)
    # influence: float = field(default=0.5, repr=False)
    # policy_vector: np.ndarray = field(
    #     default_factory=lambda: np.zeros(8, dtype=np.float32), repr=False
    # )
    # trust_scores: Dict[AgentID, float] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_collapsed(self) -> bool:
        """
        True when the district's stability has dropped to zero.

        The env sets stability exactly to 0.0 on collapse; this property
        checks that exact value so it can be used safely after clip_values().
        """
        return self.stability <= 0.0

    # ------------------------------------------------------------------
    # Mutation helpers  (called by env; districts do NOT mutate themselves)
    # ------------------------------------------------------------------

    def clip_values(self) -> None:
        """
        Enforce [0.0, 1.0] bounds on all continuous fields.

        Must be called by the environment after every mutation batch to
        prevent values from drifting out of range due to cumulative additions.
        """
        self.resources = float(np.clip(self.resources, 0.0, 1.0))
        self.stability = float(np.clip(self.stability, 0.0, 1.0))
        self.crisis_exposure = float(np.clip(self.crisis_exposure, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_vector(self) -> np.ndarray:
        """
        Serialise own state to a float32 numpy array.

        Shape: (3,) in Phase 1.
        Index mapping:
            0 → resources
            1 → stability
            2 → crisis_exposure

        Phase 3 will extend this to shape (N,) when influence, coalition
        membership, and policy_vector are added.
        """
        return np.array(
            [self.resources, self.stability, self.crisis_exposure],
            dtype=np.float32,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to plain dict (for logging, info dicts, and debugging)."""
        return {
            "district_id": self.district_id,
            "resources": round(self.resources, 6),
            "stability": round(self.stability, 6),
            "crisis_exposure": round(self.crisis_exposure, 6),
            "is_collapsed": self.is_collapsed,
            "coalition_id": self.coalition_id,
        }

    def __repr__(self) -> str:  # pragma: no cover
        status = "COLLAPSED" if self.is_collapsed else "alive"
        return (
            f"District({self.district_id}) "
            f"res={self.resources:.3f} "
            f"stab={self.stability:.3f} "
            f"exp={self.crisis_exposure:.3f} "
            f"[{status}]"
        )
