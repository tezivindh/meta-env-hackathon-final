"""
district_accord/core/trust.py
==============================
TrustSystem — Phase 3 inter-district trust network.

trust[i][j] ∈ [-1.0, 1.0]
    +1.0 = maximum trust (full cooperation expected)
    -1.0 = maximum distrust (betrayal history)
     0.0 = neutral (strangers)

Update rules:
    accept → both directions += trust_accept_bonus   (cooperation signals)
    reject → rejector→proposer -= trust_reject_penalty (asymmetric: I distrust who I rejected)
    betrayal → victim→betrayer -= trust_betrayal_penalty

Decay:
    Every turn: trust[i][j] *= trust_decay   (slow forgetting toward 0)

Design:
    - Fully centralized: env owns TrustSystem, agents never mutate it.
    - Values clipped to [-1, 1] after every update.
    - trust[i][i] is always NaN (undefined; never accessed in obs).
    - The matrix is observable (with noise) via ObservationBuilder.
"""

from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

import numpy as np

from district_accord.utils.config import EnvConfig

if TYPE_CHECKING:
    pass


class TrustSystem:
    """
    Centralized trust matrix for District Accord.

    Attributes:
        _trust:  Dict[int, Dict[int, float]].  trust[i][j] = i's trust towards j.
        config:  EnvConfig (for decay / bonus / penalty values).
        _rng:    numpy RNG (for trust initialization noise).
    """

    def __init__(self, config: EnvConfig, rng: np.random.Generator) -> None:
        self.config = config
        self._rng = rng
        self._trust: Dict[int, Dict[int, float]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, num_districts: int) -> None:
        """
        Initialise trust matrix.

        trust[i][j] ~ N(trust_init_mean, trust_init_std), clipped to [-1, 1].
        trust[i][i] set to 0.0 (self-trust; never used in policy obs).
        """
        cfg = self.config
        self._trust = {}
        for i in range(num_districts):
            self._trust[i] = {}
            for j in range(num_districts):
                if i == j:
                    self._trust[i][j] = 0.0
                else:
                    val = float(self._rng.normal(cfg.trust_init_mean, cfg.trust_init_std))
                    self._trust[i][j] = float(np.clip(val, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Update rules
    # ------------------------------------------------------------------

    def update_accept(self, agent_a: int, agent_b: int) -> None:
        """
        Coalition acceptance — both parties gain trust in each other.

        Symmetric: A trusts B more, B trusts A more.
        """
        bonus = self.config.trust_accept_bonus
        self._add(agent_a, agent_b, +bonus)
        self._add(agent_b, agent_a, +bonus)

    def update_reject(self, rejector: int, proposer: int) -> None:
        """
        Proposal rejection — rejector trusts proposer slightly less.

        Asymmetric: only the rejector's view changes (they're signaling
        disinterest in collaboration with the proposer).
        """
        penalty = self.config.trust_reject_penalty
        self._add(rejector, proposer, -penalty)

    def update_betrayal(self, victim: int, betrayer: int) -> None:
        """
        Betrayal event (Phase 4+: breaking coalition agreement).

        Victim's trust of betrayer drops sharply.
        """
        penalty = self.config.trust_betrayal_penalty
        self._add(victim, betrayer, -penalty)

    # ------------------------------------------------------------------
    # Per-turn decay
    # ------------------------------------------------------------------

    def decay(self) -> None:
        """
        Apply trust decay toward 0 (slow forgetting).

        trust[i][j] *= trust_decay  for all i ≠ j.
        """
        d = self.config.trust_decay
        for i in self._trust:
            for j in self._trust[i]:
                if i != j:
                    self._trust[i][j] = float(
                        np.clip(self._trust[i][j] * d, -1.0, 1.0)
                    )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, i: int, j: int) -> float:
        """
        Safe trust accessor.

        Returns trust_init_mean if (i, j) not yet initialized.
        """
        return self._trust.get(i, {}).get(j, self.config.trust_init_mean)

    def matrix(self, agent_id: int) -> Dict[int, float]:
        """
        Return all trust values from agent_id's perspective.

        e.g. trust.matrix(0) = {1: 0.55, 2: 0.48, 3: 0.60}
        """
        return dict(self._trust.get(agent_id, {}))

    def as_matrix(self) -> Dict[int, Dict[int, float]]:
        """
        Full trust matrix (for ObservationBuilder).

        Returns a shallow copy of the internal dict.
        """
        return {i: dict(row) for i, row in self._trust.items()}

    def num_agents(self) -> int:
        """Number of agents tracked."""
        return len(self._trust)

    def to_dict(self) -> dict:
        """Serialisable snapshot for info/logging."""
        return {
            "trust_matrix": {
                str(i): {str(j): round(v, 4) for j, v in row.items()}
                for i, row in self._trust.items()
            }
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _add(self, i: int, j: int, delta: float) -> None:
        """Add delta to trust[i][j], clip result to [-1, 1]."""
        if i not in self._trust:
            self._trust[i] = {}
        current = self._trust[i].get(j, self.config.trust_init_mean)
        self._trust[i][j] = float(np.clip(current + delta, -1.0, 1.0))

    def __repr__(self) -> str:  # pragma: no cover
        n = len(self._trust)
        return f"TrustSystem(num_agents={n})"
