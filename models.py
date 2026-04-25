"""
models.py — Pydantic models for District Accord OpenEnv API.

Defines the Action, Observation, and State schemas used by the
client and server for serialization/deserialization.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------
class Action(BaseModel):
    """
    Action submitted to the environment for one step.

    actions: mapping from agent_id (as string) to action string.
        Valid action strings: invest, defend, ignore, recover,
        request_aid, share, propose, accept, reject.
        Structured format: "<action>:<key>=<value>,..."
        e.g. "share:target=2,amount=0.1", "propose:target=1"
    """

    actions: Dict[str, str] = Field(
        ...,
        description="Mapping of agent_id -> action string for each agent.",
        examples=[{
            "0": "invest",
            "1": "defend",
            "2": "share:target=0,amount=0.1",
        }],
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
class AgentObservation(BaseModel):
    """Observation for a single agent."""

    self_obs: List[float] = Field(
        ..., alias="self",
        description="[resources, stability, crisis_exposure, stability_delta]",
    )
    others: List[List[float]] = Field(
        ...,
        description="(N-1, 4) peer view: [resources, stability, trust, coalition_flag]",
    )
    crisis: List[float] = Field(
        ..., description="[crisis_level, normalized_tier]",
    )
    turn: List[float] = Field(
        ..., description="[progress, remaining]",
    )
    action_mask: List[float] = Field(
        ..., description="(9,) binary validity mask for each action type",
    )
    flat: List[float] = Field(
        ..., description="Concatenated flat vector (4N+4,)",
    )

    model_config = {"populate_by_name": True}


class Observation(BaseModel):
    """Full multi-agent observation returned by reset() and step()."""

    obs: Dict[str, Any] = Field(
        ..., description="Mapping of agent_id -> AgentObservation dict",
    )
    info: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional info from the environment",
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class DistrictSnapshot(BaseModel):
    """Snapshot of a single district's state."""

    district_id: int
    resources: float
    stability: float
    crisis_exposure: float


class State(BaseModel):
    """Full environment state returned by /state."""

    turn: int = Field(..., description="Current turn index")
    districts: Dict[str, Any] = Field(
        ..., description="Mapping of agent_id -> DistrictSnapshot",
    )
    done: bool = Field(..., description="Whether the episode has ended")
    crisis: Dict[str, Any] = Field(
        ..., description="Crisis system state",
    )
