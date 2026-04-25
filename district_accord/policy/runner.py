"""
district_accord/policy/runner.py
==================================
EpisodeRunner — training-ready trajectory collector.

Runs one complete episode, collecting a structured trajectory that includes:
    - Per-step observations, actions, rewards, done flags
    - Full info dict (reward_breakdown, events, state_snapshot)

The trajectory format is directly usable with TRL / GRPO training pipelines.

Optional utils:
    - save_trajectory(traj, path)   — JSON serialisation
    - load_trajectory(path)         — JSON deserialisation
    - verify_replay(env, traj)      — deterministic replay check
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from district_accord.policy.self_play import SelfPlayPolicy
from district_accord.utils.types import AgentID, DiscreteAction


# ---------------------------------------------------------------------------
# Trajectory step record
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """
    One environment step in a trajectory.

    Attributes:
        step:      Turn index (0-based).
        obs:       Per-agent observation dicts (from env.step / env.reset).
        actions:   Per-agent actions submitted this step (action_type name).
        rewards:   Per-agent float rewards.
        done:      Episode done flag.
        truncated: Episode truncated flag.
        info:      Full info dict including reward_breakdown, events, state_snapshot.
    """

    step:      int
    obs:       Dict[AgentID, dict]
    actions:   Dict[AgentID, str]           # DiscreteAction.name for serialisability
    rewards:   Dict[AgentID, float]
    done:      bool
    truncated: bool
    info:      Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable representation (JSON-safe)."""
        return {
            "step":      self.step,
            "obs":       {str(k): _serialise_obs(v) for k, v in self.obs.items()},
            "actions":   {str(k): v for k, v in self.actions.items()},
            "rewards":   {str(k): round(float(v), 6) for k, v in self.rewards.items()},
            "done":      self.done,
            "truncated": self.truncated,
            "info": {
                "reward_breakdown": {
                    str(k): v
                    for k, v in self.info.get("reward_breakdown", {}).items()
                },
                "events":         self.info.get("events", []),
                "state_snapshot": self.info.get("state_snapshot", {}),
                "turn":           self.info.get("turn"),
                "coalition":      self.info.get("coalition"),
            },
        }


def _serialise_obs(obs: dict) -> dict:
    """Convert numpy arrays in obs to lists for JSON serialisation."""
    return {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in obs.items()
    }


# ---------------------------------------------------------------------------
# EpisodeRunner
# ---------------------------------------------------------------------------

class EpisodeRunner:
    """
    Collect a full episode trajectory from env + policy.

    Usage:
        runner = EpisodeRunner()
        trajectory = runner.run_episode(env, policy, seed=42)
        save_trajectory(trajectory, "episode_0.json")
    """

    def run_episode(
        self,
        env:    "DistrictAccordEnv",   # type: ignore[name-defined]
        policy: "SelfPlayPolicy",
        seed:   Optional[int] = None,
    ) -> List[StepRecord]:
        """
        Run one complete episode.

        Args:
            env:    DistrictAccordEnv instance.  Will be reset internally.
            policy: SelfPlayPolicy (or any object with .act(obs, env)).
            seed:   Optional seed passed to env.reset().

        Returns:
            List[StepRecord] — one record per step.  Length == turns played.
        """
        obs = env.reset(seed=seed)
        trajectory: List[StepRecord] = []
        step = 0

        while True:
            actions = policy.act(obs, env)

            # Serialise action_type names before calling step
            action_names: Dict[AgentID, str] = {
                a: (
                    parsed["action_type"].name
                    if isinstance(parsed, dict)
                    else parsed.name
                    if isinstance(parsed, DiscreteAction)
                    else str(parsed)
                )
                for a, parsed in actions.items()
            }

            next_obs, rewards, done, truncated, info = env.step(actions)

            record = StepRecord(
                step=step,
                obs=obs,
                actions=action_names,
                rewards={a: float(r) for a, r in rewards.items()},
                done=done,
                truncated=truncated,
                info=info,
            )
            trajectory.append(record)

            obs = next_obs
            step += 1

            if done or truncated:
                break

        return trajectory

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def episode_summary(self, trajectory: List[StepRecord]) -> Dict[str, Any]:
        """
        Compute high-level statistics from a completed trajectory.

        Returns dict with keys:
            turns_played, total_rewards, collapses, coalition_events,
            total_events, avg_reward_per_turn_per_agent
        """
        if not trajectory:
            return {}

        total_rewards: Dict[AgentID, float] = {}
        collapses      = 0
        coalition_ev   = 0
        total_events   = 0

        for rec in trajectory:
            for a, r in rec.rewards.items():
                total_rewards[a] = total_rewards.get(a, 0.0) + r
            events = rec.info.get("events", [])
            total_events += len(events)
            collapses     += sum(1 for e in events if e["type"] == "collapse")
            coalition_ev  += sum(
                1 for e in events
                if e["type"] in ("coalition_formed", "coalition_joined")
            )

        n = len(trajectory)
        n_agents = len(total_rewards)
        avg_r = sum(total_rewards.values()) / max(n * n_agents, 1)

        return {
            "turns_played":               n,
            "total_rewards":              {str(k): round(v, 4) for k, v in total_rewards.items()},
            "collapses":                  collapses,
            "coalition_events":           coalition_ev,
            "total_events":               total_events,
            "avg_reward_per_turn_per_agent": round(avg_r, 4),
        }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_trajectory(trajectory: List[StepRecord], path: Union[str, Path]) -> None:
    """
    Serialise trajectory to JSON.

    Args:
        trajectory: List[StepRecord] from EpisodeRunner.run_episode().
        path:       Output file path (created if missing).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [rec.to_dict() for rec in trajectory]
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def load_trajectory(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load a previously saved trajectory from JSON.

    Returns list of raw dicts (not StepRecord objects).
    Use this for inspection or offline analysis.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def verify_replay(
    env:       "DistrictAccordEnv",  # type: ignore[name-defined]
    trajectory: List[StepRecord],
    policy:     "SelfPlayPolicy",
    seed:       Optional[int] = None,
) -> bool:
    """
    Re-run the episode from scratch with same policy + seed and compare.

    Returns True iff every step produces identical rewards and events,
    confirming deterministic replay.

    Note: Uses policy.mode and same seed; both env and policy must use same seed.
    """
    runner = EpisodeRunner()
    replay = runner.run_episode(env, policy, seed=seed)

    if len(replay) != len(trajectory):
        return False

    for orig, rep in zip(trajectory, replay):
        # Compare rewards
        for a in orig.rewards:
            if a not in rep.rewards:
                return False
            if abs(orig.rewards[a] - rep.rewards[a]) > 1e-9:
                return False
        # Compare event types in order
        orig_etypes = [e["type"] for e in orig.info.get("events", [])]
        rep_etypes  = [e["type"] for e in rep.info.get("events", [])]
        if orig_etypes != rep_etypes:
            return False

    return True
