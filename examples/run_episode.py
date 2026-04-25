"""
examples/run_episode.py
=========================
End-to-end episode runner for District Accord (Phase 1).

Demonstrates:
  1. EnvConfig construction
  2. ActionParser usage (structured text → DiscreteAction)
  3. Gym-style reset/step loop
  4. Reading observations and info dicts
  5. Episode summary

Run from the project root:
    python examples/run_episode.py
    python examples/run_episode.py --seed 123
    python examples/run_episode.py --policy defend
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict

import numpy as np

# Ensure the package is importable when run from repo root.
sys.path.insert(0, ".")

from district_accord.env import DistrictAccordEnv
from district_accord.spaces.action_parser import ActionParser
from district_accord.utils.config import EnvConfig
from district_accord.utils.types import AgentID, DiscreteAction


# ---------------------------------------------------------------------------
# Policy functions
# ---------------------------------------------------------------------------

def policy_random(
    obs: Dict[AgentID, dict],
    num_agents: int,
    rng: np.random.Generator,
    parser: ActionParser,
) -> Dict[AgentID, DiscreteAction]:
    """Random text policy — mimics an untrained LLM."""
    choices = parser.valid_action_strings()
    raw = {i: rng.choice(choices) for i in range(num_agents)}
    return parser.parse_safe(raw)


def policy_fixed(
    obs: Dict[AgentID, dict],
    num_agents: int,
    action: str,
    parser: ActionParser,
) -> Dict[AgentID, DiscreteAction]:
    """All agents always choose the same fixed action."""
    return parser.parse({i: action for i in range(num_agents)})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(seed: int = 42, fixed_action: str | None = None) -> None:
    # ---- Config & env setup ----
    config = EnvConfig(
        num_districts=2,
        max_turns=20,
        seed=seed,
        flatten_observation=True,
    )
    env = DistrictAccordEnv(config)
    parser = ActionParser(config)
    rng = np.random.default_rng(seed)

    print("=" * 60)
    print(f"  District Accord — Phase 1 Episode")
    print(f"  Districts : {config.num_districts}")
    print(f"  Max turns : {config.max_turns}")
    print(f"  Seed      : {seed}")
    print(f"  Policy    : {'random' if fixed_action is None else fixed_action}")
    print("=" * 60)

    obs = env.reset()

    # Print initial state.
    for i, d in env.districts.items():
        print(f"  [init] {d}")
    print(f"  [init] {env.crisis}")
    print("-" * 60)

    total_rewards: Dict[AgentID, float] = {i: 0.0 for i in range(config.num_districts)}
    episode_done = False

    for _turn in range(config.max_turns):
        # Choose actions.
        if fixed_action is not None:
            actions = policy_fixed(obs, config.num_districts, fixed_action, parser)
        else:
            actions = policy_random(obs, config.num_districts, rng, parser)

        obs, rewards, done, truncated, info = env.step(actions)

        # Accumulate rewards.
        for i, r in rewards.items():
            total_rewards[i] += r

        # Per-turn console log.
        action_str = " | ".join(
            f"D{i}={info['actions_taken'].get(i, 'N/A')}" for i in range(config.num_districts)
        )
        reward_str = " | ".join(f"D{i} r={r:+.1f}" for i, r in rewards.items())
        crisis_str = f"crisis={info['crisis']['crisis_level']:.3f} ({info['crisis']['tier']})"
        stability_str = " | ".join(
            f"D{i} stab={info['districts'][i]['stability']:.3f}"
            for i in range(config.num_districts)
        )
        print(
            f"  T{info['turn']:02d} | {crisis_str} | {action_str} "
            f"| {reward_str} | {stability_str}"
        )

        if done or truncated:
            episode_done = True
            reason = "ALL COLLAPSED" if done else "TURN LIMIT"
            print(f"\n  >>> Episode ended: {reason} at turn {env.turn}")
            break

    if not episode_done:
        print(f"\n  >>> Episode ended: TURN LIMIT at turn {env.turn}")

    # ---- Summary ----
    print("=" * 60)
    print("  EPISODE SUMMARY")
    print("=" * 60)
    for i, d in env.districts.items():
        collapsed_str = " [COLLAPSED]" if env._collapsed[i] else ""
        print(f"  District {i}: total_reward={total_rewards[i]:+.1f}{collapsed_str}")
        print(f"    {d}")
    print(f"  Final crisis: {env.crisis}")
    print(f"  Crisis history (first 5): {env.crisis.history[:5]}")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run one District Accord episode.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    ap.add_argument(
        "--policy",
        choices=["random", "invest", "defend", "ignore"],
        default="random",
        help="Policy to use (default: random)",
    )
    args = ap.parse_args()
    fixed = None if args.policy == "random" else args.policy
    run(seed=args.seed, fixed_action=fixed)
