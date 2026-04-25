"""
examples/run_self_play.py
===========================
Phase 6 example: self-play episode with 12 agents, 100 turns.

Shows:
    - Episode total reward per agent
    - Coalition formation stats
    - Collapse count
    - Event distribution
    - Deterministic replay verification

Run:
    python examples/run_self_play.py [--mode MODE] [--seed SEED]

    MODE: random | mask_aware_random | rule_based  (default: rule_based)
    SEED: any integer                              (default: 42)
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, ".")

from district_accord.env import DistrictAccordEnv
from district_accord.policy.runner import EpisodeRunner, save_trajectory, verify_replay
from district_accord.policy.self_play import SelfPlayPolicy
from district_accord.utils.config import EnvConfig

# ─── ANSI ────────────────────────────────────────────────────────────────────
R    = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
YEL  = "\033[93m"
GRN  = "\033[92m"
RED  = "\033[91m"
DIM  = "\033[2m"
MAG  = "\033[95m"


# ─── Config ───────────────────────────────────────────────────────────────────

def build_cfg(seed: int) -> EnvConfig:
    return EnvConfig(
        num_districts=12,
        max_turns=100,
        seed=seed,
        trust_init_std=0.0,
        obs_neighbor_noise_std=0.0,
        proposal_cost=0.04,
        proposal_cooldown=4,
        max_pending_proposals=2,
        reward_spam_penalty=0.0,
    )


# ─── Formatting ───────────────────────────────────────────────────────────────

def bar(v: float, width: int = 12, vmax: float = None) -> str:
    if vmax is None:
        vmax = 1.0
    ratio = min(v / max(vmax, 1e-6), 1.0)
    filled = int(round(ratio * width))
    return f"{'█' * filled}{'░' * (width - filled)}"


def hbar(count: int, total: int, width: int = 20) -> str:
    filled = int(round((count / max(total, 1)) * width))
    return "█" * filled + "░" * (width - filled)


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="District Accord — self-play episode")
    parser.add_argument("--mode", default="rule_based",
                        choices=["random", "mask_aware_random", "rule_based"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true",
                        help="Save trajectory to episode.json")
    parser.add_argument("--verify", action="store_true",
                        help="Re-run and verify deterministic replay")
    args = parser.parse_args()

    cfg    = build_cfg(args.seed)
    env    = DistrictAccordEnv(cfg)
    policy = SelfPlayPolicy(mode=args.mode, seed=args.seed)
    runner = EpisodeRunner()

    print(f"\n{CYAN}{BOLD}{'═' * 72}{R}")
    print(f"{CYAN}{BOLD}  District Accord — Self-Play Episode{R}")
    print(f"{CYAN}  Agents={cfg.num_districts}  MaxTurns={cfg.max_turns}  "
          f"Mode={args.mode}  Seed={args.seed}{R}")
    print(f"{CYAN}{'═' * 72}{R}\n")

    t0 = time.perf_counter()
    trajectory = runner.run_episode(env, policy, seed=args.seed)
    elapsed = time.perf_counter() - t0

    summary = runner.episode_summary(trajectory)

    print(f"  {BOLD}Episode completed in {elapsed * 1000:.1f} ms{R}")
    print(f"  Turns played:    {BOLD}{summary['turns_played']}{R} / {cfg.max_turns}")
    print(f"  Total events:    {BOLD}{summary['total_events']}{R}")
    print(f"  Collapses:       {BOLD}{RED}{summary['collapses']}{R}")
    print(f"  Coalition events:{BOLD}{GRN}{summary['coalition_events']}{R}")
    print(f"  Avg reward/turn: {BOLD}{summary['avg_reward_per_turn_per_agent']:+.3f}{R}")

    # ── Per-agent total rewards ──────────────────────────────────────────────
    total_rewards = {int(k): v for k, v in summary["total_rewards"].items()}
    max_r = max(total_rewards.values(), default=1)
    min_r = min(total_rewards.values(), default=0)

    print(f"\n{CYAN}{BOLD}  Per-Agent Total Rewards{R}")
    print(f"  {'Agent':<6} {'Reward':>8}  {'Bar':}")
    for a, r in sorted(total_rewards.items()):
        colour = GRN if r == max_r else (RED if r == min_r else "")
        print(f"  A{a:02d}    {colour}{r:>8.3f}{R}  {bar(r, 16, max_r)}")

    # ── Coalition and collapse stats ─────────────────────────────────────────
    last_snap = trajectory[-1].info.get("state_snapshot", {})
    agents_snap = last_snap.get("agents", {})
    alive = sum(1 for a in agents_snap.values() if not a["collapsed"])
    collapsed_n = cfg.num_districts - alive

    print(f"\n{CYAN}{BOLD}  Final State{R}")
    print(f"  {'Agent':<6} {'Res':>6}  {'Stab':>6}  {'Coalition':>10}  {'Status':>9}")
    for a_str, snap in sorted(agents_snap.items(), key=lambda x: int(x[0])):
        cid    = str(snap["coalition_id"]) if snap["coalition_id"] is not None else "—"
        status = f"{RED}COLLAPSED{R}" if snap["collapsed"] else f"{GRN}alive{R}"
        print(
            f"  A{int(a_str):02d}    "
            f"{snap['resources']:>6.3f}  "
            f"{snap['stability']:>6.3f}  "
            f"{cid:>10}  "
            f"{status}"
        )
    print(f"\n  Alive: {GRN}{alive}{R}  Collapsed: {RED}{collapsed_n}{R}"
          f"  Active coalitions: {last_snap.get('active_coalitions', 0)}")

    # ── Event distribution ───────────────────────────────────────────────────
    from collections import Counter
    all_events = env._event_bus.get_events()
    counts = Counter(e.event_type for e in all_events)
    total_ev = max(len(all_events), 1)

    print(f"\n{CYAN}{BOLD}  Event Distribution (/episode){R}")
    for etype, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {etype:<22} {hbar(cnt, total_ev)} {cnt:>5}")

    # ── Save trajectory ──────────────────────────────────────────────────────
    if args.save:
        path = "episode.json"
        save_trajectory(trajectory, path)
        print(f"\n  {GRN}Trajectory saved → {path}{R}")

    # ── Deterministic replay verification ────────────────────────────────────
    if args.verify:
        print(f"\n  {YEL}Verifying deterministic replay...{R}", end=" ")
        policy2 = SelfPlayPolicy(mode=args.mode, seed=args.seed)
        env2    = DistrictAccordEnv(cfg)
        ok = verify_replay(env2, trajectory, policy2, seed=args.seed)
        if ok:
            print(f"{GRN}{BOLD}PASS ✓{R}")
        else:
            print(f"{RED}{BOLD}FAIL ✗{R}")
            sys.exit(1)

    print()


if __name__ == "__main__":
    main()
