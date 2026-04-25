"""
examples/episode_trace.py
===========================
Phase 5 example: single fully-traced episode.

Shows per-turn:
    - Events from EventBus (pipeline boundary log)
    - State snapshot (per-agent resources / stability / coalition)
    - Reward breakdown (per-agent, all components)

Run:
    python examples/episode_trace.py

Policy: mixed — form coalition on turn 1, then DEFEND + INVEST by state.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from district_accord.env import DistrictAccordEnv
from district_accord.spaces.action import make_default_parsed_action
from district_accord.utils.config import EnvConfig
from district_accord.utils.types import DiscreteAction

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

CFG = EnvConfig(
    num_districts=4,
    max_turns=6,          # short trace for readability
    seed=42,
    trust_init_std=0.0,
    obs_neighbor_noise_std=0.0,
    proposal_cost=0.04,
    max_pending_proposals=2,
    proposal_cooldown=4,
    reward_spam_penalty=0.0,
)

N = CFG.num_districts


# ─── Formatting helpers ───────────────────────────────────────────────────────

EVENT_COLOURS = {
    "action_validated":    DIM,
    "action_invalid":      RED,
    "proposal_created":    YEL,
    "proposal_rejected":   RED,
    "proposal_expired":    RED,
    "coalition_formed":    GRN,
    "coalition_joined":    GRN,
    "resource_transferred": CYAN,
    "trust_updated":       MAG,
    "collapse":            RED + BOLD,
}

def bar(v: float, width: int = 10) -> str:
    filled = int(round(v * width))
    return f"{'█' * filled}{'░' * (width - filled)} {v:.2f}"


def print_turn_header(turn: int) -> None:
    print(f"\n{CYAN}{BOLD}{'─' * 70}{R}")
    print(f"{CYAN}{BOLD}  TURN {turn:>2}{R}")
    print(f"{CYAN}{'─' * 70}{R}")


def print_events(events: list) -> None:
    if not events:
        print(f"  {DIM}(no events){R}")
        return
    # Group: skip high-volume action_validated unless something interesting
    non_action = [e for e in events if e["type"] not in ("action_validated",)]
    action_val  = [e for e in events if e["type"] == "action_validated"]

    # Print summary of action_validated
    actions_summary = ", ".join(
        f"A{e['payload']['agent_id']}:{e['payload']['action'][:3]}"
        for e in action_val
    )
    if actions_summary:
        print(f"  {DIM}[actions] {actions_summary}{R}")

    for e in non_action:
        colour = EVENT_COLOURS.get(e["type"], "")
        tag = f"{colour}[{e['type']}]{R}"
        p = e["payload"]
        # Format payload compactly
        payload_str = "  ".join(f"{k}={v}" for k, v in p.items())
        print(f"  {tag}  {payload_str}")


def print_state(snapshot: dict) -> None:
    agents = snapshot["agents"]
    print(f"  {BOLD}{'Agent':<6} {'Resources':>12}  {'Stability':>12}  "
          f"{'Coalition':>10}  {'Collapsed':>9}{R}")
    for aid_str, a in sorted(agents.items(), key=lambda x: int(x[0])):
        cid   = str(a["coalition_id"]) if a["coalition_id"] is not None else "—"
        coll  = f"{RED}YES{R}" if a["collapsed"] else "no"
        res   = bar(a["resources"], 8)
        stab  = bar(a["stability"], 8)
        print(f"  A{aid_str}     {res}  {stab}  {cid:>10}  {coll:>9}")
    print(f"  {DIM}crisis_level={snapshot['crisis_level']:.3f}  "
          f"coalitions={snapshot['active_coalitions']}{R}")


def print_rewards(breakdown: dict) -> None:
    print(f"  {BOLD}{'Agent':<6} {'Total':>7}  {'Surv':>6}  {'ΔStab':>6}  "
          f"{'Crisis':>7}  {'Coop':>6}  {'Trust':>6}{R}")
    for a in sorted(breakdown):
        bd = breakdown[a]
        print(
            f"  A{a}     "
            f"{bd['total']:>7.3f}  "
            f"{bd['survival']:>6.3f}  "
            f"{bd['stability_delta']:>6.3f}  "
            f"{bd['crisis_mitigation']:>7.3f}  "
            f"{bd['cooperation']:>6.3f}  "
            f"{bd['trust_alignment']:>6.3f}"
        )


# ─── Policy ───────────────────────────────────────────────────────────────────

def choose_actions(env: DistrictAccordEnv, turn: int) -> dict:
    obs = env._get_obs()
    actions = {}

    if turn == 0:
        for proposer, target in [(0, 1), (2, 3)]:
            if (not env._collapsed[proposer] and
                    obs[proposer]["action_mask"][DiscreteAction.PROPOSE_COALITION]):
                actions[proposer] = make_default_parsed_action(
                    DiscreteAction.PROPOSE_COALITION, target=target
                )
    elif turn == 1:
        for a in [1, 3]:
            if (not env._collapsed[a] and
                    obs[a]["action_mask"][DiscreteAction.ACCEPT_COALITION]):
                actions[a] = make_default_parsed_action(DiscreteAction.ACCEPT_COALITION)

    for a in range(N):
        if a not in actions and not env._collapsed[a]:
            d = env._districts[a]
            if d.crisis_exposure > 0.1:
                actions[a] = DiscreteAction.DEFEND
            elif d.stability < 0.7 and d.resources > 0.25:
                actions[a] = DiscreteAction.INVEST
            else:
                actions[a] = DiscreteAction.RECOVER

    return actions


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{CYAN}{BOLD}{'═' * 70}{R}")
    print(f"{CYAN}{BOLD}  District Accord — Phase 5 Episode Trace{R}")
    print(f"{CYAN}{BOLD}  Agents={N}  Turns={CFG.max_turns}  Seed={CFG.seed}{R}")
    print(f"{CYAN}{BOLD}{'═' * 70}{R}")

    env = DistrictAccordEnv(CFG)
    env.reset()

    cumulative = {a: 0.0 for a in range(N)}
    total_events = 0

    for turn in range(CFG.max_turns):
        actions = choose_actions(env, turn)
        _, rewards, done, trunc, info = env.step(actions)

        n_events = len(info["events"])
        total_events += n_events

        print_turn_header(info["turn"])

        print(f"\n  {BOLD}● Events ({n_events}):{R}")
        print_events(info["events"])

        print(f"\n  {BOLD}● State snapshot:{R}")
        print_state(info["state_snapshot"])

        print(f"\n  {BOLD}● Reward breakdown:{R}")
        print_rewards(info["reward_breakdown"])

        for a, r in rewards.items():
            cumulative[a] += r

        if done or trunc:
            reason = "all collapsed" if done else "turn limit"
            print(f"\n  {YEL}Episode ended: {reason}{R}")
            break

    # ── Episode summary ──────────────────────────────────────────────────────
    print(f"\n{CYAN}{BOLD}{'═' * 70}{R}")
    print(f"{CYAN}{BOLD}  EPISODE SUMMARY{R}")
    print(f"{CYAN}{'═' * 70}{R}")
    print(f"\n  Total events logged: {BOLD}{total_events}{R}")
    print(f"  State snapshots:     {BOLD}{len(env._state_tracker)}{R}")
    print(f"\n  {BOLD}Cumulative rewards:{R}")
    for a, total in cumulative.items():
        colour = GRN if total == max(cumulative.values()) else ""
        print(f"    A{a}: {colour}{total:+.3f}{R}")

    # ── Full event log stats ─────────────────────────────────────────────────
    from collections import Counter
    all_events = env._event_bus.get_events()
    type_counts = Counter(e.event_type for e in all_events)
    print(f"\n  {BOLD}Event type distribution:{R}")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        bar_w = min(count, 30)
        print(f"    {etype:<22} {'█' * bar_w} {count}")

    print()


if __name__ == "__main__":
    main()
