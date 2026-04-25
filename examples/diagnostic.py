"""
examples/diagnostic.py
=========================
Pre-Phase-4 diagnostic: run 10-20 episodes and measure four health signals.

Usage:
    python examples/diagnostic.py

Output:
    Per-check statistics, flag table, and a pass/warn/fail badge per check.
"""

from __future__ import annotations

import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, ".")   # run from repo root

from district_accord.env import DistrictAccordEnv
from district_accord.spaces.action import make_default_parsed_action
from district_accord.utils.config import EnvConfig
from district_accord.utils.types import AgentID, DiscreteAction


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------

def mask_aware_policy(
    obs: dict,
    agents: List[int],
    rng: np.random.Generator,
) -> dict:
    """
    Sample uniformly from valid actions only (mask[i] == 1.0).
    For directed actions (PROPOSE/SHARE/REQUEST_AID) pick a random other agent.
    """
    n = len(agents)
    actions = {}
    for agent_id in agents:
        mask = obs[agent_id]["action_mask"]
        valid_idxs = [i for i in range(len(mask)) if mask[i] == 1.0]
        action_idx = int(rng.choice(valid_idxs))
        action = DiscreteAction(action_idx)

        target: Optional[int] = None
        amount: Optional[float] = None

        if action in (
            DiscreteAction.PROPOSE_COALITION,
            DiscreteAction.SHARE_RESOURCES,
            DiscreteAction.REQUEST_AID,
        ):
            others = [a for a in agents if a != agent_id]
            target = int(rng.choice(others))

        if action == DiscreteAction.SHARE_RESOURCES:
            amount = 0.10

        actions[agent_id] = make_default_parsed_action(action, target=target, amount=amount)
    return actions


def fully_random_policy(
    agents: List[int],
    rng: np.random.Generator,
) -> dict:
    """Sample uniformly from ALL 9 actions, ignoring the mask."""
    n_actions = len(DiscreteAction)
    actions = {}
    for agent_id in agents:
        action_idx = int(rng.integers(0, n_actions))
        action = DiscreteAction(action_idx)

        target: Optional[int] = None
        amount: Optional[float] = None

        if action in (
            DiscreteAction.PROPOSE_COALITION,
            DiscreteAction.SHARE_RESOURCES,
            DiscreteAction.REQUEST_AID,
        ):
            others = [a for a in agents if a != agent_id]
            if others:
                target = int(rng.choice(others))

        if action == DiscreteAction.SHARE_RESOURCES:
            amount = 0.10

        actions[agent_id] = make_default_parsed_action(action, target=target, amount=amount)
    return actions


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

@dataclass
class EpisodeStat:
    ep_idx:                 int
    turns_played:           int
    proposals_created:      int     # total across all turns & agents
    max_proposals_per_turn: int     # worst single-turn spike
    coalition_formed_turn:  Optional[int]   # None if no coalition ever formed
    avg_trust_change:       float   # mean |Δtrust| per turn
    mask_violations:        int     # total violations across all turns & agents
    collapse_count:         int     # how many districts collapsed


@dataclass
class RunReport:
    mode:           str
    config:         EnvConfig
    episodes:       List[EpisodeStat] = field(default_factory=list)


def run_episode(
    env: DistrictAccordEnv,
    mode: str,
    seed: int,
    agents: List[int],
    rng: np.random.Generator,
) -> EpisodeStat:
    obs = env.reset(seed=seed)

    proposals_created   = 0
    max_proposals_turn  = 0
    coalition_turn: Optional[int] = None
    mask_violations     = 0
    collapse_count      = 0

    # Track trust for delta computation
    prev_trust_flat: Optional[np.ndarray] = None
    trust_deltas: List[float] = []

    def _flatten_trust(info: dict) -> np.ndarray:
        tm = info["trust"]["trust_matrix"]
        vals = []
        for i_str, row in sorted(tm.items()):
            for j_str, v in sorted(row.items()):
                if i_str != j_str:
                    vals.append(float(v))
        return np.array(vals, dtype=np.float64) if vals else np.array([])

    for turn in range(env.config.max_turns):
        # Build actions
        active_agents = [a for a in agents if not env._collapsed[a]]
        if not active_agents:
            break

        if mode == "mask_aware":
            active_obs = {a: obs[a] for a in active_agents}
            raw_actions = mask_aware_policy(active_obs, active_agents, rng)
        else:
            raw_actions = fully_random_policy(active_agents, rng)

        # Fill collapsed agents with IGNORE (required by env)
        full_actions = {}
        for a in agents:
            if env._collapsed[a]:
                full_actions[a] = DiscreteAction.IGNORE
            else:
                full_actions[a] = raw_actions[a]

        obs, rewards, done, trunc, info = env.step(full_actions)

        # ── Check 1: proposals ───────────────────────────────────────────
        pending_now = info["negotiation"]["pending_count"]
        # Count how many were created this turn = pending AFTER - pending BEFORE + accepted + rejected + expired
        # Instead, sum up from all_pending diff. Simpler: track via the negotiation's next_id delta.
        # We use the pending_count delta as a proxy.
        turn_proposals = max(0, pending_now - max(0, proposals_created - mask_violations))
        # Actually: just count pending_count as a measure of spam
        max_proposals_turn = max(max_proposals_turn, pending_now)
        proposals_created += pending_now  # cumulative approximation

        # ── Check 2: coalition formation ─────────────────────────────────
        if coalition_turn is None:
            coal_info = info["coalition"]["coalitions"]
            if coal_info:  # any coalition formed
                coalition_turn = info["turn"]  # turn index after step

        # ── Check 3: trust evolution ──────────────────────────────────────
        trust_flat = _flatten_trust(info)
        if prev_trust_flat is not None and len(trust_flat) == len(prev_trust_flat):
            delta = float(np.mean(np.abs(trust_flat - prev_trust_flat)))
            trust_deltas.append(delta)
        prev_trust_flat = trust_flat

        # ── Check 4: mask violations ─────────────────────────────────────
        mask_violations += len(info["mask_violations"])

        # ── Collapse count ───────────────────────────────────────────────
        collapse_count = sum(info["collapsed"].values())

        if done or trunc:
            break

    avg_trust_change = float(np.mean(trust_deltas)) if trust_deltas else 0.0

    return EpisodeStat(
        ep_idx=seed,
        turns_played=info["turn"],
        proposals_created=proposals_created,
        max_proposals_per_turn=max_proposals_turn,
        coalition_formed_turn=coalition_turn,
        avg_trust_change=avg_trust_change,
        mask_violations=mask_violations,
        collapse_count=collapse_count,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

RESET   = "\033[0m"
BOLD    = "\033[1m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
CYAN    = "\033[96m"
DIM     = "\033[2m"


def badge(label: str, status: str) -> str:
    colour = {"PASS": GREEN, "WARN": YELLOW, "FAIL": RED}.get(status, "")
    return f"{colour}{BOLD}[{status}]{RESET} {label}"


def print_check(
    number: str,
    title: str,
    description: str,
    rows: List[str],
    verdict: str,
    status: str,
) -> None:
    SEP = "─" * 62
    print(f"\n{CYAN}{BOLD}CHECK {number}: {title}{RESET}")
    print(f"{DIM}{description}{RESET}")
    print(SEP)
    for row in rows:
        print(f"  {row}")
    print(SEP)
    print(f"  {badge(verdict, status)}")


def analyze_and_print(report: RunReport) -> Dict[str, str]:
    eps = report.episodes
    n   = len(eps)
    mode_label = "mask-aware random" if report.mode == "mask_aware" else "fully-random (ignoring mask)"

    print(f"\n{'═'*62}")
    print(f"{BOLD} District Accord — Pre-Phase-4 Diagnostic{RESET}")
    print(f"{'═'*62}")
    print(f"  Policy mode : {mode_label}")
    print(f"  Episodes    : {n}")
    print(f"  Agents      : {report.config.num_districts}")
    print(f"  Max turns   : {report.config.max_turns}")

    statuses = {}

    # ── CHECK 1: Proposal Spam ───────────────────────────────────────────
    avg_pending = np.mean([e.proposals_created / max(e.turns_played, 1) for e in eps])
    max_spam    = max(e.max_proposals_per_turn for e in eps)
    ep_with_spam = sum(1 for e in eps if e.max_proposals_per_turn >= 3)

    # With max_pending=M outgoing+incoming and N agents, the system-wide
    # peak pending can legitimately reach M * N. Flag FAIL only when the
    # observed peak exceeds that theoretical maximum (cap not enforced).
    system_cap = report.config.max_pending_proposals * len(report.config.__dataclass_fields__)
    n_agents   = report.config.num_districts
    per_agent_max = report.config.max_pending_proposals
    # A single-turn snapshot > (per_agent_max * n_agents) means cap is broken
    spam_status = (
        "PASS" if max_spam <= per_agent_max * n_agents else
        "WARN" if max_spam <= per_agent_max * n_agents * 1.5 else
        "FAIL"
    )
    statuses["spam"] = spam_status

    print_check(
        "1", "Proposal Spam",
        "Do agents flood the system with proposals?",
        [
            f"{'Avg pending per turn':<35} {avg_pending:.2f}",
            f"{'Max proposals in one turn (any ep)':<35} {max_spam}",
            f"{'Episodes with max ≥ 3':<35} {ep_with_spam}/{n}",
            f"{'Config max_pending_proposals':<35} {report.config.max_pending_proposals}",
            f"{'Config proposal_cooldown':<35} {report.config.proposal_cooldown} turns",
        ],
        (   "No spam detected" if spam_status == "PASS" else
            "Moderate spam — cooldown may need tuning" if spam_status == "WARN" else
            "Spam detected — increase cooldown or proposal_cost"
        ),
        spam_status,
    )

    # ── CHECK 2: Coalition Formation ─────────────────────────────────────
    formed_eps    = [e for e in eps if e.coalition_formed_turn is not None]
    n_formed      = len(formed_eps)
    if formed_eps:
        avg_turn  = np.mean([e.coalition_formed_turn for e in formed_eps])
        min_turn  = min(e.coalition_formed_turn for e in formed_eps)
        max_turn  = max(e.coalition_formed_turn for e in formed_eps)
    else:
        avg_turn = min_turn = max_turn = float("nan")

    coal_status = (
        "WARN" if n_formed == 0 else
        "WARN" if n_formed == n and (avg_turn if avg_turn == avg_turn else 999) <= 2 else
        "PASS"
    )
    statuses["coal"] = coal_status

    print_check(
        "2", "Coalition Formation Speed",
        "Do coalitions form organically or never/instantly?",
        [
            f"{'Episodes with coalition formed':<35} {n_formed}/{n}",
            f"{'Avg turn of first coalition':<35} {avg_turn:.1f}" if n_formed else f"{'Avg turn of first coalition':<35} —",
            f"{'Range':<35} [{min_turn:.0f}, {max_turn:.0f}]" if n_formed else f"{'Range':<35} —",
            f"{'Config proposal_ttl':<35} {report.config.proposal_ttl} turns",
        ],
        (   "No coalition ever formed — check PROPOSE action probability" if n_formed == 0 else
            "Coalitions form instantly (turn ≤ 2) — may need more turns to negotiate" if avg_turn <= 2 else
            "Coalitions form organically over multiple turns"
        ),
        coal_status,
    )

    # ── CHECK 3: Trust Signal Usage ──────────────────────────────────────
    avg_delta   = np.mean([e.avg_trust_change for e in eps])
    max_delta   = max(e.avg_trust_change for e in eps)
    flat_eps    = sum(1 for e in eps if e.avg_trust_change < 1e-4)

    trust_status = (
        "FAIL" if avg_delta < 1e-5 else
        "WARN" if flat_eps > n // 2 else
        "PASS"
    )
    statuses["trust"] = trust_status

    print_check(
        "3", "Trust Signal Evolution",
        "Does trust evolve meaningfully (accept/reject + decay)?",
        [
            f"{'Mean |Δtrust| per turn':<35} {avg_delta:.5f}",
            f"{'Max  |Δtrust| per turn':<35} {max_delta:.5f}",
            f"{'Episodes with zero trust change':<35} {flat_eps}/{n}",
            f"{'Config trust_decay':<35} {report.config.trust_decay}",
            f"{'Config trust_accept_bonus':<35} {report.config.trust_accept_bonus}",
        ],
        (   "Trust is completely static — check update calls" if trust_status == "FAIL" else
            "Trust changes slowly (mostly decay-driven)" if trust_status == "WARN" else
            "Trust evolves meaningfully from both interactions and decay"
        ),
        trust_status,
    )

    # ── CHECK 4: Invalid Action Rate ─────────────────────────────────────
    total_violations = sum(e.mask_violations for e in eps)
    avg_per_turn     = np.mean([
        e.mask_violations / max(e.turns_played, 1) for e in eps
    ])
    ep_zero_viol     = sum(1 for e in eps if e.mask_violations == 0)

    # For mask_aware policy: 0 violations is PASS; for fully_random: some violations expected
    if report.mode == "mask_aware":
        viol_status = "PASS" if total_violations == 0 else "FAIL"
    else:
        # Fully random — some violations expected; check they're handled gracefully
        viol_status = "PASS" if total_violations > 0 else "WARN"

    statuses["viol"] = viol_status

    print_check(
        "4", "Invalid Action Exploitation",
        (   "With mask-aware policy: violations = 0 expected.\n"
            "  With fully-random policy: violations should be caught and penalised."
        ),
        [
            f"{'Total mask violations':<35} {total_violations}",
            f"{'Avg violations per turn':<35} {avg_per_turn:.2f}",
            f"{'Episodes with 0 violations':<35} {ep_zero_viol}/{n}",
            f"{'Config mask_violation_penalty':<35} {report.config.mask_violation_penalty}",
        ],
        (   "Zero violations — mask-aware policy working correctly" if (report.mode == "mask_aware" and total_violations == 0) else
            "Violations present — env correctly converts to IGNORE + penalty" if report.mode == "fully_random" else
            "Violations detected with mask-aware policy — check mask logic"
        ),
        viol_status,
    )

    # ── Overall summary ──────────────────────────────────────────────────
    overall = (
        "PASS" if all(s == "PASS" for s in statuses.values()) else
        "FAIL" if any(s == "FAIL" for s in statuses.values()) else
        "WARN"
    )

    print(f"\n{'═'*62}")
    print(f"{BOLD} OVERALL ({mode_label}){RESET}")
    print(f"{'═'*62}")
    print(f"  Spam        {statuses['spam']}")
    print(f"  Coalition   {statuses['coal']}")
    print(f"  Trust       {statuses['trust']}")
    print(f"  Mask        {statuses['viol']}")
    print(f"  {'━'*30}")
    print(f"  {badge('Environment health', overall)}")
    print(f"{'═'*62}\n")

    return statuses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    N_EPISODES = 20
    N_AGENTS   = 4
    MAX_TURNS  = 30

    cfg = EnvConfig(
        num_districts=N_AGENTS,
        max_turns=MAX_TURNS,
        # Phase 4 anti-spam defaults
        proposal_cost=0.04,
        max_pending_proposals=2,
        proposal_ttl=4,
        proposal_cooldown=4,
        trust_init_std=0.05,
        trust_decay=0.99,
        trust_accept_bonus=0.10,
        trust_reject_penalty=0.05,
        # Phase 4 reward weights
        reward_cooperation_per_peer=0.05,
        reward_stability_weight=1.0,
        reward_crisis_weight=1.0,
        reward_trust_alignment=0.02,
        coalition_exposure_damping=0.15,
        mask_violation_penalty=-0.5,
        obs_neighbor_noise_std=0.02,
    )

    rng = np.random.default_rng(42)

    print("\nRunning mask-aware random policy episodes...")
    env = DistrictAccordEnv(cfg)
    aware_report = RunReport(mode="mask_aware", config=cfg)
    for ep in range(N_EPISODES):
        stat = run_episode(
            env=env,
            mode="mask_aware",
            seed=ep,
            agents=list(range(N_AGENTS)),
            rng=np.random.default_rng(ep * 17 + 3),
        )
        aware_report.episodes.append(stat)

    analyze_and_print(aware_report)

    print("\nRunning fully-random policy episodes (ignoring mask)...")
    env2 = DistrictAccordEnv(cfg)
    random_report = RunReport(mode="fully_random", config=cfg)
    for ep in range(N_EPISODES):
        stat = run_episode(
            env=env2,
            mode="fully_random",
            seed=ep,
            agents=list(range(N_AGENTS)),
            rng=np.random.default_rng(ep * 31 + 7),
        )
        random_report.episodes.append(stat)

    analyze_and_print(random_report)


if __name__ == "__main__":
    main()
