# District Accord — Progress Tracker

> Last updated: **Phase 5 complete** | 2026-04-25

---

## Phase Status

| Phase | Scope | Status | Tests |
|---|---|---|---|
| **Phase 1** | Foundation (types, config, district, crisis, env) | ✅ Complete | 69 |
| **Phase 2** | Action space + Observation space | ✅ Complete | 83 |
| **Phase 3** | Coalition + Negotiation + Trust | ✅ Complete | 113 |
| **Phase 4** | Reward Engine (RL-stable, exploit-resistant) | ✅ Complete | 51 |
| **Phase 5** | Engine Layer (EventBus, StateTracker, TurnManager) | ✅ Complete | 40 |
| **Phase 6** | Full env (12 agents, 100 turns) + RL training | 🔒 Next | — |

**Total tests passing: 356 (0 failures)**

---

## ✅ Phase 5 Complete — Engine Layer

**Date:** 2026-04-25 | **Tests:** 40 new (356 total)

### New Modules

```
district_accord/engine/
├── event_bus.py      ← EventBus + Event (append-only episode log)
├── state_tracker.py  ← StateTracker + TurnSnapshot + AgentSnapshot
└── turn_manager.py   ← TurnManager (pipeline coordinator)
```

### New Files

```
tests/test_phase5_engine.py
examples/episode_trace.py
examples/exploit_tests.py
```

### EventBus

| Method | Description |
|---|---|
| `set_turn(t)` | Tag subsequent events with turn `t` |
| `emit(type, payload)` | Append event; raises ValueError for unknown types |
| `get_events(turn, event_type)` | Filtered query — returns copy |
| `get_current_turn_events()` | Shorthand for current turn |
| `clear()` | Reset for new episode (called by `env.reset()`) |
| `to_list()` | Serialisable `List[dict]` |

**10 mandatory event types:**
`action_validated`, `action_invalid`, `proposal_created`, `proposal_rejected`,
`proposal_expired`, `coalition_formed`, `coalition_joined`, `resource_transferred`,
`trust_updated`, `collapse`

### StateTracker

Records a `TurnSnapshot` at the end of every step:

```python
TurnSnapshot(turn, crisis_level, active_coalitions, total_events,
             agents: Dict[int, AgentSnapshot])

AgentSnapshot(agent_id, resources, stability, crisis_exposure,
              coalition_id, trust_row, collapsed)
```

| Method | Description |
|---|---|
| `record(turn, districts, crisis, coalition, trust, collapsed, n_events)` | Build + store snapshot |
| `get_history()` | Full time-series (copy) |
| `get_turn(t)` | Single snapshot by turn number |
| `reset()` | Clear — called by `env.reset()` |
| `to_list()` | Serialisable `List[dict]` |

### TurnManager

Thin coordinator above `env._execute_step_pipeline()`:

```
run_turn(raw_actions, env)
    │
    ├── event_bus.set_turn(env._turn)
    ├── mark events_before = len(event_bus)
    ├── env._execute_step_pipeline(raw_actions)   ← all 19 steps + event emission
    ├── state_tracker.record(...)
    └── inject info["events"] + info["state_snapshot"]
```

### env.py Changes

| Change | Detail |
|---|---|
| `__init__` | Added `_event_bus`, `_state_tracker`, `_turn_manager` |
| `reset()` | Calls `event_bus.clear()` + `state_tracker.reset()` |
| `step()` | Delegates to `turn_manager.run_turn(actions, self)` |
| `_execute_step_pipeline()` | Former step body; emits events at 7 pipeline boundaries |
| `_process_negotiation_actions()` | Now returns 3-tuple: `(accept, reject, created_proposals)` |
| `_apply_resource_actions()` | Now returns `List[(from, to, amount)]` transfers |
| `metadata["version"]` | Bumped to `"0.5.0"` / phase 5 |

### Info dict (augmented)

```python
info["events"]          # List[dict] — events from this turn (type, turn, seq, payload)
info["state_snapshot"]  # dict — TurnSnapshot for this turn
info["reward_breakdown"] # Phase 4 — unchanged
info["coalition"]        # Phase 3 — unchanged
info["negotiation"]      # Phase 3 — unchanged
info["trust"]            # Phase 3 — unchanged
```

### Pipeline Event Order (locked)

```
1.  action_validated  / action_invalid   (after mask enforcement, sorted agent_id)
2.  proposal_created                      (after successful PROPOSE/REQUEST_AID)
3.  coalition_formed  / coalition_joined  (after ACCEPT resolves)
4.  proposal_rejected                     (after REJECT resolves)
5.  resource_transferred                  (after SHARE_RESOURCES)
6.  proposal_expired                      (after TTL tick)
7.  collapse                              (after update_collapse_status)
8.  trust_updated                         (after trust update, from accept/reject events)
```

### Determinism Guarantee

Same seed → identical event sequence, identical state snapshots across episodes.
Verified by `test_deterministic_replay_same_events`.

---

## ✅ Phase 4 Complete — Reward Engine

**Date:** 2026-04-25 | **Tests:** 51 new

### RewardEngine (`engine/reward.py`)

Stateless, deterministic. 8 modular components per agent per turn:

| Component | Formula | Cap |
|---|---|---|
| Survival | `+config.survival_reward` | — |
| Stability delta | `clip(w × (curr − prev), ±0.50)` | ±0.50 |
| Crisis mitigation | `−w × exposure` | — |
| Cooperation | `min(per_peer × (size−1), 0.15)` | 0.15 |
| Trust alignment | `min(w × avg_pos_trust, 0.05)` | 0.05 |
| Mask penalty | `config.mask_violation_penalty` if violated | — |
| Spam penalty | `−w × max(0, pending − cap)` | — |
| Collapse penalty | override — survival only on collapse turn | — |

### Anti-spam (Phase 4 fixes, still active)

| Config field | Value |
|---|---|
| `proposal_cooldown` | 4 |
| `proposal_cost` | 0.04 |
| `max_pending_proposals` | 2 |
| Incoming cap | enforced in `NegotiationSystem.create()` |

### Exploit Tests (all pass)

| Test | Result |
|---|---|
| Coalition + Idle < Mixed | ✅ PASS |
| Trust signal decays (no farm) | ✅ PASS |
| DEFEND alone < Mixed | ✅ PASS |
| Mixed (cooperate+stabilize) wins | ✅ PASS |

---

## ✅ Phase 3 Complete — Coalition, Negotiation & Trust

**Date:** 2026-04-25 | **Tests:** 113 new

### Systems

| System | File | Description |
|---|---|---|
| TrustSystem | `core/trust.py` | `trust[i][j] ∈ [-1,1]`, bilateral accept bonus, asymmetric reject penalty, per-turn decay |
| NegotiationSystem | `core/negotiation.py` | `Proposal` dataclass, create/tick/accept/reject, TTL, per-proposer cap, cooldown, incoming cap |
| CoalitionSystem | `core/coalition.py` | Pure membership: new_coalition/join/leave/same_coalition/coalition_size |

### Observation (Phase 3+)

| Key | Shape | Description |
|---|---|---|
| `"self"` | `(4,)` | `[resources, stability, crisis_exposure, stability_delta]` |
| `"others"` | `(N-1, 4)` | Peer view: `[resources, stability, trust, coalition_flag]` |
| `"crisis"` | `(2,)` | `[crisis_level, normalized_tier]` |
| `"turn"` | `(2,)` | `[progress, remaining]` |
| `"action_mask"` | `(9,)` | Dynamic per-state mask |
| `"flat"` | `(4N+4,)` | Concatenated flat vector |

---

## ✅ Phase 2 Complete — Action & Observation Space

**Date:** 2026-04-25 | **Tests:** 83 new

### Action Space

| Action | Effect |
|---|---|
| INVEST | resources += gain − cost; stability += effect |
| DEFEND | resources −= cost; stability += gain; exposure −= reduction |
| IGNORE | No effect |
| RECOVER | resources −= cost; stability += gain (larger) |
| REQUEST_AID | Creates aid proposal to target |
| SHARE_RESOURCES | Transfers amount to target (loss ratio applied) |
| PROPOSE_COALITION | Creates coalition proposal to target |
| ACCEPT_COALITION | Resolves first coalition proposal |
| REJECT_COALITION | Rejects first coalition proposal |

---

## ✅ Phase 1 Complete — Foundation

**Date:** 2026-04-25 | **Tests:** 69

Core: `env.py`, `core/district.py`, `core/crisis.py`, `spaces/action_parser.py`,
`utils/types.py`, `utils/config.py`

---

## Environment

| Item | Value |
|---|---|
| Python | 3.12 |
| Virtual env | `.venv/` |
| Install | `pip install -e ".[dev]"` |
| Run tests | `.venv/bin/pytest tests/ -v` |
| Episode trace | `.venv/bin/python examples/episode_trace.py` |
| Diagnostic | `.venv/bin/python examples/diagnostic.py` |
| Exploit tests | `.venv/bin/python examples/exploit_tests.py` |

## Locked Design Decisions

| Key | Value |
|---|---|
| `ACTION_FORMAT` | `"structured_text"` |
| `ACTION_PARSER` | `"external"` |
| `OBSERVATION_MODE` | `"dict"` + flat vector |
| `STATE_AUTHORITY` | `"centralized_env"` |
| Resolution order | `sorted(agent_ids)` — all pipelines |
| Proposal ownership | `NegotiationSystem` |
| `OTHERS_SCHEMA` | `["resources","stability","trust","coalition_flag"]` |
| Reward determinism | stateless `RewardEngine`, `sorted(agents)` |
| Event ordering | insertion order = pipeline order |

## File Structure

```
district_accord/
├── env.py                      ← DistrictAccordEnv (Phase 5: delegates to TurnManager)
├── core/
│   ├── coalition.py
│   ├── crisis.py
│   ├── district.py
│   ├── negotiation.py
│   └── trust.py
├── engine/
│   ├── event_bus.py            ← NEW Phase 5
│   ├── reward.py               ← Phase 4
│   ├── state_tracker.py        ← NEW Phase 5
│   └── turn_manager.py         ← NEW Phase 5
├── spaces/
│   ├── action.py
│   ├── action_parser.py
│   └── observation.py
└── utils/
    ├── config.py
    └── types.py

tests/
├── test_crisis.py
├── test_district.py
├── test_env.py
├── test_phase2.py
├── test_phase3_coalition.py
├── test_phase3_integration.py
├── test_phase3_negotiation.py
├── test_phase3_trust.py
├── test_phase4_reward.py
└── test_phase5_engine.py       ← NEW Phase 5

examples/
├── diagnostic.py
├── episode_trace.py            ← NEW Phase 5
├── exploit_tests.py            ← NEW Phase 4 validation
└── run_episode.py
```
