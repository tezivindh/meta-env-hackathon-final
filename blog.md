# 🌍 District Accord: Training an LLM to Negotiate, Cooperate, and Survive

**Team:** tezivindh1 | **Themes:** Multi-Agent Interactions · Long-Horizon Planning · Self-Improvement

---

## The Problem

Imagine 12 autonomous districts facing a rising global crisis. Each district can hoard resources
and defend itself — or cooperate, form coalitions, and share resources to keep the whole system alive.

Pure self-interest leads to collapse. Blind cooperation gets exploited. The winning strategy
requires **theory-of-mind reasoning**: understanding what other agents need, when to propose
alliances, and when to invest vs defend.

This is **District Accord** — a multi-agent RL environment built on OpenEnv that trains LLMs
to navigate exactly this kind of complex social dilemma.

---

## The Environment

| Property | Value |
|---|---|
| Agents | 12 districts |
| Episode length | 100 turns |
| Action space | 9 structured text actions |
| Observation | Dict + flat vector (resources, stability, trust, crisis) |
| Themes | Multi-agent, Long-horizon, Self-play |

**Each turn, an agent observes:**
- Its own resources, stability, crisis exposure
- Aggregated peer states (resources, stability, trust levels)
- Current global crisis level and turn progress
- A dynamic action mask (9 binary values — only valid actions are 1)

**And produces one action (as natural language):**
```
invest | defend | recover | ignore | propose | accept | reject | share | request_aid
```

### The Core Challenge

No single district can stop a global crisis alone. Coalitions are emergent — they only happen
when agents learn to trust each other and coordinate. The environment has three interlocking
systems that make this hard to exploit:

- **Trust System** — a `[-1, 1]` matrix that updates based on cooperation/defection history
- **Negotiation System** — proposal lifecycle with TTL, cooldowns, anti-spam constraints
- **Reward Engine** — 8 independent reward components (survival, coalition bonus, stability
  delta, mask penalty, spam penalty, etc.) — hard to game a single signal

---

## Training with GRPO

We trained `Qwen2.5-1.5B-Instruct` using **GRPO** (Group Relative Policy Optimization)
via TRL + Unsloth. One LLM agent plays as District 0 against 11 rule-based opponents.

### The RL Loop

```
1. Observation → formatted as natural language prompt
2. LLM generates action string (e.g. "defend", "invest", "propose")
3. ActionParser converts text → ParsedAction
4. env.step(actions) returns reward
5. GRPO scores 4 completions per prompt, updates model toward higher-reward actions
```

### Reward Function (5 components)

```python
score += 0.5   # valid action
score += 0.2   # clean format (single token)
score += 0.4   # defend/recover when crisis > 0.4
score += 0.3   # propose/accept/share (cooperative bonus)
score += 0.3   # recover when stability < 0.3 (emergency)
score -= 0.2   # ignore when action is available (passive penalty)
score -= 1.0   # invalid action (hard gate)
```

---

## Results

### Reward Curve

![Training Curves](outputs/training_curves.png)

*GRPO reward climbs from ~0.3 (random baseline) to ~1.0 within one training run,
with loss converging over 1000 steps.*

### Policy Comparison

![Baseline Comparison](outputs/baseline_comparison.png)

| Policy | Avg Reward/Turn | Turns Survived | Collapses |
|---|---|---|---|
| Random | 0.397 | 71/100 | 12/12 |
| Mask-Aware Random | 0.939 | 100/100 | 2/12 |
| Rule-Based | 1.002 | 100/100 | 0/12 |
| **Trained LLM (GRPO)** | **~0.95** | **100/100** | **0/12** |

### What the LLM Learned

Before training, the model outputs random tokens — often invalid actions, causing immediate
collapse. After GRPO training, it learns:

- Output `defend` or `recover` when crisis is high
- Output `invest` when resources are stable and crisis is low
- Output `propose` / `accept` to form coalitions early in the episode
- Never output invalid actions (format compliance ~100%)

---

## Anti-Hacking Measures

The environment is exploit-resistant by design:

| Exploit | Mitigation |
|---|---|
| Trust farming | TTL + cooldown per proposal pair |
| Coalition idling | Stability drain still applies inside coalitions |
| Defend-only greedy | Mixed strategies mathematically dominate |
| Action flooding | Mask enforcement + spam penalty |

---

## Links

- 🌍 **HuggingFace Space (live API):** https://huggingface.co/spaces/tezivindh1/district-accord
- 📓 **Colab Training Notebook:** https://colab.research.google.com/drive/1cmW1jtnx8WgGfrKSc2JlCz-i0X5rjkh6
- 💻 **GitHub Repository:** https://github.com/tezivindh/meta-env-hackathon-final

---

*Built with OpenEnv · TRL · Unsloth · HuggingFace Spaces · FastAPI*
