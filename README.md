# District Accord

> **A Multi-Agent Reinforcement Learning Environment for Complex Social Dilemmas**

![Environment Status](https://img.shields.io/badge/Status-Phase_6_Training-blue.svg)
![Tests](https://img.shields.io/badge/Tests-356%20Passed-success.svg)

## 🌍 The Problem (Motivation)
In real-world crises—whether climate emergencies, economic downturns, or localized disasters—autonomous districts or nations must balance **self-preservation** with **collective survival**. 

**District Accord** is designed as a rigorous MARL (Multi-Agent Reinforcement Learning) testbed to simulate these high-stakes dynamics. In this environment:
- Pure self-interest rapidly leads to systemic collapse as global crisis levels rise unchecked.
- Naive cooperation leaves agents vulnerable to exploitation by free-riders (the "Defend Alone" dilemma).
- Successful policies must learn to dynamically negotiate, build trust, form coalitions, and distribute resources under the pressure of severe TTL-based (Time-to-Live) actions.

The goal is to study how RL agents learn advanced negotiation, diplomacy, and trust-based strategies in a non-zero-sum game.

---

## ⚙️ How the Environment Works
District Accord provides a compliant, deterministic interface modeled after standard RL suites (OpenAI Gym/PettingZoo) supporting up to 12 agents operating over 100 turns.

### 1. Action Space
Agents submit structured actions per turn to navigate their economy and diplomacy:
- **Economy**: `INVEST` (grow resources), `DEFEND` (lower crisis exposure), `RECOVER` (boost stability)
- **Diplomacy**: `PROPOSE_COALITION`, `ACCEPT_COALITION`, `REJECT_COALITION`
- **Resource Management**: `REQUEST_AID`, `SHARE_RESOURCES`

### 2. Observation Space
Agents receive a highly structured observation dict and flattened vectors containing:
- **Self State**: Current resources, stability, crisis exposure, and stability delta.
- **Peer State**: Visible resources, stability, exact trust metrics representing historical cooperations, and coalition flags of all other N-1 agents.
- **Global State**: Overall crisis level and episode progress.
- **Action Constraints**: Dynamic action masks (e.g., cannot accept a proposal if none exist).

### 3. Core Engine Subsystems
- **Negotiation & Coalition System:** Manages propose/accept/reject lifecycles natively within the environment to prevent prompt spam.
- **Trust System:** A bounded matrix `[-1, 1]` dynamically reacting to shared resources and rejected proposals.
- **Reward Engine:** A carefully balanced, stateless calculator driving agents toward survival, cooperation, and stability, with built-in caps to prevent exploit farming.
- **Event Bus & State Tracker:** Maintains a comprehensive, perfectly deterministic replay log of all interactions per turn for deep post-episode analysis.

---

## 📊 Performance & Results
Rigorous exploit testing and system auditing demonstrate the environment's RL readiness:

1. **Exploits Mitigated:**
   - *Trust Farming:* Pure trust signaling and cyclic proposing decays naturally. The environment caps negotiation loops via built-in cooldowns and TTL constraints.
   - *Greedy Survival:* Running "Defend Alone" continuously is statistically dominated by Mixed Cooperative strategies.
   - *Coalition Idling:* Forming a coalition but failing to act results in starvation/collapse.
2. **Strategy Dominance:**
   - Benchmarks show that **Mixed Strategies (Cooperate + Stabilize)** mathematically win over isolated, self-interested agents across diverse crisis seeds.
3. **Robustness:**
   - 100% deterministic test suite coverage (350+ tests passing). 
   - State pipelines verify that identical seeds strictly enforce identical event sequences regardless of hardware.

---

## 🚀 Quick Start

### Installation
Requires Python 3.10+
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Running Tests
The suite covers all 6 phases of development:
```bash
pytest tests/ -v
```

### Diagnostics & Trace
To run a randomized agent trace or check environment health:
```bash
python examples/episode_trace.py
python examples/diagnostic.py
python examples/exploit_tests.py
```
