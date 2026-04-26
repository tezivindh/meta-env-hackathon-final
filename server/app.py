"""
district_accord/server/app.py
==============================
FastAPI server exposing the District Accord environment
as an OpenEnv-compatible HTTP API.

Endpoints:
    GET  /health  → {"status": "ok"}
    POST /reset   → reset env, return initial observations
    POST /step    → advance env by one turn, return (obs, rewards, done, info)
    GET  /state   → return current env snapshot
"""

from typing import Any, Dict, Optional

import enum
import os

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from district_accord.env import DistrictAccordEnv
from district_accord.spaces.action_parser import ActionParser

app = FastAPI(title="District Accord OpenEnv")


# ---------------------------------------------------------------------------
# Web interface (served at /web for HuggingFace Spaces)
# ---------------------------------------------------------------------------
WEB_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>District Accord — OpenEnv</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;display:flex;align-items:center;justify-content:center}
.card{background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%);border:1px solid #334155;border-radius:16px;padding:48px;max-width:720px;width:90%;box-shadow:0 25px 50px rgba(0,0,0,.5)}
h1{font-size:2rem;background:linear-gradient(90deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px}
.subtitle{color:#94a3b8;font-size:1.05rem;margin-bottom:32px}
.badges{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:28px}
.badge{padding:6px 14px;border-radius:20px;font-size:.8rem;font-weight:600}
.b1{background:#1e3a5f;color:#60a5fa}.b2{background:#312e81;color:#a78bfa}.b3{background:#1a3329;color:#6ee7b7}
table{width:100%;border-collapse:collapse;margin-bottom:24px}
th{text-align:left;color:#94a3b8;font-size:.8rem;text-transform:uppercase;letter-spacing:.05em;padding:8px 12px;border-bottom:1px solid #334155}
td{padding:10px 12px;border-bottom:1px solid #1e293b;font-size:.95rem}
.mono{font-family:'SF Mono',monospace;color:#60a5fa;font-size:.85rem}
.endpoints{margin-top:24px}
.ep{display:flex;align-items:center;gap:12px;padding:10px 16px;background:#1e293b;border-radius:10px;margin-bottom:8px}
.method{font-weight:700;font-size:.75rem;padding:4px 10px;border-radius:6px;min-width:52px;text-align:center}
.get{background:#065f46;color:#6ee7b7}.post{background:#1e3a5f;color:#60a5fa}
.path{font-family:monospace;color:#f1f5f9;font-size:.9rem}
.foot{margin-top:28px;text-align:center;color:#475569;font-size:.8rem}
</style>
</head>
<body>
<div class="card">
  <h1>🌍 District Accord</h1>
  <p class="subtitle">Multi-Agent RL Environment for Complex Social Dilemmas</p>
  <div class="badges">
    <span class="badge b1">multi-agent-interactions</span>
    <span class="badge b2">long-horizon-planning</span>
    <span class="badge b3">self-improvement</span>
  </div>
  <table>
    <tr><th>Property</th><th>Value</th></tr>
    <tr><td>Agents</td><td class="mono">12 districts</td></tr>
    <tr><td>Episode Length</td><td class="mono">100 turns</td></tr>
    <tr><td>Action Space</td><td class="mono">9 actions (structured text)</td></tr>
    <tr><td>Observation</td><td class="mono">dict + flat vector (4N+4)</td></tr>
    <tr><td>Version</td><td class="mono">0.1.0 / Phase 6</td></tr>
  </table>
  <div class="endpoints">
    <div class="ep"><span class="method get">GET</span><span class="path">/health</span></div>
    <div class="ep"><span class="method post">POST</span><span class="path">/reset</span></div>
    <div class="ep"><span class="method post">POST</span><span class="path">/step</span></div>
    <div class="ep"><span class="method get">GET</span><span class="path">/state</span></div>
  </div>
  <p class="foot">OpenEnv &middot; Powered by FastAPI &middot; Running on HuggingFace Spaces</p>
</div>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def root():
    """Root route — serves the web dashboard."""
    return WEB_HTML


@app.get("/web", response_class=HTMLResponse)
def web_interface():
    """Web dashboard served at /web for HuggingFace Spaces."""
    return WEB_HTML

# ---------------------------------------------------------------------------
# Global instances
# ---------------------------------------------------------------------------
global_env: Optional[DistrictAccordEnv] = None
global_parser: Optional[ActionParser] = None


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    seed: Optional[int] = None


class StepRequest(BaseModel):
    actions: Dict[str, str]


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------
def to_json_serializable(obj: Any) -> Any:
    """Recursively cast Numpy types and objects to standard JSON natives."""
    if isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    elif isinstance(obj, enum.Enum):
        return obj.value
    elif hasattr(obj, "__dict__"):
        return to_json_serializable(obj.__dict__)
    return obj


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest) -> Dict[str, Any]:
    global global_env, global_parser
    try:
        if global_env is None:
            global_env = DistrictAccordEnv()
            global_parser = ActionParser(global_env.config)

        obs = global_env.reset(seed=req.seed)
        return {
            "obs": to_json_serializable(obs),
            "info": {"message": "Environment reset successful."},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(req: StepRequest) -> Dict[str, Any]:
    global global_env, global_parser

    if global_env is None or global_parser is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first.",
        )
    if global_env._done:
        raise HTTPException(
            status_code=400,
            detail="Episode is finished. Please call /reset.",
        )

    try:
        # Convert str keys from JSON back to int AgentIDs.
        raw_actions: Dict[int, str] = {}
        for k, v in req.actions.items():
            try:
                agent_id = int(k)
                raw_actions[agent_id] = v
            except ValueError:
                pass

        parsed_actions = global_parser.parse_structured_safe(raw_actions)
        obs, rewards, done, truncated, info = global_env.step(parsed_actions)

        return {
            "obs": to_json_serializable(obs),
            "rewards": to_json_serializable(rewards),
            "done": to_json_serializable(done),
            "truncated": to_json_serializable(truncated),
            "info": to_json_serializable(info),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state() -> Dict[str, Any]:
    global global_env
    if global_env is None:
        raise HTTPException(status_code=400, detail="Environment not loaded.")

    try:
        return to_json_serializable(
            {
                "turn": global_env.turn,
                "districts": global_env.districts,
                "done": global_env._done,
                "crisis": {"level": global_env.crisis.crisis_level},
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Entry point — used by `project.scripts` in pyproject.toml
# ---------------------------------------------------------------------------
def main() -> None:
    """Launch the uvicorn server (called via `uv run --project . server`)."""
    uvicorn.run(
        "district_accord.server.app:app",
        host="0.0.0.0",
        port=7860,
    )


if __name__ == "__main__":
    main()
