"""
client.py — OpenEnv client for District Accord.

Provides a typed async/sync client for interacting with the
District Accord environment server.

Usage (async):
    async with DistrictAccordClient(base_url="ws://localhost:7860") as env:
        result = await env.reset(seed=42)
        result = await env.step({"actions": {"0": "invest", "1": "defend"}})

Usage (sync):
    env = DistrictAccordClient(base_url="ws://localhost:7860").sync()
    with env:
        result = env.reset(seed=42)
        result = env.step({"actions": {"0": "invest", "1": "defend"}})
"""

from typing import Any, Dict

from openenv import GenericEnvClient


class DistrictAccordClient(GenericEnvClient):
    """
    Client for interacting with the District Accord environment server.

    Inherits from GenericEnvClient which works with raw dictionaries
    for actions and observations.

    Actions format:
        {
            "actions": {
                "0": "invest",
                "1": "defend",
                "2": "share:target=0,amount=0.1",
                ...
            }
        }

    Valid action strings:
        invest, defend, ignore, recover, request_aid, share,
        propose, accept, reject

    Structured format: "<action>:<key>=<value>,..."
        e.g. "share:target=2,amount=0.1", "propose:target=1"
    """

    pass
