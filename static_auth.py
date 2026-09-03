"""POC-only custom auth: resolve a fake API key to a LiteLLM identity.

Stands in for LiteLLM virtual keys so the POC runs without Postgres. The hook
only ever reads `key_alias` / `user_id` / `team_id` off UserAPIKeyAuth, which is
exactly what real virtual keys populate -- so swapping this out for a database
changes nothing downstream.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import Request
from litellm.proxy._types import UserAPIKeyAuth

KEYS_PATH = Path(__file__).resolve().parent / "poc_keys.yaml"


async def user_api_key_auth(request: Request, api_key: str) -> UserAPIKeyAuth:
    token = api_key.removeprefix("Bearer ").strip()
    keys = yaml.safe_load(KEYS_PATH.read_text()) or {}
    identity = keys.get(token)
    if identity is None:
        raise Exception(f"unknown api key: {token[:12]}...")
    return UserAPIKeyAuth(api_key=token, **identity)
