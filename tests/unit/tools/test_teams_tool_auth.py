"""Auth-resolution tests for the Teams tool.

Verifies the tool mints its own Microsoft Graph token from the environment
(Entra client-credentials) when no token is injected, prefers an injected
context token, and reports availability from either source.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict

import pytest

from victor.tools import teams_tool

_AZURE_VARS = (
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "TEAMS_TENANT_ID",
    "TEAMS_CLIENT_ID",
    "TEAMS_CLIENT_SECRET",
)


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch):
    for var in _AZURE_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _install_token_aiohttp(monkeypatch, token: str = "ENV_TOKEN"):
    class _Resp:
        status = 200

        async def json(self) -> Dict[str, Any]:
            return {"access_token": token, "expires_in": 3600}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Session:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def post(self, url, data=None, **k):
            return _Resp()

    module = types.ModuleType("aiohttp")
    module.ClientSession = _Session
    monkeypatch.setitem(sys.modules, "aiohttp", module)


def test_is_teams_configured_false_when_nothing(clean_env):
    assert teams_tool.is_teams_configured({}) is False
    assert teams_tool.is_teams_configured(None) is False


def test_is_teams_configured_true_with_context_token(clean_env):
    assert teams_tool.is_teams_configured({"teams_access_token": "abc"}) is True


def test_is_teams_configured_true_with_env_aliases(clean_env):
    clean_env.setenv("TEAMS_TENANT_ID", "t")
    clean_env.setenv("TEAMS_CLIENT_ID", "c")
    clean_env.setenv("TEAMS_CLIENT_SECRET", "s")
    assert teams_tool.is_teams_configured({}) is True


async def test_resolve_prefers_context_token(clean_env):
    # Even with env creds set, an injected context token wins (no minting).
    clean_env.setenv("AZURE_TENANT_ID", "t")
    clean_env.setenv("AZURE_CLIENT_ID", "c")
    clean_env.setenv("AZURE_CLIENT_SECRET", "s")
    token = await teams_tool._resolve_access_token({"teams_access_token": "ctx-token"})
    assert token == "ctx-token"


async def test_resolve_mints_from_env_when_no_context_token(clean_env):
    _install_token_aiohttp(clean_env, token="MINTED")
    clean_env.setenv("AZURE_TENANT_ID", "t")
    clean_env.setenv("AZURE_CLIENT_ID", "c")
    clean_env.setenv("AZURE_CLIENT_SECRET", "s")
    token = await teams_tool._resolve_access_token({})
    assert token == "MINTED"


async def test_resolve_returns_none_when_unconfigured(clean_env):
    assert await teams_tool._resolve_access_token({}) is None
