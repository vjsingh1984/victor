"""Unit tests for the Microsoft Teams HITL transport.

Covers the Entra (Azure AD) client-credentials delivery path, the legacy
incoming-webhook fallback, token caching, and the unconfigured error. A fake
``aiohttp`` module is injected so the tests never touch the network and do not
require aiohttp to be installed.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any, Dict, List

import pytest

from victor.workflows.hitl import HITLNodeType, HITLRequest


class _FakeResponse:
    def __init__(self, status: int, payload: Dict[str, Any]) -> None:
        self.status = status
        self._payload = payload

    async def json(self) -> Dict[str, Any]:
        return self._payload

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _FakeSession:
    """Records every POST and returns canned token / Graph / webhook responses."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def _session_factory(self, *args: Any, **kwargs: Any) -> "_FakeSession":
        return self

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def post(self, url, data=None, json=None, headers=None):  # noqa: A002 - mirror aiohttp API
        self.calls.append({"url": url, "data": data, "json": json, "headers": headers})
        if "oauth2/v2.0/token" in url:
            return _FakeResponse(200, {"access_token": "FAKE_TOKEN", "expires_in": 3600})
        if "graph.microsoft.com" in url:
            return _FakeResponse(201, {"id": "graph-msg-123"})
        return _FakeResponse(200, {})


@pytest.fixture()
def fake_aiohttp(monkeypatch: pytest.MonkeyPatch) -> _FakeSession:
    session = _FakeSession()
    module = types.ModuleType("aiohttp")
    module.ClientSession = session._session_factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aiohttp", module)
    return session


@pytest.fixture()
def request_obj() -> HITLRequest:
    return HITLRequest(
        request_id="abc123def456ghi",
        node_id="n1",
        hitl_type=HITLNodeType.APPROVAL,
        prompt="Deploy to prod?",
        context={"env": "prod", "change": "#277"},
        timeout=300,
    )


async def test_graph_path_uses_client_credentials_then_bearer(fake_aiohttp, request_obj):
    from victor.workflows.hitl_transports import TeamsConfig, TeamsTransport

    cfg = TeamsConfig(
        tenant_id="tenantA",
        client_id="C",
        client_secret="S",
        team_id="team1",
        channel_id="chan1",
        callback_url="https://cb.example",
    )
    transport = TeamsTransport(cfg)

    ref = await transport.send(request_obj, "wf-1")

    token_call = next(c for c in fake_aiohttp.calls if "oauth2" in c["url"])
    graph_call = next(c for c in fake_aiohttp.calls if "graph.microsoft" in c["url"])

    # tenant_id only scopes the token URL; the secret is what authenticates.
    assert "tenantA/oauth2/v2.0/token" in token_call["url"]
    assert token_call["data"]["grant_type"] == "client_credentials"
    assert token_call["data"]["client_secret"] == "S"
    assert token_call["data"]["scope"] == "https://graph.microsoft.com/.default"
    # Graph call carries the bearer token, never the tenant_id as a credential.
    assert graph_call["headers"]["Authorization"] == "Bearer FAKE_TOKEN"
    assert "teams/team1/channels/chan1/messages" in graph_call["url"]
    # An adaptive card is attached.
    attachment = graph_call["json"]["attachments"][0]
    assert attachment["contentType"] == "application/vnd.microsoft.card.adaptive"
    assert json.loads(attachment["content"])["type"] == "AdaptiveCard"
    assert ref == "graph-msg-123"


async def test_token_is_cached_across_sends(fake_aiohttp, request_obj):
    from victor.workflows.hitl_transports import TeamsConfig, TeamsTransport

    transport = TeamsTransport(
        TeamsConfig(tenant_id="T", client_id="C", client_secret="S", team_id="t", channel_id="c")
    )
    await transport.send(request_obj, "wf")
    await transport.send(request_obj, "wf")

    token_calls = [c for c in fake_aiohttp.calls if "oauth2" in c["url"]]
    assert len(token_calls) == 1  # second send reuses the cached token


async def test_webhook_fallback_does_not_authenticate(fake_aiohttp, request_obj):
    from victor.workflows.hitl_transports import TeamsConfig, TeamsTransport

    transport = TeamsTransport(TeamsConfig(webhook_url="https://wh.example/x"))
    ref = await transport.send(request_obj, "wf-2")

    assert fake_aiohttp.calls[0]["url"] == "https://wh.example/x"
    assert "oauth2" not in json.dumps(fake_aiohttp.calls)  # no token acquisition
    assert ref == request_obj.request_id


async def test_unconfigured_raises(fake_aiohttp, request_obj):
    from victor.workflows.hitl_transports import TeamsConfig, TeamsTransport

    with pytest.raises(ValueError, match="not configured"):
        await TeamsTransport(TeamsConfig()).send(request_obj, "wf-3")


def test_teams_transport_is_registered():
    from victor.workflows.hitl import HITLMode
    from victor.workflows.hitl_transports import TeamsTransport, get_transport

    transport = get_transport(HITLMode.TEAMS, TeamsConfig_none())
    assert isinstance(transport, TeamsTransport)


def TeamsConfig_none():
    from victor.workflows.hitl_transports import TeamsConfig

    return TeamsConfig()
