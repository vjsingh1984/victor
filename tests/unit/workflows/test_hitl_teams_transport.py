"""Unit tests for the Microsoft Teams HITL transport.

Token acquisition is injected as a ``TokenCredential`` (Dependency Inversion),
so these tests inject a fake credential rather than mocking the Entra token HTTP
(the token flow itself is covered in tests/unit/core/identity). A fake
``aiohttp`` module is injected for the Graph/webhook POST so the tests never
touch the network and don't require aiohttp to be installed."""
from __future__ import annotations

import json
import sys
import types
from typing import Any, Dict, List

import pytest

from victor.core.identity import AccessToken
=======
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
    """Records every POST and returns canned Graph / webhook responses."""

>>>>>>> b09eb5ddf (feat(identity): provider-agnostic TokenCredential layer; inject into Teams auth)
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
        if "graph.microsoft.com" in url:
            return _FakeResponse(201, {"id": "graph-msg-123"})
        return _FakeResponse(200, {})  # webhook
<<<<<<< HEAD

class _FakeCredential:
    """A TokenCredential that hands back a fixed token and records the scope."""

    def __init__(self, token: str = "FAKE_TOKEN") -> None:
        self.token = token
        self.calls = 0
        self.scopes: tuple = ()

    async def get_token(self, *scopes: str) -> AccessToken:
        self.calls += 1
        self.scopes = scopes
        return AccessToken(token=self.token, expires_on=2_000_000_000.0)

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


async def test_graph_path_uses_injected_credential_token(fake_aiohttp, request_obj):
    from victor.workflows.hitl_transports import TeamsConfig, TeamsTransport

    cred = _FakeCredential("INJECTED_TOKEN")
    cfg = TeamsConfig(team_id="team1", channel_id="chan1", callback_url="https://cb.example")
    transport = TeamsTransport(cfg, credential=cred)

    ref = await transport.send(request_obj, "wf-1")

    graph_call = next(c for c in fake_aiohttp.calls if "graph.microsoft" in c["url"])
    # The token came from the injected credential, scoped to Graph.
    assert cred.calls == 1
    assert cred.scopes == ("https://graph.microsoft.com/.default",)
    assert graph_call["headers"]["Authorization"] == "Bearer INJECTED_TOKEN"
    assert "teams/team1/channels/chan1/messages" in graph_call["url"]    attachment = graph_call["json"]["attachments"][0]    assert attachment["contentType"] == "application/vnd.microsoft.card.adaptive"
    assert json.loads(attachment["content"])["type"] == "AdaptiveCard"
    assert ref == "graph-msg-123"


async def test_credential_built_from_config_when_not_injected(fake_aiohttp, request_obj):
    from victor.workflows.hitl_transports import TeamsConfig, TeamsTransport

    # No credential injected, but Entra config present -> one is built lazily.
    cfg = TeamsConfig(tenant_id="T", client_id="C", client_secret="S", team_id="t", channel_id="c")    transport = TeamsTransport(cfg)
    assert transport._resolve_credential() is not None  # built from config
    assert transport._can_use_graph() is True


async def test_webhook_fallback_when_no_credential(fake_aiohttp, request_obj):    from victor.workflows.hitl_transports import TeamsConfig, TeamsTransport
    transport = TeamsTransport(TeamsConfig(webhook_url="https://wh.example/x"))
    ref = await transport.send(request_obj, "wf-2")

    assert fake_aiohttp.calls[0]["url"] == "https://wh.example/x"
=======
    assert "oauth2" not in json.dumps(fake_aiohttp.calls)  # no token acquisition
>>>>>>> 5f96e6ad8 (feat(hitl): authenticate Teams transport via Entra client-credentials)