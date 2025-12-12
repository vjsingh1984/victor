import importlib
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def server_module(monkeypatch):
    # Ensure API key is set before module import so dependency enforces auth
    # Note: pydantic-settings expects the env var name to match the field name (uppercase)
    monkeypatch.setenv("SERVER_API_KEY", "secret")
    # Reload module to pick up env changes
    import web.server.main as main

    importlib.reload(main)
    yield main

    # Cleanup global state
    main.SESSION_AGENTS.clear()
    main.SESSION_TOKENS.clear()


@pytest.fixture
def client(server_module):
    return TestClient(server_module.app)


def test_issue_session_token_requires_auth(client, server_module):
    # Missing API key should be rejected
    resp = client.post("/session/token", json={})
    assert resp.status_code == 401

    # Correct API key should succeed
    resp = client.post(
        "/session/token",
        headers={"Authorization": "Bearer secret"},
        json={},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "session_token" in data
    assert "session_id" in data


def test_reissue_unknown_session_404(client, server_module):
    resp = client.post(
        "/session/token",
        headers={"Authorization": "Bearer secret"},
        json={"session_id": "does-not-exist"},
    )
    assert resp.status_code == 404


def test_issue_session_token_respects_max_sessions(monkeypatch, client, server_module):
    # Force max sessions to 0 to trigger 429
    monkeypatch.setattr(server_module, "MAX_SESSIONS", 0)

    resp = client.post(
        "/session/token",
        headers={"Authorization": "Bearer secret"},
        json={},
    )
    assert resp.status_code == 429
