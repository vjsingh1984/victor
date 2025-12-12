import importlib
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def server_module(monkeypatch):
    # Enforce API key for these tests
    # Note: pydantic-settings expects the env var name to match the field name (uppercase)
    monkeypatch.setenv("SERVER_API_KEY", "secret")
    import web.server.main as main

    importlib.reload(main)
    yield main

    main.SESSION_AGENTS.clear()
    main.SESSION_TOKENS.clear()


@pytest.fixture
def client(server_module):
    return TestClient(server_module.app)


def test_health_requires_auth(client):
    resp = client.get("/health")
    assert resp.status_code == 401

    resp = client.get("/health", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_render_requires_auth_and_size_limit(monkeypatch, client, server_module):
    # Set small limit to force 413
    monkeypatch.setattr(server_module, "RENDER_MAX_BYTES", 10)

    # Unauthorized should be blocked - must send content-type for FastAPI Body validation
    resp = client.post(
        "/render/plantuml",
        content="@startuml\nA-->B\n@enduml",
        headers={"Content-Type": "text/plain"},
    )
    assert resp.status_code == 401

    # Authorized but too large should return 413
    payload = "x" * 50
    resp = client.post(
        "/render/plantuml",
        content=payload,
        headers={"Authorization": "Bearer secret", "Content-Type": "text/plain"},
    )
    assert resp.status_code == 413
