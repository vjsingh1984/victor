from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from victor.framework.session_config import SessionConfig
from victor.integrations.api.routes.config_routes import create_router


class _Server:
    def __init__(self) -> None:
        self._settings = SimpleNamespace(
            provider=SimpleNamespace(default_provider="ollama", default_model="qwen"),
            load_profiles=lambda: {
                "default": SimpleNamespace(
                    provider="ollama",
                    model="qwen",
                    description="local",
                ),
                "cloud": SimpleNamespace(
                    provider="anthropic",
                    model="claude",
                    description="cloud",
                ),
            },
        )
        self._session_config = SessionConfig.from_cli_flags(agent_profile="default")
        self.updates: list[SessionConfig] = []

    async def update_session_config(self, session_config: SessionConfig) -> None:
        self._session_config = session_config
        self.updates.append(session_config)


@pytest.fixture
def client_and_server():
    server = _Server()
    app = FastAPI()
    app.include_router(create_router(server))  # type: ignore[arg-type]
    return TestClient(app), server


def test_effective_config_profiles_and_modes(client_and_server) -> None:
    client, _server = client_and_server

    response = client.get("/config/effective")

    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"] == "default"
    assert payload["provider"] == "ollama"
    assert {mode["name"] for mode in payload["modes"]} >= {
        "build",
        "review",
        "delegate",
    }
    assert {profile["name"] for profile in payload["profiles"]} == {"default", "cloud"}


def test_switch_mode_updates_server_session_config(client_and_server) -> None:
    client, server = client_and_server

    response = client.post("/mode/switch", json={"mode": "review"})

    assert response.status_code == 200
    assert response.json() == {"success": True, "mode": "review"}
    assert server.updates[-1].mode == "review"


def test_switch_model_updates_provider_override(client_and_server) -> None:
    client, server = client_and_server

    response = client.post(
        "/model/switch", json={"provider": "openai", "model": "gpt-4o"}
    )

    assert response.status_code == 200
    assert server.updates[-1].provider_override.provider == "openai"
    assert server.updates[-1].provider_override.model == "gpt-4o"


def test_switch_profile_updates_session_profile_and_clears_provider_override(
    client_and_server,
) -> None:
    client, server = client_and_server
    server._session_config = SessionConfig.from_cli_flags(
        agent_profile="default",
        provider="openai",
        model="gpt-4o",
    )

    response = client.post("/profile/switch", json={"profile": "cloud"})

    assert response.status_code == 200
    assert response.json() == {"success": True, "profile": "cloud"}
    assert server.updates[-1].agent_profile == "cloud"
    assert not server.updates[-1].provider_override.is_active


def test_switch_profile_rejects_unknown_profile(client_and_server) -> None:
    client, server = client_and_server

    response = client.post("/profile/switch", json={"profile": "missing"})

    assert response.status_code == 404
    assert response.json()["success"] is False
    assert server.updates == []
