from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from typer.testing import CliRunner

from victor.ui.cli import app


def _strip_ansi(text: str) -> str:
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


class _FakeAgent:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self._skill_matcher = None


class _FakeVictorClient:
    last_config = None
    last_agent = None

    def __init__(self, config):
        type(self).last_config = config
        type(self).last_agent = _FakeAgent()

    async def initialize(self):
        return type(self).last_agent

    async def start_embedding_preload(self):
        return None

    async def get_session_metrics(self):
        return {"tool_calls": 1}

    async def stream_chat(self, message: str):
        type(self).last_agent.messages.append(message)
        yield SimpleNamespace(
            content="",
            metadata={
                "tool_start": {
                    "name": "database",
                    "arguments": {
                        "action": "query",
                        "sql": (
                            "SELECT generation, text_hash "
                            "FROM agent_prompt_candidate ORDER BY generation"
                        ),
                    },
                }
            },
        )
        yield SimpleNamespace(
            content="",
            metadata={
                "tool_result": {
                    "name": "database",
                    "success": True,
                    "elapsed": 0.02,
                    "arguments": {"action": "query"},
                    "result": "generation | text_hash\n1 | hash_gen_1\n2 | hash_gen_2",
                }
            },
        )
        yield SimpleNamespace(
            content=(
                "I queried agent_prompt_candidate directly and found generations 1 and 2. "
                "Generation 2 is the active evolved candidate."
            ),
            metadata={},
        )

    async def close(self):
        return None


def test_chat_oneshot_renders_database_tool_flow_for_explicit_sqllite_request(monkeypatch):
    monkeypatch.setattr("victor.ui.commands.chat.VictorClient", _FakeVictorClient)
    monkeypatch.setattr("victor.ui.commands.chat.graceful_shutdown", AsyncMock())

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "chat",
            "--provider",
            "ollama",
            "--model",
            "mistral-tools:7b-instruct",
            "--no-stream",
            "also review the sqllite db for evolved prompts and explain significance",
        ],
    )

    rendered = _strip_ansi(result.stdout)

    assert result.exit_code == 0, rendered
    assert "database" in rendered
    assert "agent_prompt_candidate" in rendered
    assert "Generation 2 is the active evolved candidate." in rendered
    assert _FakeVictorClient.last_agent is not None
    assert _FakeVictorClient.last_agent.messages == [
        "also review the sqllite db for evolved prompts and explain significance"
    ]
    assert _FakeVictorClient.last_config.tool_budget is None
