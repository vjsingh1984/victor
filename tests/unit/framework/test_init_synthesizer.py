# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for InitSynthesizer — TDD."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.framework.init_synthesizer import InitSynthesizer


class TestInitSynthesizer:
    """Core synthesize() method tests."""

    @pytest.mark.asyncio
    async def test_synthesize_with_agent_reuses_orchestrator(self):
        """When agent is provided, synthesize() uses agent.chat() — no new Agent created."""
        mock_agent = AsyncMock()
        mock_agent.chat.return_value = MagicMock(content="# init.md\n\nProject overview.")

        synthesizer = InitSynthesizer()
        result = await synthesizer.synthesize("raw data here", agent=mock_agent)

        mock_agent.chat.assert_called_once()
        assert "init.md" in mock_agent.chat.call_args[0][0]  # Prompt contains template
        assert result == "# init.md\n\nProject overview."

    @pytest.mark.asyncio
    async def test_synthesize_without_agent_creates_fresh(self):
        """When no agent, synthesize() uses Agent.create() + run()."""
        mock_result = MagicMock(success=True, content="# Generated init.md")

        with patch("victor.framework.agent.Agent") as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.create = AsyncMock(return_value=mock_agent_instance)

            synthesizer = InitSynthesizer()
            result = await synthesizer.synthesize(
                "raw data", provider="ollama", model="qwen3:8b"
            )

            MockAgent.create.assert_called_once()
            mock_agent_instance.run.assert_called_once()
            assert result == "# Generated init.md"

    @pytest.mark.asyncio
    async def test_synthesize_cleans_code_fences(self):
        """Output wrapped in ``` gets cleaned."""
        mock_agent = AsyncMock()
        mock_agent.chat.return_value = MagicMock(
            content="```markdown\n# Project\n\nOverview.\n```"
        )

        synthesizer = InitSynthesizer()
        result = await synthesizer.synthesize("raw data", agent=mock_agent)

        assert result == "# Project\n\nOverview."
        assert "```" not in result

    @pytest.mark.asyncio
    async def test_synthesize_prompt_contains_base_content(self):
        """The synthesis prompt includes the base_content."""
        mock_agent = AsyncMock()
        mock_agent.chat.return_value = MagicMock(content="# Result")

        synthesizer = InitSynthesizer()
        await synthesizer.synthesize("UNIQUE_BASE_DATA_12345", agent=mock_agent)

        prompt = mock_agent.chat.call_args[0][0]
        assert "UNIQUE_BASE_DATA_12345" in prompt

    @pytest.mark.asyncio
    async def test_synthesize_handles_failure_gracefully(self):
        """If agent.chat() raises, return empty string."""
        mock_agent = AsyncMock()
        mock_agent.chat.side_effect = RuntimeError("LLM failed")

        synthesizer = InitSynthesizer()
        result = await synthesizer.synthesize("data", agent=mock_agent)

        assert result == ""


class TestInitSynthesizerToolsFallback:
    """synthesize_with_tools() fallback mode."""

    @pytest.mark.asyncio
    async def test_tools_fallback_with_agent(self):
        """Fallback uses agent.chat() with tools prompt."""
        mock_agent = AsyncMock()
        mock_agent.chat.return_value = MagicMock(content="# Fallback init.md")

        synthesizer = InitSynthesizer()
        result = await synthesizer.synthesize_with_tools(agent=mock_agent)

        assert result == "# Fallback init.md"
        prompt = mock_agent.chat.call_args[0][0]
        assert "overview" in prompt.lower() or "analyze" in prompt.lower()

    @pytest.mark.asyncio
    async def test_tools_fallback_without_agent(self):
        """Fallback creates Agent with vertical=coding for tool access."""
        mock_result = MagicMock(success=True, content="# Tools init.md")

        with patch("victor.framework.agent.Agent") as MockAgent:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            MockAgent.create = AsyncMock(return_value=mock_agent_instance)

            synthesizer = InitSynthesizer()
            result = await synthesizer.synthesize_with_tools(
                provider="ollama", model="qwen3:8b"
            )

            # Should create with vertical="coding" for tool access
            create_kwargs = MockAgent.create.call_args[1]
            assert create_kwargs.get("vertical") == "coding"
            assert result == "# Tools init.md"
