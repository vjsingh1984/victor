# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for InitSynthesizer — TDD."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.framework.init_synthesizer import (
    InitSynthesizer,
    SYNTHESIS_PROMPT,
    _build_synthesis_prompt,
)


class TestInitSynthesizer:
    """Core synthesize() method tests."""

    @pytest.mark.asyncio
    async def test_synthesize_with_agent_uses_provider_directly(self):
        """When agent is provided, synthesize() calls agent.provider.chat() directly.

        Bypasses AgenticLoop — single LLM call: prompt → markdown.
        Uses the already-initialized provider (same credential path as victor chat).
        """
        mock_provider = AsyncMock()
        mock_provider.name = "test-provider"
        mock_provider.chat.return_value = MagicMock(content="# init.md\n\nProject overview.")

        mock_agent = MagicMock()
        mock_agent.provider = mock_provider
        mock_agent.model = "test-model"

        synthesizer = InitSynthesizer()
        result = await synthesizer.synthesize("raw data here", agent=mock_agent)

        mock_provider.chat.assert_called_once()
        call_kwargs = mock_provider.chat.call_args
        messages = call_kwargs[1].get("messages") or call_kwargs[0][0]
        assert any("raw data here" in str(m) for m in messages)
        assert result == "# init.md\n\nProject overview."

    @pytest.mark.asyncio
    async def test_synthesize_without_agent_creates_fresh(self):
        """When no agent, synthesize() uses direct provider call via _run_with_fresh_agent."""
        with patch.object(
            InitSynthesizer, "_run_with_fresh_agent", return_value="# Generated init.md"
        ) as mock_fresh:
            synthesizer = InitSynthesizer()
            result = await synthesizer.synthesize(
                "raw data", provider="ollama", model="qwen3-coder:30b"
            )

            mock_fresh.assert_called_once()
            assert result == "# Generated init.md"

    @pytest.mark.asyncio
    async def test_synthesize_cleans_code_fences(self):
        """Output wrapped in ``` gets cleaned."""
        mock_provider = AsyncMock()
        mock_provider.name = "test-provider"
        mock_provider.chat.return_value = MagicMock(
            content="```markdown\n# Project\n\nOverview.\n```"
        )
        mock_agent = MagicMock()
        mock_agent.provider = mock_provider
        mock_agent.model = None

        synthesizer = InitSynthesizer()
        result = await synthesizer.synthesize("raw data", agent=mock_agent)

        assert result == "# Project\n\nOverview."
        assert "```" not in result

    @pytest.mark.asyncio
    async def test_synthesize_prompt_contains_base_content(self):
        """The synthesis prompt includes the base_content."""
        mock_provider = AsyncMock()
        mock_provider.name = "test-provider"
        mock_provider.chat.return_value = MagicMock(content="# Result")
        mock_agent = MagicMock()
        mock_agent.provider = mock_provider
        mock_agent.model = None

        synthesizer = InitSynthesizer()
        await synthesizer.synthesize("UNIQUE_BASE_DATA_12345", agent=mock_agent)

        call_kwargs = mock_provider.chat.call_args
        messages = call_kwargs[1].get("messages") or call_kwargs[0][0]
        assert any("UNIQUE_BASE_DATA_12345" in str(m) for m in messages)

    @pytest.mark.asyncio
    async def test_synthesize_handles_failure_gracefully(self):
        """If provider.chat() raises, return empty string."""
        mock_provider = AsyncMock()
        mock_provider.name = "test-provider"
        mock_provider.chat.side_effect = RuntimeError("LLM failed")
        mock_agent = MagicMock()
        mock_agent.provider = mock_provider
        mock_agent.model = None

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
            result = await synthesizer.synthesize_with_tools(provider="ollama", model="qwen3:8b")

            # Should create with vertical="coding" for tool access
            create_kwargs = MockAgent.create.call_args[1]
            assert create_kwargs.get("vertical") == "coding"
            assert result == "# Tools init.md"


class TestInitSynthesizerEvolvableSection:
    """INIT_SYNTHESIS_RULES is registered as an evolvable section."""

    def test_init_synthesis_rules_in_evolvable_sections(self):
        """INIT_SYNTHESIS_RULES must be in PromptOptimizerLearner.EVOLVABLE_SECTIONS."""
        from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner

        assert "INIT_SYNTHESIS_RULES" in PromptOptimizerLearner.EVOLVABLE_SECTIONS

    def test_init_section_uses_gepa_only(self):
        """INIT_SYNTHESIS_RULES uses GEPA only — MIPROv2/CoT irrelevant for single-shot."""
        from victor.framework.rl.learners.prompt_optimizer import (
            PromptOptimizerLearner,
        )

        mock_db = MagicMock()
        mock_db.execute.return_value = MagicMock(fetchall=MagicMock(return_value=[]))
        mock_db.executescript = MagicMock()
        mock_db.commit = MagicMock()

        learner = PromptOptimizerLearner("test", mock_db)
        strategies = learner._extra_strategies.get("INIT_SYNTHESIS_RULES", [])

        # Single-shot: GEPA only (MIPROv2/CoT add tool patterns, not useful)
        assert len(strategies) == 1
        assert type(strategies[0]).__name__ == "GEPAStrategy"

    def test_build_synthesis_prompt_preserves_base_content(self):
        """Frame always keeps {base_content} safe regardless of rules content."""
        custom_rules = "RULES:\n- Custom rule 1\n- Custom rule 2"
        prompt = _build_synthesis_prompt("MY_RAW_DATA", rules=custom_rules)
        assert "MY_RAW_DATA" in prompt
        assert "Custom rule 1" in prompt
        assert "init.md" in prompt  # Frame intro preserved

    def test_synthesis_prompt_backward_compat(self):
        """SYNTHESIS_PROMPT constant still works with .format(base_content=...)."""
        result = SYNTHESIS_PROMPT.format(base_content="test data")
        assert "test data" in result
        assert "RULES:" in result


class TestInitSynthesizerGEPAWiring:
    """Tests for GEPA-evolved rules retrieval and usage."""

    @pytest.mark.asyncio
    async def test_uses_evolved_rules_when_available(self):
        """When GEPA has evolved rules, synthesize() injects them into frame."""
        evolved_rules = "RULES:\n- Improved rule 1\n- Improved rule 2"

        mock_provider = AsyncMock()
        mock_provider.name = "test-provider"
        mock_provider.chat.return_value = MagicMock(content="# Evolved result")
        mock_agent = MagicMock()
        mock_agent.provider = mock_provider
        mock_agent.model = None

        synthesizer = InitSynthesizer()
        with patch.object(InitSynthesizer, "_get_evolved_rules", return_value=evolved_rules):
            result = await synthesizer.synthesize("raw data", agent=mock_agent)

        call_kwargs = mock_provider.chat.call_args
        messages = call_kwargs[1].get("messages") or call_kwargs[0][0]
        prompt_sent = str(messages)
        assert "Improved rule 1" in prompt_sent
        assert "raw data" in prompt_sent  # base_content always present (frame)
        assert result == "# Evolved result"

    @pytest.mark.asyncio
    async def test_falls_back_to_static_when_no_evolution(self):
        """When GEPA has no evolved rules, uses static SYNTHESIS_RULES."""
        mock_provider = AsyncMock()
        mock_provider.name = "test-provider"
        mock_provider.chat.return_value = MagicMock(content="# Static result")
        mock_agent = MagicMock()
        mock_agent.provider = mock_provider
        mock_agent.model = None

        synthesizer = InitSynthesizer()
        with patch.object(InitSynthesizer, "_get_evolved_rules", return_value=None):
            result = await synthesizer.synthesize("raw data", agent=mock_agent)

        call_kwargs = mock_provider.chat.call_args
        messages = call_kwargs[1].get("messages") or call_kwargs[0][0]
        prompt_sent = str(messages)
        assert "RULES:" in prompt_sent
        assert "raw data" in prompt_sent
        assert result == "# Static result"

    @pytest.mark.asyncio
    async def test_evolved_rules_always_preserve_base_content(self):
        """Even with evolved rules, {base_content} is always in the prompt."""
        evolved_rules = "RULES:\n- Any content here"

        mock_provider = AsyncMock()
        mock_provider.name = "test-provider"
        mock_provider.chat.return_value = MagicMock(content="# Result")
        mock_agent = MagicMock()
        mock_agent.provider = mock_provider
        mock_agent.model = None

        synthesizer = InitSynthesizer()
        with patch.object(InitSynthesizer, "_get_evolved_rules", return_value=evolved_rules):
            await synthesizer.synthesize("UNIQUE_DATA_XYZ", agent=mock_agent)

        call_kwargs = mock_provider.chat.call_args
        messages = call_kwargs[1].get("messages") or call_kwargs[0][0]
        prompt_sent = str(messages)
        assert "UNIQUE_DATA_XYZ" in prompt_sent  # Frame always includes data

    def test_get_evolved_rules_gates_on_settings(self):
        """_get_evolved_rules returns None when optimization disabled."""
        mock_settings = MagicMock()
        mock_settings.prompt_optimization.enabled = False

        with patch("victor.config.settings.get_settings", return_value=mock_settings):
            result = InitSynthesizer._get_evolved_rules("ollama")

        assert result is None

    def test_get_evolved_rules_gates_on_section_strategies(self):
        """_get_evolved_rules returns None when section strategies are empty."""
        mock_po = MagicMock()
        mock_po.enabled = True
        mock_po.get_strategies_for_section.return_value = []

        mock_settings = MagicMock()
        mock_settings.prompt_optimization = mock_po

        with patch("victor.config.settings.get_settings", return_value=mock_settings):
            result = InitSynthesizer._get_evolved_rules("ollama")

        assert result is None


class TestInitQualityScoring:
    """Tests for init output quality signal logging."""

    def test_quality_score_perfect_output(self):
        """Perfect output with all sections scores high."""
        mock_usage = MagicMock()
        result = (
            "# Project Overview\nA project.\n"
            "# System Flow\nUser → Core → Output.\n"
            "# Package Layout\n| Path | Description |\n"
            "# Key Entry Points\n| Component | Path |\n"
            "# Architecture Patterns\n- Pattern A.\n"
            "# Development Commands\n```bash\nmake test\n```\n"
            "# Dependencies\nCore: pydantic.\n"
            "# Configuration\nSettings via .env.\n"
            "# Codebase Scale\n100K lines.\n"
        )
        # Pad to ~80 lines
        result += "\n" * 70

        InitSynthesizer._log_init_quality(mock_usage, result)

        call_data = mock_usage.log_event.call_args[0][1]
        assert call_data["sections_found"] == 9
        assert call_data["sections_total"] == 9
        assert call_data["section_score"] == 1.0
        assert 60 <= call_data["line_count"] <= 100
        assert call_data["length_score"] == 1.0
        assert call_data["quality_score"] == pytest.approx(1.0, abs=0.01)

    def test_quality_score_missing_sections(self):
        """Output missing sections scores lower."""
        mock_usage = MagicMock()
        # Only 2 of 9 sections
        result = "# Project Overview\nA project.\n# Dependencies\nCore: pydantic.\n"

        InitSynthesizer._log_init_quality(mock_usage, result)

        call_data = mock_usage.log_event.call_args[0][1]
        assert call_data["sections_found"] == 2
        assert call_data["section_score"] == pytest.approx(2 / 9, abs=0.01)
        assert call_data["quality_score"] < 0.5

    def test_quality_score_empty_output(self):
        """Empty output gets minimal quality score."""
        mock_usage = MagicMock()
        InitSynthesizer._log_init_quality(mock_usage, "")

        call_data = mock_usage.log_event.call_args[0][1]
        assert call_data["sections_found"] == 0
        assert call_data["quality_score"] < 0.3
