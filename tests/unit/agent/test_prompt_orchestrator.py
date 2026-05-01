"""Tests for PromptOrchestrator."""

from unittest.mock import MagicMock

import pytest

from victor.agent.prompt_orchestrator import (
    PromptOrchestrator,
    OrchestratorConfig,
    get_prompt_orchestrator,
)


class TestOrchestratorConfig:
    """Test suite for OrchestratorConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OrchestratorConfig()

        assert config.use_evolved_content is True
        assert config.enable_constraint_activation is True
        assert config.fallback_to_static is True
        assert config.cache_evolved_content is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = OrchestratorConfig(
            use_evolved_content=False,
            enable_constraint_activation=False,
            fallback_to_static=False,
            cache_evolved_content=False,
        )

        assert config.use_evolved_content is False
        assert config.enable_constraint_activation is False
        assert config.fallback_to_static is False
        assert config.cache_evolved_content is False


class TestPromptOrchestrator:
    """Test suite for PromptOrchestrator."""

    def test_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = PromptOrchestrator()

        assert orchestrator._config is not None
        assert orchestrator._injector is None
        assert orchestrator._registry is None
        assert orchestrator._resolver is None
        assert orchestrator._constraint_activator is None

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = OrchestratorConfig(use_evolved_content=False)
        orchestrator = PromptOrchestrator(config=config)

        assert orchestrator._config is config
        assert orchestrator._config.use_evolved_content is False

    def test_detect_builder_type_legacy(self):
        """Test auto-detection of legacy builder."""
        orchestrator = PromptOrchestrator()

        builder_type = orchestrator._detect_builder_type(prompt_contributors=[])

        assert builder_type == "legacy"

    def test_detect_builder_type_framework(self):
        """Test auto-detection of framework builder."""
        orchestrator = PromptOrchestrator()

        builder_type = orchestrator._detect_builder_type(base_prompt="You are an assistant.")

        assert builder_type == "framework"

    def test_detect_builder_type_default(self):
        """Test default builder type when no hints provided."""
        orchestrator = PromptOrchestrator()

        builder_type = orchestrator._detect_builder_type()

        assert builder_type == "legacy"  # Default for backward compatibility

    def test_detect_builder_type_existing_builder(self):
        """Test auto-detection of legacy builder from existing builder instance."""
        orchestrator = PromptOrchestrator()

        builder_type = orchestrator._detect_builder_type(builder=object())

        assert builder_type == "legacy"

    def test_build_system_prompt_with_auto_detection(self):
        """Test build_system_prompt with auto builder type detection."""
        orchestrator = PromptOrchestrator()

        # Should detect legacy builder from prompt_contributors
        prompt = orchestrator.build_system_prompt(builder_type="auto", prompt_contributors=[])

        # Should return a string (even if minimal)
        assert isinstance(prompt, str)

    def test_build_system_prompt_with_existing_legacy_builder(self):
        """Test legacy facade path can reuse an existing builder instance."""
        orchestrator = PromptOrchestrator()
        builder = MagicMock()
        builder.build.return_value = "Base prompt"
        hook = MagicMock()

        prompt = orchestrator.build_system_prompt(
            builder_type="legacy",
            builder=builder,
            get_context_window=lambda: 65536,
            on_prompt_built=hook,
        )

        assert "Base prompt" in prompt
        assert "PARALLEL READ BUDGET" in prompt
        builder.build.assert_called_once_with()
        hook.assert_called_once_with(prompt)

    def test_activate_constraints(self):
        """Test constraint activation through orchestrator."""
        from victor.workflows.definition import FullAccessConstraints

        orchestrator = PromptOrchestrator(
            config=OrchestratorConfig(enable_constraint_activation=True)
        )

        constraints = FullAccessConstraints()
        success = orchestrator.activate_constraints(constraints, "coding")

        assert success is True

    def test_activate_constraints_disabled(self):
        """Test constraint activation when disabled."""
        from victor.workflows.definition import FullAccessConstraints

        orchestrator = PromptOrchestrator(
            config=OrchestratorConfig(enable_constraint_activation=False)
        )

        constraints = FullAccessConstraints()
        success = orchestrator.activate_constraints(constraints, "coding")

        assert success is True  # Disabled means always return True

    def test_deactivate_constraints(self):
        """Test constraint deactivation."""
        orchestrator = PromptOrchestrator()

        # Should not raise any errors
        orchestrator.deactivate_constraints()

    def test_get_resolver_creates_resolver(self):
        """Test that _get_resolver() creates resolver."""
        orchestrator = PromptOrchestrator()

        resolver = orchestrator._get_resolver()

        assert resolver is not None
        assert hasattr(resolver, "resolve_section")

    def test_get_constraint_activator_creates_service(self):
        """Test that _get_constraint_activator() creates service."""
        orchestrator = PromptOrchestrator()

        activator = orchestrator._get_constraint_activator()

        assert activator is not None
        assert hasattr(activator, "activate_constraints")

    def test_resolver_is_cached(self):
        """Test that resolver is cached across calls."""
        orchestrator = PromptOrchestrator()

        resolver1 = orchestrator._get_resolver()
        resolver2 = orchestrator._get_resolver()

        assert resolver1 is resolver2

    def test_constraint_activator_is_cached(self):
        """Test that constraint activator is cached across calls."""
        orchestrator = PromptOrchestrator()

        activator1 = orchestrator._get_constraint_activator()
        activator2 = orchestrator._get_constraint_activator()

        assert activator1 is activator2


class TestGetPromptOrchestrator:
    """Test suite for get_prompt_orchestrator() singleton."""

    def test_returns_singleton(self):
        """Test that get_prompt_orchestrator() returns singleton."""
        orchestrator1 = get_prompt_orchestrator()
        orchestrator2 = get_prompt_orchestrator()

        assert orchestrator1 is orchestrator2

    def test_direct_initialization_with_config(self):
        """Test that config is used when creating orchestrator directly."""
        config = OrchestratorConfig(use_evolved_content=False)
        orchestrator = PromptOrchestrator(config=config)

        assert orchestrator._config.use_evolved_content is False

    def test_singleton_persists(self):
        """Test that singleton persists across calls."""
        orchestrator1 = get_prompt_orchestrator()

        # Get again without config
        orchestrator2 = get_prompt_orchestrator()

        assert orchestrator1 is orchestrator2
