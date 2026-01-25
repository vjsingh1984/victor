# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Dependency Inversion Principle compliance (Phase 12.1).

Tests that step handlers properly use dependency injection instead of
hard-coded fallback imports.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from unittest.mock import MagicMock, patch

import pytest


class TestRegistryProtocols:
    """Tests for registry service protocols."""

    def test_workflow_registry_protocol_is_defined(self):
        """Test that workflow registry protocol is defined."""
        from victor.framework.service_provider import WorkflowRegistryService

        assert WorkflowRegistryService is not None

    def test_team_registry_protocol_is_defined(self):
        """Test that team registry protocol is defined."""
        from victor.framework.service_provider import TeamRegistryService

        assert TeamRegistryService is not None

    def test_chain_registry_protocol_is_defined(self):
        """Test that chain registry protocol is defined."""
        from victor.framework.service_provider import ChainRegistryService

        assert ChainRegistryService is not None

    def test_persona_registry_protocol_is_defined(self):
        """Test that persona registry protocol is defined."""
        from victor.framework.service_provider import PersonaRegistryService

        assert PersonaRegistryService is not None

    def test_handler_registry_protocol_is_defined(self):
        """Test that handler registry protocol is defined."""
        from victor.framework.service_provider import HandlerRegistryService

        assert HandlerRegistryService is not None


class TestServiceProviderRegistries:
    """Tests for service provider registry registrations."""

    def test_framework_service_provider_includes_registries(self):
        """Test that FrameworkServiceProvider includes registry options."""
        from victor.framework.service_provider import FrameworkServiceProvider

        provider = FrameworkServiceProvider(
            include_workflow_registry=True,
            include_team_registry=True,
            include_chain_registry=True,
            include_persona_registry=True,
            include_handler_registry=True,
        )

        registrations = provider.get_registrations()
        registration_types = [reg.service_type.__name__ for reg in registrations]

        assert "WorkflowRegistryService" in registration_types
        assert "TeamRegistryService" in registration_types
        assert "ChainRegistryService" in registration_types
        assert "PersonaRegistryService" in registration_types
        assert "HandlerRegistryService" in registration_types

    def test_container_provides_all_registries(self):
        """Test that container can provide all registries after configuration."""
        from victor.core.container import ServiceContainer
        from victor.framework.service_provider import (
            configure_framework_services,
            WorkflowRegistryService,
            TeamRegistryService,
            ChainRegistryService,
            PersonaRegistryService,
            HandlerRegistryService,
        )

        container = ServiceContainer()
        configure_framework_services(container, include_registries=True)

        # Each registry should be resolvable
        workflow_reg = container.get(WorkflowRegistryService)
        assert workflow_reg is not None

        team_reg = container.get(TeamRegistryService)
        assert team_reg is not None

        chain_reg = container.get(ChainRegistryService)
        assert chain_reg is not None

        persona_reg = container.get(PersonaRegistryService)
        assert persona_reg is not None

        handler_reg = container.get(HandlerRegistryService)
        assert handler_reg is not None


class TestDecomposedHandlersDI:
    """Tests for decomposed handlers using DI."""

    def test_workflow_handler_uses_injected_registry(self):
        """Test that WorkflowStepHandler uses injected registry."""
        from victor.framework.decomposed_handlers import WorkflowStepHandler

        mock_registry = MagicMock()
        handler = WorkflowStepHandler(workflow_registry=mock_registry)

        # The handler should have the injected registry
        assert handler._workflow_registry is mock_registry

    def test_team_spec_handler_uses_injected_registry(self):
        """Test that TeamSpecStepHandler uses injected registry."""
        from victor.framework.decomposed_handlers import TeamSpecStepHandler

        mock_registry = MagicMock()
        handler = TeamSpecStepHandler(team_registry=mock_registry)

        assert handler._team_registry is mock_registry

    def test_chain_handler_uses_injected_registry(self):
        """Test that ChainStepHandler uses injected registry."""
        from victor.framework.decomposed_handlers import ChainStepHandler

        mock_registry = MagicMock()
        handler = ChainStepHandler(chain_registry=mock_registry)

        assert handler._chain_registry is mock_registry

    def test_persona_handler_uses_injected_registry(self):
        """Test that PersonaStepHandler uses injected registry."""
        from victor.framework.decomposed_handlers import PersonaStepHandler

        mock_registry = MagicMock()
        handler = PersonaStepHandler(persona_registry=mock_registry)

        assert handler._persona_registry is mock_registry

    def test_handler_registration_uses_injected_registry(self):
        """Test that HandlerRegistrationStepHandler uses injected registry."""
        from victor.framework.decomposed_handlers import HandlerRegistrationStepHandler

        mock_registry = MagicMock()
        handler = HandlerRegistrationStepHandler(handler_registry=mock_registry)

        assert handler._handler_registry is mock_registry


class TestCreateDecomposedHandlersWithDI:
    """Tests for creating decomposed handlers with full DI."""

    def test_create_decomposed_handlers_with_all_registries(self):
        """Test creating handlers with all registries injected."""
        from victor.framework.decomposed_handlers import create_decomposed_handlers

        mock_workflow = MagicMock()
        mock_trigger = MagicMock()
        mock_team = MagicMock()
        mock_chain = MagicMock()
        mock_persona = MagicMock()
        mock_handler = MagicMock()

        handlers = create_decomposed_handlers(
            workflow_registry=mock_workflow,
            trigger_registry=mock_trigger,
            team_registry=mock_team,
            chain_registry=mock_chain,
            persona_registry=mock_persona,
            handler_registry=mock_handler,
        )

        # Find specific handlers and verify injection
        workflow_handler = next(h for h in handlers if h.name == "workflow")
        assert workflow_handler._workflow_registry is mock_workflow
        assert workflow_handler._trigger_registry is mock_trigger

        team_handler = next(h for h in handlers if h.name == "team_spec")
        assert team_handler._team_registry is mock_team

        chain_handler = next(h for h in handlers if h.name == "chain")
        assert chain_handler._chain_registry is mock_chain

        persona_handler = next(h for h in handlers if h.name == "persona")
        assert persona_handler._persona_registry is mock_persona

        handler_reg_handler = next(h for h in handlers if h.name == "handler_registration")
        assert handler_reg_handler._handler_registry is mock_handler


class TestDIPComplianceDocumentation:
    """Tests documenting DIP compliance patterns."""

    def test_no_hard_coded_imports_in_handlers(self):
        """Document expected pattern: no hard-coded imports, use DI."""
        # This test documents the expected usage pattern
        # Old pattern (bad):
        #   if self._registry is None:
        #       from victor.workflows.registry import get_global_registry
        #       self._registry = get_global_registry()
        #
        # New pattern (good):
        #   registry = self._registry  # Injected via constructor
        #   if registry is None:
        #       raise ValueError("registry is required")

        from victor.framework.decomposed_handlers import WorkflowStepHandler

        # When created without injection, handler still works
        # but prefers injection
        handler = WorkflowStepHandler()
        assert handler._workflow_registry is None  # Not injected

        # When created with injection, uses the injected registry
        mock_registry = MagicMock()
        handler_with_di = WorkflowStepHandler(workflow_registry=mock_registry)
        assert handler_with_di._workflow_registry is mock_registry  # Injected

    def test_di_pattern_for_step_handlers(self):
        """Document the DI pattern for step handlers."""
        # Expected pattern for step handlers:
        # 1. Constructor accepts optional registry parameters
        # 2. When injected, use the injected registry
        # 3. When not injected, fall back to lazy import (deprecated)
        # 4. Future: require injection, remove fallback

        from victor.framework.decomposed_handlers import (
            WorkflowStepHandler,
            TeamSpecStepHandler,
            ChainStepHandler,
            PersonaStepHandler,
            HandlerRegistrationStepHandler,
        )

        # All handlers support DI
        mock = MagicMock()

        assert WorkflowStepHandler(workflow_registry=mock)._workflow_registry is mock
        assert TeamSpecStepHandler(team_registry=mock)._team_registry is mock
        assert ChainStepHandler(chain_registry=mock)._chain_registry is mock
        assert PersonaStepHandler(persona_registry=mock)._persona_registry is mock
        assert HandlerRegistrationStepHandler(handler_registry=mock)._handler_registry is mock
