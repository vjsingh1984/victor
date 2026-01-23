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

"""Centralized singleton reset utilities for test isolation.

This module provides comprehensive singleton reset functionality to ensure
test isolation. All singletons that maintain state between tests should be
registered here.

Design Principles:
- Fail silently on import errors (modules may not be installed)
- Reset both before AND after each test for bidirectional isolation
- Group resets by category for maintainability
- Use lazy imports to avoid import-time side effects
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class SingletonResetRegistry:
    """Registry for singleton reset functions.

    This class collects all singleton reset functions and provides
    a single entry point to reset all registered singletons.

    Usage:
        registry = SingletonResetRegistry()
        registry.register(lambda: MyClass.reset_instance())
        registry.reset_all()
    """

    def __init__(self) -> None:
        self._reset_functions: List[Callable[[], None]] = []
        self._initialized = False

    def register(self, reset_fn: Callable[[], None]) -> None:
        """Register a reset function."""
        self._reset_functions.append(reset_fn)

    def reset_all(self) -> None:
        """Execute all registered reset functions."""
        for reset_fn in self._reset_functions:
            try:
                reset_fn()
            except Exception as e:
                # Log but don't fail - some resets may fail if module state is unexpected
                logger.debug(f"Singleton reset warning: {e}")

    def initialize(self) -> None:
        """Initialize the registry with all known singletons."""
        if self._initialized:
            return

        self._register_embedding_singletons()
        self._register_agent_singletons()
        self._register_framework_singletons()
        self._register_tool_singletons()
        self._register_workflow_singletons()
        self._register_observability_singletons()
        self._register_storage_singletons()
        self._register_processing_singletons()
        self._register_classification_singletons()
        self._register_rl_hooks_singletons()
        self._register_core_singletons()

        self._initialized = True

    def _safe_reset(self, module_path: str, class_name: str, method: str = "reset_instance") -> None:
        """Safely attempt to reset a singleton.

        Args:
            module_path: Full module path (e.g., "victor.core.registry")
            class_name: Name of the class with singleton
            method: Reset method name (default: "reset_instance")
        """
        def _do_reset():
            try:
                import importlib
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name, None)
                if cls is not None:
                    reset_method = getattr(cls, method, None)
                    if reset_method is not None:
                        reset_method()
            except ImportError:
                pass  # Module not available
            except Exception as e:
                logger.debug(f"Reset {module_path}.{class_name} failed: {e}")

        self.register(_do_reset)

    def _safe_reset_module_var(self, module_path: str, var_name: str) -> None:
        """Safely reset a module-level singleton variable to None.

        Args:
            module_path: Full module path
            var_name: Name of the module-level variable
        """
        def _do_reset():
            try:
                import importlib
                module = importlib.import_module(module_path)
                if hasattr(module, var_name):
                    setattr(module, var_name, None)
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"Reset {module_path}.{var_name} failed: {e}")

        self.register(_do_reset)

    # ==========================================================================
    # Registration Methods by Category
    # ==========================================================================

    def _register_embedding_singletons(self) -> None:
        """Reset embedding-related singletons."""
        self._safe_reset(
            "victor.storage.embeddings.task_classifier",
            "TaskTypeClassifier"
        )
        self._safe_reset(
            "victor.storage.embeddings.service",
            "EmbeddingService"
        )
        self._safe_reset(
            "victor.storage.embeddings.intent_classifier",
            "IntentClassifier"
        )
        self._safe_reset(
            "victor.storage.cache.embedding_cache_manager",
            "EmbeddingCacheManager"
        )

    def _register_agent_singletons(self) -> None:
        """Reset agent-related singletons."""
        self._safe_reset(
            "victor.agent.shared_tool_registry",
            "SharedToolRegistry"
        )
        self._safe_reset(
            "victor.agent.usage_analytics",
            "UsageAnalytics"
        )
        # Model capability loaders
        self._safe_reset(
            "victor.agent.tool_calling.capabilities",
            "ModelCapabilityLoader"
        )

    def _register_framework_singletons(self) -> None:
        """Reset framework-related singletons."""
        self._safe_reset(
            "victor.framework.chain_registry",
            "ChainRegistry"
        )
        self._safe_reset(
            "victor.framework.escape_hatch_registry",
            "EscapeHatchRegistry"
        )
        self._safe_reset(
            "victor.framework.event_registry",
            "EventRegistry"
        )
        self._safe_reset(
            "victor.framework.module_loader",
            "ModuleLoader"
        )
        self._safe_reset(
            "victor.framework.handler_registry",
            "HandlerRegistry"
        )
        self._safe_reset(
            "victor.framework.persona_registry",
            "PersonaRegistry"
        )
        self._safe_reset(
            "victor.framework.tools",
            "ToolRegistry"
        )
        self._safe_reset(
            "victor.framework.task_types",
            "TaskTypeRegistry"
        )
        self._safe_reset(
            "victor.framework.multi_agent.persona_provider",
            "PersonaProvider"
        )

    def _register_tool_singletons(self) -> None:
        """Reset tool-related singletons."""
        self._safe_reset(
            "victor.tools.progressive_registry",
            "ProgressiveToolsRegistry"
        )
        self._safe_reset(
            "victor.tools.tool_graph",
            "ToolGraphRegistry"
        )
        self._safe_reset(
            "victor.tools.selection.registry",
            "ToolSelectionStrategyRegistry"
        )
        self._safe_reset(
            "victor.tools.metadata",
            "ToolMetadataRegistry"
        )
        self._safe_reset(
            "victor.tools.alias_resolver",
            "ToolAliasResolver"
        )

    def _register_workflow_singletons(self) -> None:
        """Reset workflow-related singletons."""
        self._safe_reset(
            "victor.workflows.isolation",
            "SandboxProviderRegistry"
        )
        # Module-level singletons with dedicated reset functions
        def _reset_trigger_registry():
            try:
                from victor.workflows.trigger_registry import reset_trigger_registry
                reset_trigger_registry()
            except ImportError:
                pass

        def _reset_scheduler():
            try:
                from victor.workflows.scheduler import reset_scheduler
                reset_scheduler()
            except ImportError:
                pass

        def _reset_version_registry():
            try:
                from victor.workflows.versioning import reset_version_registry
                reset_version_registry()
            except ImportError:
                pass

        self.register(_reset_trigger_registry)
        self.register(_reset_scheduler)
        self.register(_reset_version_registry)

    def _register_observability_singletons(self) -> None:
        """Reset observability-related singletons."""
        self._safe_reset(
            "victor.observability.event_bus",
            "EventBus"
        )
        self._safe_reset(
            "victor.observability.bridge",
            "ObservabilityBridge"
        )
        self._safe_reset(
            "victor.observability.event_registry",
            "EventTypeRegistry"
        )
        self._safe_reset(
            "victor.native.observability",
            "NativeMetrics"
        )

    def _register_storage_singletons(self) -> None:
        """Reset storage-related singletons."""
        self._safe_reset(
            "victor.core.database",
            "DatabaseManager"
        )

    def _register_processing_singletons(self) -> None:
        """Reset processing-related singletons."""
        self._safe_reset(
            "victor.processing.file_types.detector",
            "FileTypeRegistry"
        )
        self._safe_reset(
            "victor.evaluation.correction.registry",
            "CodeValidatorRegistry",
            "reset_singleton"
        )
        self._safe_reset(
            "victor.coding.codebase.indexer",
            "BackgroundIndexerService"
        )
        # Native processing module-level instances
        for var in [
            "_symbol_extractor_instance",
            "_argument_normalizer_instance",
            "_similarity_computer_instance",
            "_text_chunker_instance",
            "_ast_indexer_instance",
        ]:
            self._safe_reset_module_var("victor.processing.native", var)

    def _register_classification_singletons(self) -> None:
        """Reset classification-related singletons."""
        # Use the module's own reset function if available
        def _reset_classification():
            try:
                from victor.classification.nudge_engine import reset_singletons
                reset_singletons()
            except ImportError:
                pass
            except Exception:
                pass

        self.register(_reset_classification)

    def _register_rl_hooks_singletons(self) -> None:
        """Reset RL hooks module-level singletons."""
        self._safe_reset_module_var(
            "victor.dataanalysis.rl",
            "_hooks_instance"
        )
        self._safe_reset_module_var(
            "victor.coding.rl.hooks",
            "_hooks_instance"
        )
        self._safe_reset_module_var(
            "victor.devops.rl",
            "_hooks_instance"
        )

    def _register_core_singletons(self) -> None:
        """Reset core module singletons."""
        self._safe_reset(
            "victor.core.mode_config",
            "ModeConfigRegistry"
        )
        self._safe_reset(
            "victor.core.registry_base",
            "RegistryBase"
        )
        self._safe_reset(
            "victor.core.tool_tier_registry",
            "ToolTierRegistry"
        )
        # Integration singletons
        self._safe_reset(
            "victor.integrations.api.event_bridge",
            "EventBroadcaster"
        )


# Global registry instance
_global_registry: Optional[SingletonResetRegistry] = None


def get_singleton_reset_registry() -> SingletonResetRegistry:
    """Get or create the global singleton reset registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = SingletonResetRegistry()
        _global_registry.initialize()
    return _global_registry


def reset_all_singletons() -> None:
    """Reset all registered singletons.

    This is the main entry point for test fixtures.
    """
    registry = get_singleton_reset_registry()
    registry.reset_all()


def reset_event_loop_singletons() -> None:
    """Reset singletons that may hold async resources.

    Call this for async test cleanup to prevent event loop warnings.
    """
    try:
        from victor.observability.event_bus import EventBus
        EventBus.reset_instance()
    except ImportError:
        pass

    try:
        from victor.integrations.api.event_bridge import EventBroadcaster
        if hasattr(EventBroadcaster, '_instance') and EventBroadcaster._instance is not None:
            EventBroadcaster._instance = None
    except ImportError:
        pass
