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

"""Base YAML workflow provider for vertical-specific workflow implementations.

This module provides a Template Method + Strategy pattern base class that eliminates
workflow provider duplication across verticals. Subclasses only need to specify
the escape hatches module path and optionally the workflows directory.

Design Patterns:
    - Template Method: Common algorithm in base class, subclass customization via hooks
    - Strategy: Escape hatches module loaded dynamically based on subclass specification
    - Lazy Loading: Workflows loaded on first access for performance

Example:
    class ResearchWorkflowProvider(BaseYAMLWorkflowProvider):
        '''Provides research-specific workflows.'''

        def _get_escape_hatches_module(self) -> str:
            return "victor.research.escape_hatches"

    # Usage
    provider = ResearchWorkflowProvider()
    async for chunk in provider.astream("deep_research", orchestrator, {}):
        print(f"[{chunk.progress:.0f}%] {chunk.event_type.value}")
"""

from __future__ import annotations

import importlib
import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from victor.core.verticals.protocols import WorkflowProviderProtocol
from victor.workflows.definition import WorkflowDefinition

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.core.protocols import OrchestratorProtocol as AgentOrchestrator
    from victor.framework.graph import ExecutionResult
    from victor.workflows.executor import WorkflowExecutor, WorkflowResult
    from victor.workflows.streaming import WorkflowStreamChunk
    from victor.workflows.streaming_executor import StreamingWorkflowExecutor
    from victor.workflows.unified_compiler import CachedCompiledGraph, UnifiedWorkflowCompiler
    from victor.workflows.yaml_loader import YAMLWorkflowConfig


class _MinimalOrchestrator:
    """Minimal orchestrator mock for compute-only workflow execution.

    This class provides the minimal interface required by WorkflowExecutor
    for workflows that only contain compute nodes (no agent nodes).
    Agent nodes require a full orchestrator with LLM capabilities.
    """

    pass


class BaseYAMLWorkflowProvider(WorkflowProviderProtocol, ABC):
    """Base class for YAML-based workflow providers.

    This abstract base class implements the common workflow provider pattern
    used across verticals (Research, DevOps, DataAnalysis, etc.), eliminating
    ~200 lines of duplicated boilerplate per vertical.

    Subclasses only need to implement:
        - _get_escape_hatches_module(): Return module path for CONDITIONS/TRANSFORMS

    Optionally override:
        - _get_workflows_directory(): Return Path to YAML workflow files
        - get_auto_workflows(): Return automatic workflow triggers
        - get_workflow_for_task_type(): Map task types to workflow names

    Features:
        - Lazy loading of YAML workflows with caching
        - Automatic escape hatches registration from vertical-specific modules
        - Standard and streaming workflow execution
        - Error handling with graceful degradation

    Example:
        class DevOpsWorkflowProvider(BaseYAMLWorkflowProvider):
            '''Provides DevOps-specific workflows.'''

            def _get_escape_hatches_module(self) -> str:
                return "victor.devops.escape_hatches"

            def get_auto_workflows(self) -> List[Tuple[str, str]]:
                return [
                    (r"deploy\\s+infrastructure", "deploy_infrastructure"),
                    (r"container(ize)?", "container_setup"),
                ]

        provider = DevOpsWorkflowProvider()
        workflows = provider.get_workflows()  # Lazy-loads YAML files
    """

    def __init__(self) -> None:
        """Initialize the workflow provider with lazy loading support."""
        self._workflows: Optional[Dict[str, WorkflowDefinition]] = None
        self._config: Optional["YAMLWorkflowConfig"] = None
        self._compiler: Optional["UnifiedWorkflowCompiler"] = None

    @abstractmethod
    def _get_escape_hatches_module(self) -> str:
        """Return the fully qualified module path for escape hatches.

        This method must be implemented by subclasses to specify where
        the CONDITIONS and TRANSFORMS dictionaries are defined.

        Returns:
            Fully qualified module path string, e.g., "victor.research.escape_hatches"

        Example:
            def _get_escape_hatches_module(self) -> str:
                return "victor.research.escape_hatches"
        """
        ...

    def _get_capability_provider_module(self) -> Optional[str]:
        """Return the module path for the capability provider.

        This optional method allows subclasses to specify their capability
        provider module for automatic registration. This enables workflows
        to access vertical-specific capabilities without manual configuration.

        Returns:
            Fully qualified module path string or None if not applicable

        Example:
            def _get_capability_provider_module(self) -> Optional[str]:
                return "victor.research.capabilities"

        Note:
            Returning None indicates this vertical doesn't have a capability
            provider or doesn't want automatic registration.
        """
        return None

    def get_capability_provider(self) -> Optional[Any]:
        """Get the capability provider for this vertical.

        This convenience method loads and returns the capability provider
        if the vertical has defined one via _get_capability_provider_module().

        Returns:
            Capability provider instance or None if not configured

        Raises:
            ImportError: If the capability provider module cannot be imported
            AttributeError: If the module doesn't have the expected provider

        Example:
            provider = ResearchWorkflowProvider()
            capabilities = provider.get_capability_provider()
            if capabilities:
                print(f"Capabilities: {list(capabilities.get_capabilities().keys())}")
        """
        module_path = self._get_capability_provider_module()
        if not module_path:
            return None

        try:
            module = importlib.import_module(module_path)
            # Look for concrete capability provider classes (not abstract base)
            for attr_name in dir(module):
                # Skip private attributes and the abstract base
                if attr_name.startswith("_"):
                    continue
                if attr_name == "BaseCapabilityProvider":
                    continue

                attr = getattr(module, attr_name)
                # Check if it's a concrete class (not abstract) with "CapabilityProvider" in the name
                if (
                    isinstance(attr, type)
                    and "CapabilityProvider" in attr_name
                    and not attr_name.startswith("Base")
                    and hasattr(attr, "get_capabilities")
                    and hasattr(attr, "get_capability_metadata")
                ):
                    try:
                        # Try to instantiate - this will fail for abstract classes
                        instance = attr()
                        # Verify it has the required methods
                        if hasattr(instance, "get_capabilities") and hasattr(instance, "get_capability_metadata"):
                            return instance
                    except TypeError:
                        # Abstract class or can't be instantiated, skip
                        continue

            logger.warning(
                f"No concrete capability provider class found in {module_path}"
            )
            return None
        except ImportError as e:
            logger.warning(f"Failed to import capability provider from {module_path}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to instantiate capability provider from {module_path}: {e}")
            return None

    def _get_workflows_directory(self) -> Path:
        """Return the directory containing YAML workflow files.

        Override this method if workflows are in a non-standard location.
        By default, returns the parent directory of the escape hatches module,
        with "workflows" appended (e.g., victor/research/workflows/).

        Returns:
            Path to the directory containing *.yaml workflow files

        Example:
            def _get_workflows_directory(self) -> Path:
                return Path("/custom/path/to/workflows")
        """
        # Default: derive from escape hatches module path
        # e.g., "victor.research.escape_hatches" -> victor/research/workflows/
        module_path = self._get_escape_hatches_module()
        # Remove ".escape_hatches" suffix and convert to path
        base_module = module_path.rsplit(".", 1)[0]  # "victor.research"
        module_parts = base_module.split(".")

        # Get the directory of the first module part (victor)
        try:
            base_module_obj = importlib.import_module(module_parts[0])
            if hasattr(base_module_obj, "__path__"):
                base_path = Path(base_module_obj.__path__[0])
            elif hasattr(base_module_obj, "__file__") and base_module_obj.__file__:
                base_path = Path(base_module_obj.__file__).parent
            else:
                raise ImportError(f"Cannot determine path for module {module_parts[0]}")

            # Navigate to the submodule directory
            for part in module_parts[1:]:
                base_path = base_path / part

            return base_path / "workflows"
        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to determine workflows directory: {e}")
            # Fallback: use current file's parent as base
            return Path(__file__).parent

    def _load_escape_hatches(self) -> Tuple[
        Dict[str, Callable[[Dict[str, Any]], str]],
        Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
    ]:
        """Dynamically load escape hatches from the vertical-specific module.

        Returns:
            Tuple of (CONDITIONS dict, TRANSFORMS dict) from the escape hatches module

        Raises:
            ImportError: If the escape hatches module cannot be imported
            AttributeError: If CONDITIONS or TRANSFORMS are not defined
        """
        module_path = self._get_escape_hatches_module()
        try:
            module = importlib.import_module(module_path)
            conditions = getattr(module, "CONDITIONS", {})
            transforms = getattr(module, "TRANSFORMS", {})
            logger.debug(
                f"Loaded escape hatches from {module_path}: "
                f"{len(conditions)} conditions, {len(transforms)} transforms"
            )
            return conditions, transforms
        except ImportError as e:
            logger.warning(f"Failed to import escape hatches from {module_path}: {e}")
            return {}, {}
        except AttributeError as e:
            logger.warning(f"Missing CONDITIONS/TRANSFORMS in {module_path}: {e}")
            return {}, {}

    def _get_config(self) -> "YAMLWorkflowConfig":
        """Get YAML workflow config with escape hatches registered.

        Creates a YAMLWorkflowConfig instance with the vertical-specific
        conditions and transforms loaded from the escape hatches module.

        Returns:
            YAMLWorkflowConfig instance configured for this vertical
        """
        if self._config is None:
            from victor.workflows.yaml_loader import YAMLWorkflowConfig

            conditions, transforms = self._load_escape_hatches()
            workflows_dir = self._get_workflows_directory()

            self._config = YAMLWorkflowConfig(
                base_dir=workflows_dir,
                condition_registry=conditions,
                transform_registry=transforms,
            )
            logger.debug(f"Created YAML config with base_dir={workflows_dir}")

        return self._config

    def _load_workflows(self) -> Dict[str, WorkflowDefinition]:
        """Lazy load all YAML workflows from the workflows directory.

        Uses escape hatches for complex conditions that can't be expressed in YAML.
        Caches loaded workflows for subsequent access.

        Returns:
            Dict mapping workflow names to WorkflowDefinition instances
        """
        if self._workflows is None:
            try:
                from victor.workflows.yaml_loader import load_workflows_from_directory

                workflows_dir = self._get_workflows_directory()
                config = self._get_config()

                self._workflows = load_workflows_from_directory(
                    workflows_dir,
                    pattern="*.yaml",
                    config=config,
                )
                logger.debug(f"Loaded {len(self._workflows)} YAML workflows from {workflows_dir}")
            except Exception as e:
                logger.warning(f"Failed to load YAML workflows: {e}")
                self._workflows = {}

        return self._workflows

    def get_workflows(self) -> Dict[str, WorkflowDefinition]:
        """Get all workflow definitions for this vertical.

        Returns:
            Dict mapping workflow names to WorkflowDefinition instances
        """
        return self._load_workflows()

    def get_workflow(self, name: str) -> Optional[WorkflowDefinition]:
        """Get a specific workflow by name.

        Args:
            name: The workflow name to retrieve

        Returns:
            WorkflowDefinition if found, None otherwise
        """
        return self._load_workflows().get(name)

    def get_workflow_names(self) -> List[str]:
        """Get list of all available workflow names.

        Returns:
            List of workflow name strings
        """
        return list(self._load_workflows().keys())

    def get_auto_workflows(self) -> List[Tuple[str, str]]:
        """Get automatic workflow triggers based on query patterns.

        Override this method in subclasses to define patterns that
        automatically trigger specific workflows based on user input.

        Returns:
            List of (regex_pattern, workflow_name) tuples

        Example:
            def get_auto_workflows(self) -> List[Tuple[str, str]]:
                return [
                    (r"deep\\s+research", "deep_research"),
                    (r"fact\\s*check", "fact_check"),
                ]
        """
        return []

    def get_workflow_for_task_type(self, task_type: str) -> Optional[str]:
        """Get recommended workflow for a task type.

        Override this method in subclasses to map task types to workflow names.

        Args:
            task_type: Type of task (e.g., "research", "deploy", "eda")

        Returns:
            Workflow name string or None if no mapping exists

        Example:
            def get_workflow_for_task_type(self, task_type: str) -> Optional[str]:
                mapping = {
                    "research": "deep_research",
                    "fact_check": "fact_check",
                }
                return mapping.get(task_type.lower())
        """
        return None

    # =========================================================================
    # UnifiedWorkflowCompiler Integration (Recommended Pattern)
    # =========================================================================

    def get_compiler(self) -> "UnifiedWorkflowCompiler":
        """Get or create the unified compiler with caching.

        Returns the UnifiedWorkflowCompiler instance for this provider,
        creating it lazily on first access. The compiler provides consistent
        caching across all workflow compilations.

        **Architecture Note**: This uses UnifiedWorkflowCompiler (not the plugin API)
        because vertical providers need:
        - Escape hatch integration (condition_registry, transform_registry)
        - Two-level caching (definition + execution)
        - Cache management APIs (get_cache_stats, clear_cache)

        For simple workflow compilation without these features, consider using
        the plugin API: create_compiler("workflow.yaml", enable_caching=True)

        Returns:
            UnifiedWorkflowCompiler configured with escape hatches

        Example:
            provider = ResearchWorkflowProvider()
            compiler = provider.get_compiler()
            stats = compiler.get_cache_stats()
        """
        if self._compiler is None:
            from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

            self._compiler = UnifiedWorkflowCompiler(enable_caching=True)
        return self._compiler

    def _get_workflow_path(self, workflow_name: str) -> Path:
        """Get the path to a specific workflow YAML file.

        Args:
            workflow_name: Name of the workflow

        Returns:
            Path to the workflow YAML file

        Raises:
            ValueError: If workflow not found
        """
        workflows_dir = self._get_workflows_directory()

        # Try exact filename match first
        exact_path = workflows_dir / f"{workflow_name}.yaml"
        if exact_path.exists():
            return exact_path

        # Try finding in loaded workflows
        workflow = self.get_workflow(workflow_name)
        if workflow is None:
            raise ValueError(f"Workflow not found: {workflow_name}")

        # Default to workflows directory with .yaml extension
        return workflows_dir / f"{workflow_name}.yaml"

    def compile_workflow(self, workflow_name: str) -> "CachedCompiledGraph":
        """Compile a workflow using the unified compiler.

        This is the recommended method for compiling workflows. It uses
        the UnifiedWorkflowCompiler for consistent caching behavior.

        Args:
            workflow_name: Name of the workflow to compile

        Returns:
            CachedCompiledGraph ready for execution with invoke() and stream()

        Raises:
            ValueError: If workflow not found

        Example:
            provider = ResearchWorkflowProvider()
            compiled = provider.compile_workflow("deep_research")
            result = await compiled.invoke({"query": "AI trends"})
        """
        conditions, transforms = self._load_escape_hatches()
        workflow_path = self._get_workflow_path(workflow_name)

        return self.get_compiler().compile_yaml(
            workflow_path,
            workflow_name,
            condition_registry=conditions,
            transform_registry=transforms,
        )

    async def run_compiled_workflow(
        self,
        workflow_name: str,
        context: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> "ExecutionResult":
        """Execute a workflow using the unified compiler.

        This method compiles the workflow using the UnifiedWorkflowCompiler
        and executes it. Results benefit from consistent caching.

        Args:
            workflow_name: Name of the workflow to execute
            context: Initial context data for the workflow
            thread_id: Thread ID for checkpointing

        Returns:
            ExecutionResult with final state

        Raises:
            ValueError: If workflow not found

        Example:
            provider = ResearchWorkflowProvider()
            result = await provider.run_compiled_workflow(
                "fact_check",
                {"claim": "The Earth is round"}
            )
            print(result.state)
        """
        compiled = self.compile_workflow(workflow_name)
        return await compiled.invoke(context or {}, thread_id=thread_id)

    async def stream_compiled_workflow(
        self,
        workflow_name: str,
        context: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[tuple]:
        """Stream workflow execution using the unified compiler.

        This method compiles the workflow using the UnifiedWorkflowCompiler
        and streams its execution. Yields (node_id, state) tuples after
        each node completes.

        Args:
            workflow_name: Name of the workflow to execute
            context: Initial context data for the workflow
            thread_id: Thread ID for checkpointing

        Yields:
            Tuple of (node_id, state) after each node execution

        Raises:
            ValueError: If workflow not found

        Example:
            provider = ResearchWorkflowProvider()
            async for node_id, state in provider.stream_compiled_workflow(
                "deep_research",
                {"query": "AI trends"}
            ):
                print(f"Completed: {node_id}")
        """
        compiled = self.compile_workflow(workflow_name)
        async for node_id, state in compiled.stream(context or {}, thread_id=thread_id):
            yield node_id, state

    # =========================================================================
    # Legacy Executor Methods (Deprecated)
    # =========================================================================

    def create_executor(
        self,
        orchestrator: "AgentOrchestrator",
    ) -> "WorkflowExecutor":
        """Create a standard workflow executor.

        .. deprecated::
            Use compile_workflow() and invoke() instead for consistent caching.

        Args:
            orchestrator: Agent orchestrator instance for LLM interactions

        Returns:
            WorkflowExecutor configured for this orchestrator
        """
        warnings.warn(
            "create_executor() is deprecated. Use compile_workflow() and invoke() "
            "for consistent caching via UnifiedWorkflowCompiler.",
            DeprecationWarning,
            stacklevel=2,
        )
        from victor.workflows.executor import WorkflowExecutor

        return WorkflowExecutor(orchestrator)

    def create_streaming_executor(
        self,
        orchestrator: "AgentOrchestrator",
    ) -> "StreamingWorkflowExecutor":
        """Create a streaming workflow executor.

        .. deprecated::
            Use compile_workflow() and stream() instead for consistent caching.

        Args:
            orchestrator: Agent orchestrator instance for LLM interactions

        Returns:
            StreamingWorkflowExecutor for real-time progress streaming
        """
        warnings.warn(
            "create_streaming_executor() is deprecated. Use compile_workflow() and stream() "
            "for consistent caching via UnifiedWorkflowCompiler.",
            DeprecationWarning,
            stacklevel=2,
        )
        from victor.workflows.streaming_executor import StreamingWorkflowExecutor

        return StreamingWorkflowExecutor(orchestrator)

    async def astream(
        self,
        workflow_name: str,
        orchestrator: "AgentOrchestrator",
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator["WorkflowStreamChunk"]:
        """Stream workflow execution with real-time events.

        .. deprecated::
            Use stream_compiled_workflow() instead for consistent caching.

        Convenience method that creates a streaming executor and
        streams the specified workflow. Yields progress events
        as the workflow executes.

        Args:
            workflow_name: Name of the workflow to execute
            orchestrator: Agent orchestrator instance
            context: Initial context data for the workflow

        Yields:
            WorkflowStreamChunk events during execution

        Raises:
            ValueError: If workflow_name is not found

        Example:
            provider = ResearchWorkflowProvider()
            async for chunk in provider.astream("fact_check", orchestrator, {}):
                if chunk.event_type == WorkflowEventType.NODE_START:
                    print(f"Starting: {chunk.node_name}")
                elif chunk.event_type == WorkflowEventType.NODE_COMPLETE:
                    print(f"Completed: {chunk.node_name}")
        """
        warnings.warn(
            "astream() is deprecated. Use stream_compiled_workflow() "
            "for consistent caching via UnifiedWorkflowCompiler.",
            DeprecationWarning,
            stacklevel=2,
        )
        workflow = self.get_workflow(workflow_name)
        if not workflow:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        # Suppress deprecation warning for internal call
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            executor = self.create_streaming_executor(orchestrator)
        async for chunk in executor.astream(workflow, context or {}):
            yield chunk

    async def run_workflow(
        self,
        workflow_name: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> "WorkflowResult":
        """Execute a YAML workflow directly without requiring a full orchestrator.

        .. deprecated::
            Use run_compiled_workflow() instead for consistent caching.

        This method is designed for compute-only workflows that use registered
        handlers. For workflows with agent nodes, use create_executor() with
        a proper orchestrator or astream().

        The workflow DAG is executed with:
        - Compute nodes: Invoke registered handlers
        - Parallel nodes: Execute child nodes concurrently
        - Condition nodes: Evaluate escape hatches for branching
        - Transform nodes: Apply data transformations

        Note: Agent nodes will fail with this method. Use astream() or
        create_executor() with a proper orchestrator for workflows
        containing agent nodes.

        Args:
            workflow_name: Name of the YAML workflow to execute
            context: Initial context data (e.g., {"symbol": "AAPL"})
            timeout: Optional overall timeout in seconds (default: 300)

        Returns:
            WorkflowResult with execution outcome and outputs

        Raises:
            ValueError: If workflow_name is not found

        Example:
            provider = InvestmentWorkflowProvider()
            result = await provider.run_workflow(
                "comprehensive",
                {"symbol": "AAPL"}
            )
            if result.success:
                synthesis = result.context.get("synthesis")
                print(f"Recommendation: {synthesis.get('recommendation')}")
        """
        warnings.warn(
            "run_workflow() is deprecated. Use run_compiled_workflow() "
            "for consistent caching via UnifiedWorkflowCompiler.",
            DeprecationWarning,
            stacklevel=2,
        )
        from victor.workflows.executor import WorkflowExecutor
        from victor.tools.registry import ToolRegistry

        workflow = self.get_workflow(workflow_name)
        if not workflow:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        # Create minimal orchestrator for compute-only workflows
        # Agent nodes would fail with this mock, but compute handlers work fine
        orchestrator = _MinimalOrchestrator()

        # Create tool registry for handlers that need it
        tool_registry = ToolRegistry()

        # Create executor with minimal orchestrator and explicit tool registry
        executor = WorkflowExecutor(
            orchestrator,
            tool_registry=tool_registry,
            default_timeout=timeout or 300.0,
        )

        # Execute workflow with initial context
        return await executor.execute(
            workflow,
            initial_context=context or {},
            timeout=timeout,
        )

    def __repr__(self) -> str:
        """Return a string representation of this provider."""
        class_name = self.__class__.__name__
        workflow_count = len(self._load_workflows())
        return f"{class_name}(workflows={workflow_count})"


__all__ = [
    "BaseYAMLWorkflowProvider",
]
