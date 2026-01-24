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

"""Agent module - orchestrator and supporting components.

This module uses lazy loading for all components to improve startup time.
Components are loaded on-demand via __getattr__(), providing the same API
with faster import times.

Startup optimization: All imports are deferred until first access.
"""

from typing import Any

# NO other imports at module level - everything is lazy-loaded via __getattr__()
# This significantly improves `import victor` startup time


def __getattr__(name: str) -> Any:
    """Lazy import agent components on first access.

    This function is called by Python when an attribute is not found in the module.
    It dynamically imports the requested component, providing transparent lazy loading.

    Args:
        name: The name of the component to import

    Returns:
        The requested component

    Raises:
        AttributeError: If the component is not recognized
    """
    import importlib

    # Map of component names to their (module_path, attribute_name) tuples
    # Only import the specific module when the component is accessed
    lazy_imports = {
        # Core configuration
        "UnifiedAgentConfig": ("victor.agent.config", "UnifiedAgentConfig"),
        "AgentMode": ("victor.agent.config", "AgentMode"),
        # Orchestrator
        "AgentOrchestrator": ("victor.agent.orchestrator", "AgentOrchestrator"),
        # Argument normalization
        "ArgumentNormalizer": ("victor.agent.argument_normalizer", "ArgumentNormalizer"),
        "NormalizationStrategy": ("victor.agent.argument_normalizer", "NormalizationStrategy"),
        # Message history
        "MessageHistory": ("victor.agent.message_history", "MessageHistory"),
        # Tool selection
        "get_critical_tools": ("victor.agent.tool_selection", "get_critical_tools"),
        # Stream handling
        "StreamHandler": ("victor.agent.stream_handler", "StreamHandler"),
        "StreamResult": ("victor.agent.stream_handler", "StreamResult"),
        "StreamMetrics": ("victor.agent.stream_handler", "StreamMetrics"),
        "StreamBuffer": ("victor.agent.stream_handler", "StreamBuffer"),
        # Tool execution
        "ToolExecutor": ("victor.agent.tool_executor", "ToolExecutor"),
        "ToolExecutionResult": ("victor.agent.tool_executor", "ToolExecutionResult"),
        # Conversation control
        "ConversationController": (
            "victor.agent.conversation_controller",
            "ConversationController",
        ),
        "ConversationConfig": ("victor.agent.conversation_controller", "ConversationConfig"),
        "ContextMetrics": ("victor.agent.conversation_controller", "ContextMetrics"),
        # Tool pipeline
        "ToolPipeline": ("victor.agent.tool_pipeline", "ToolPipeline"),
        "ToolPipelineConfig": ("victor.agent.tool_pipeline", "ToolPipelineConfig"),
        "ToolCallResult": ("victor.agent.tool_pipeline", "ToolCallResult"),
        "PipelineExecutionResult": ("victor.agent.tool_pipeline", "PipelineExecutionResult"),
        # Streaming controller
        "StreamingController": ("victor.agent.streaming_controller", "StreamingController"),
        "StreamingControllerConfig": (
            "victor.agent.streaming_controller",
            "StreamingControllerConfig",
        ),
        "StreamingSession": ("victor.agent.streaming_controller", "StreamingSession"),
        # Task analyzer
        "TaskAnalyzer": ("victor.agent.task_analyzer", "TaskAnalyzer"),
        "TaskAnalysis": ("victor.agent.task_analyzer", "TaskAnalysis"),
        "get_task_analyzer": ("victor.agent.task_analyzer", "get_task_analyzer"),
        "reset_task_analyzer": ("victor.agent.task_analyzer", "reset_task_analyzer"),
        # Configuration manager
        "ConfigurationManager": ("victor.agent.configuration_manager", "ConfigurationManager"),
        "create_configuration_manager": (
            "victor.agent.configuration_manager",
            "create_configuration_manager",
        ),
        "get_configuration_manager": (
            "victor.agent.configuration_manager",
            "get_configuration_manager",
        ),
        "reset_configuration_manager": (
            "victor.agent.configuration_manager",
            "reset_configuration_manager",
        ),
        # Memory manager
        "MemoryManager": ("victor.agent.memory_manager", "MemoryManager"),
        "SessionRecoveryManager": ("victor.agent.memory_manager", "SessionRecoveryManager"),
        "create_memory_manager": ("victor.agent.memory_manager", "create_memory_manager"),
        "create_session_recovery_manager": (
            "victor.agent.memory_manager",
            "create_session_recovery_manager",
        ),
        # Search router
        "SearchRouter": ("victor.agent.search_router", "SearchRouter"),
        "SearchRoute": ("victor.agent.search_router", "SearchRoute"),
        "SearchType": ("victor.agent.search_router", "SearchType"),
        "route_query": ("victor.agent.search_router", "route_query"),
        "suggest_search_tool": ("victor.agent.search_router", "suggest_search_tool"),
        "is_keyword_query": ("victor.agent.search_router", "is_keyword_query"),
        "is_semantic_query": ("victor.agent.search_router", "is_semantic_query"),
        # Intelligent pipeline
        "IntelligentAgentPipeline": (
            "victor.agent.intelligent_pipeline",
            "IntelligentAgentPipeline",
        ),
        "RequestContext": ("victor.agent.intelligent_pipeline", "RequestContext"),
        "ResponseResult": ("victor.agent.intelligent_pipeline", "ResponseResult"),
        "PipelineStats": ("victor.agent.intelligent_pipeline", "PipelineStats"),
        "get_pipeline": ("victor.agent.intelligent_pipeline", "get_pipeline"),
        "clear_pipeline_cache": ("victor.agent.intelligent_pipeline", "clear_pipeline_cache"),
        # Orchestrator integration
        "OrchestratorIntegration": (
            "victor.agent.orchestrator_integration",
            "OrchestratorIntegration",
        ),
        "IntegrationConfig": ("victor.agent.orchestrator_integration", "IntegrationConfig"),
        "IntegrationMetrics": ("victor.agent.orchestrator_integration", "IntegrationMetrics"),
        "enhance_orchestrator": ("victor.agent.orchestrator_integration", "enhance_orchestrator"),
        # Agentic AI (Phase 3)
        "HierarchicalPlanner": (
            "victor.agent.planning.hierarchical_planner",
            "HierarchicalPlanner",
        ),
        "AutonomousPlanner": ("victor.agent.planning.autonomous", "AutonomousPlanner"),
        "ExecutionPlan": ("victor.agent.planning.base", "ExecutionPlan"),
        "PlanResult": ("victor.agent.planning.base", "PlanResult"),
        "PlanStep": ("victor.agent.planning.base", "PlanStep"),
        "TaskDecomposition": ("victor.agent.planning.task_decomposition", "TaskDecomposition"),
        "TaskGraph": ("victor.agent.planning.task_decomposition", "TaskGraph"),
        "EpisodicMemory": ("victor.agent.memory.episodic_memory", "EpisodicMemory"),
        "Episode": ("victor.agent.memory.episodic_memory", "Episode"),
        "create_episodic_memory": ("victor.agent.memory.episodic_memory", "create_episodic_memory"),
        "SemanticMemory": ("victor.agent.memory.semantic_memory", "SemanticMemory"),
        "Knowledge": ("victor.agent.memory.semantic_memory", "Knowledge"),
        "SkillDiscoveryEngine": ("victor.agent.skills.skill_discovery", "SkillDiscoveryEngine"),
        "Skill": ("victor.agent.skills.skill_discovery", "Skill"),
        "SkillChainer": ("victor.agent.skills.skill_chaining", "SkillChainer"),
        "SkillChain": ("victor.agent.skills.skill_chaining", "SkillChain"),
        "ProficiencyTracker": (
            "victor.agent.improvement.proficiency_tracker",
            "ProficiencyTracker",
        ),
        "EnhancedRLCoordinator": (
            "victor.agent.improvement.rl_coordinator",
            "EnhancedRLCoordinator",
        ),
    }

    if name in lazy_imports:
        module_path, attr_name = lazy_imports[name]
        module = importlib.import_module(module_path)
        # Cache the imported attribute in module globals for faster subsequent access
        globals()[name] = getattr(module, attr_name)
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentOrchestrator",
    "ArgumentNormalizer",
    "NormalizationStrategy",
    # "ConfigLoader",  # OBSOLETE - moved to archive/obsolete/
    "UnifiedAgentConfig",
    "AgentMode",
    "get_critical_tools",
    # Conversation
    "MessageHistory",
    "ConversationController",
    "ConversationConfig",
    "ContextMetrics",
    # Observability (OBSOLETE - use victor.core.events instead)
    # "TracingProvider",
    # "Span",
    # "SpanKind",
    # "SpanStatus",
    # "get_observability",
    # "set_observability",
    # "traced",
    # Streaming
    "StreamHandler",
    "StreamResult",
    "StreamMetrics",
    "StreamBuffer",
    "StreamingController",
    "StreamingControllerConfig",
    "StreamingSession",
    # Tool execution
    "ToolExecutor",
    "ToolExecutionResult",
    "ToolPipeline",
    "ToolPipelineConfig",
    "ToolCallResult",
    "PipelineExecutionResult",
    # Task analysis
    "TaskAnalyzer",
    "TaskAnalysis",
    "get_task_analyzer",
    "reset_task_analyzer",
    # Phase 1 extraction components (Task 1)
    "ConfigurationManager",
    "create_configuration_manager",
    "get_configuration_manager",
    "reset_configuration_manager",
    "MemoryManager",
    "SessionRecoveryManager",
    "create_memory_manager",
    "create_session_recovery_manager",
    "SearchRouter",
    "SearchRoute",
    "SearchType",
    "route_query",
    "suggest_search_tool",
    "is_keyword_query",
    "is_semantic_query",
    # Intelligent Pipeline
    "IntelligentAgentPipeline",
    "RequestContext",
    "ResponseResult",
    "PipelineStats",
    "get_pipeline",
    "clear_pipeline_cache",
    # Orchestrator Integration
    "OrchestratorIntegration",
    "IntegrationConfig",
    "IntegrationMetrics",
    "enhance_orchestrator",
    # Agentic AI (Phase 3)
    "HierarchicalPlanner",
    "AutonomousPlanner",
    "ExecutionPlan",
    "PlanResult",
    "PlanStep",
    "TaskDecomposition",
    "TaskGraph",
    "EpisodicMemory",
    "Episode",
    "create_episodic_memory",
    "SemanticMemory",
    "Knowledge",
    "SkillDiscoveryEngine",
    "Skill",
    "SkillChainer",
    "SkillChain",
    "ProficiencyTracker",
    "EnhancedRLCoordinator",
]
