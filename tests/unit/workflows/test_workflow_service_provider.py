import asyncio
from types import SimpleNamespace

from victor.agent.protocols import ToolRegistryProtocol
from victor.config.settings import Settings
from victor.core.bootstrap import bootstrap_container
from victor.core.plugins.context import HostPluginContext
from victor.workflows.compiler_protocols import (
    ExecutionContextProtocol,
    NodeExecutorFactoryProtocol,
)
from victor.workflows.executors.registry import clear_registered_workflow_node_executors
from victor.workflows.orchestrator_pool import OrchestratorPool


def test_workflow_service_provider_passes_container_to_container_aware_specs() -> None:
    container = bootstrap_container(Settings())

    pool = container.get(OrchestratorPool)
    assert pool is not None
    assert getattr(pool, "_container", None) is container

    root_context = container.get_optional(ExecutionContextProtocol)
    assert root_context is not None
    assert root_context.services is container

    with container.create_scope() as scope:
        scoped_context = scope.get(ExecutionContextProtocol)
        assert scoped_context is not None
        assert scoped_context.services is container


def test_node_executor_factory_can_resolve_registered_executor_context_from_container() -> (
    None
):
    container = bootstrap_container(Settings())

    factory = container.get(NodeExecutorFactoryProtocol)
    context = factory._resolve_execution_context()

    assert context is not None
    assert context.services is container


def test_plugin_context_registers_custom_workflow_node_executors_for_factory() -> None:
    clear_registered_workflow_node_executors()

    try:
        container = bootstrap_container(Settings())

        class CustomExecutor:
            def __init__(self, context=None):
                self.context = context

            async def execute(self, node, state):
                return {**state, "custom_registered": node.id}

        context = HostPluginContext(container)
        context.register_workflow_node_executor("custom_plugin", CustomExecutor)

        factory = container.get(NodeExecutorFactoryProtocol)
        node = SimpleNamespace(
            id="custom",
            name="Custom",
            node_type=SimpleNamespace(value="custom_plugin"),
        )

        result = asyncio.run(factory.create_executor(node)({"value": 1}))

        assert factory.supports_node_type("custom_plugin")
        assert result["custom_registered"] == "custom"
    finally:
        clear_registered_workflow_node_executors()


def test_plugin_context_register_tool_uses_live_container_registry() -> None:
    class _Registry:
        def __init__(self) -> None:
            self.tools = []

        def register_tool(self, tool) -> None:
            self.tools.append(tool)

    class _Container:
        def __init__(self, registry) -> None:
            self.registry = registry

        def get_optional(self, service_type):
            if service_type is ToolRegistryProtocol:
                return self.registry
            return None

    registry = _Registry()
    context = HostPluginContext(_Container(registry))
    tool = SimpleNamespace(name="plugin_tool")

    context.register_tool(tool)

    assert registry.tools == [tool]
