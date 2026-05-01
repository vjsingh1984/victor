"""Tests for plugin context protocol surface."""

from victor_sdk import PluginContext


def test_plugin_context_exposes_workflow_node_executor_registration() -> None:
    assert hasattr(PluginContext, "register_workflow_node_executor")


def test_plugin_context_exposes_category_and_typed_service_helpers() -> None:
    for name in (
        "register_category",
        "extend_category",
        "get_provider_registry",
        "get_graph_store",
        "get_vector_store",
        "get_embedding_service",
        "get_memory_coordinator",
    ):
        assert hasattr(PluginContext, name)
