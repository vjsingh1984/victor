"""Tests for plugin context protocol surface."""

from victor_sdk import PluginContext


def test_plugin_context_exposes_workflow_node_executor_registration() -> None:
    assert hasattr(PluginContext, "register_workflow_node_executor")
