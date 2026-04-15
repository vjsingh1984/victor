"""Interop tests for SDK runtime adapter seams."""

from victor_sdk.agent_spec_runtime import AgentSpec
from victor_sdk.capability_runtime import CodebaseIndexFactoryProtocol, create_lazy_capability_proxy
from victor_sdk.chain_runtime import get_chain_registry
from victor_sdk.handler_runtime import BaseHandler
from victor_sdk.init_runtime import InitSynthesizer
from victor_sdk.lsp_runtime import CompletionItemKind
from victor_sdk.processing_runtime import FileEditor
from victor_sdk.provider_runtime import ProviderRegistry
from victor_sdk.rl_runtime import RLManager
from victor_sdk.search_runtime import QueryExpander
from victor_sdk.subagent_runtime import set_role_tool_provider
from victor_sdk.workflow_executor_runtime import WorkflowExecutor


def test_sdk_runtime_adapters_resolve_host_types() -> None:
    assert FileEditor.__name__ == "FileEditor"
    assert CompletionItemKind.__name__ == "CompletionItemKind"
    assert QueryExpander.__name__ == "QueryExpander"
    assert RLManager.__name__ == "RLManager"
    assert CodebaseIndexFactoryProtocol.__name__ == "CodebaseIndexFactoryProtocol"
    assert callable(create_lazy_capability_proxy)
    assert callable(get_chain_registry)
    assert InitSynthesizer.__name__ == "InitSynthesizer"
    assert ProviderRegistry.__name__ == "ProviderRegistry"
    assert AgentSpec.__name__ == "AgentSpec"
    assert callable(set_role_tool_provider)
    assert WorkflowExecutor.__name__ == "WorkflowExecutor"
    assert BaseHandler.__name__ == "BaseHandler"
