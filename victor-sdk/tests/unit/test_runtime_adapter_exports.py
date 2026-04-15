"""Tests for processing, LSP, search, and RL runtime adapters."""

from victor_sdk.capability_runtime import (
    EditorProtocol,
    TreeSitterParserProtocol,
    create_lazy_capability_proxy,
    detect_enhanced_index_factory,
)
from victor_sdk.chain_runtime import ChainRegistry, get_chain_registry
from victor_sdk.init_runtime import InitSynthesizer
from victor_sdk.lsp_runtime import LSPServiceProtocol
from victor_sdk.processing_runtime import (
    CompletionItem,
    CompletionItemKind,
    EditTransaction,
    Position,
    get_default_text_chunker,
)
from victor_sdk.rl_runtime import RLManager, get_rl_coordinator
from victor_sdk.provider_runtime import Message, ProviderRegistry
from victor_sdk.search_runtime import QueryExpander, QueryExpansionConfig


def test_processing_runtime_exports_host_helpers() -> None:
    assert EditTransaction.__name__ == "EditTransaction"
    assert CompletionItem.__name__ == "CompletionItem"
    assert Position.__name__ == "Position"
    assert CompletionItemKind.__name__ == "CompletionItemKind"
    assert callable(get_default_text_chunker)


def test_lsp_runtime_exports_host_helpers() -> None:
    assert Position.__name__ == "Position"
    assert LSPServiceProtocol.__name__ == "LSPServiceProtocol"


def test_search_runtime_exports_host_helpers() -> None:
    assert QueryExpander.__name__ == "QueryExpander"
    assert QueryExpansionConfig.__name__ == "QueryExpansionConfig"


def test_rl_runtime_exports_host_helpers() -> None:
    assert RLManager.__name__ == "RLManager"
    assert callable(get_rl_coordinator)


def test_capability_runtime_exports_host_helpers() -> None:
    assert TreeSitterParserProtocol.__name__ == "TreeSitterParserProtocol"
    assert EditorProtocol.__name__ == "EditorProtocol"
    assert callable(create_lazy_capability_proxy)
    assert callable(detect_enhanced_index_factory)


def test_chain_provider_and_init_runtime_exports_host_helpers() -> None:
    assert ChainRegistry.__name__ == "ChainRegistry"
    assert callable(get_chain_registry)
    assert InitSynthesizer.__name__ == "InitSynthesizer"
    assert Message.__name__ == "Message"
    assert ProviderRegistry.__name__ == "ProviderRegistry"
