"""SDK host adapters for LSP runtime types and protocols."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from victor.framework.lsp import (
        CompletionItem,
        CompletionItemKind,
        Diagnostic,
        DiagnosticRelatedInformation,
        DiagnosticSeverity,
        DiagnosticTag,
        DocumentSymbol,
        Hover,
        Location,
        LocationLink,
        Position,
        Range,
        SymbolInformation,
        SymbolKind,
        TextDocumentEdit,
        TextDocumentIdentifier,
        TextEdit,
        VersionedTextDocumentIdentifier,
    )
    from victor.framework.lsp_protocols import LSPPoolProtocol, LSPServiceProtocol

__all__ = [
    "DiagnosticSeverity",
    "CompletionItemKind",
    "SymbolKind",
    "DiagnosticTag",
    "Position",
    "Range",
    "Location",
    "LocationLink",
    "DiagnosticRelatedInformation",
    "Diagnostic",
    "CompletionItem",
    "Hover",
    "DocumentSymbol",
    "SymbolInformation",
    "TextEdit",
    "TextDocumentIdentifier",
    "VersionedTextDocumentIdentifier",
    "TextDocumentEdit",
    "LSPServiceProtocol",
    "LSPPoolProtocol",
]

_LAZY_IMPORTS = {
    "DiagnosticSeverity": "victor.framework.lsp",
    "CompletionItemKind": "victor.framework.lsp",
    "SymbolKind": "victor.framework.lsp",
    "DiagnosticTag": "victor.framework.lsp",
    "Position": "victor.framework.lsp",
    "Range": "victor.framework.lsp",
    "Location": "victor.framework.lsp",
    "LocationLink": "victor.framework.lsp",
    "DiagnosticRelatedInformation": "victor.framework.lsp",
    "Diagnostic": "victor.framework.lsp",
    "CompletionItem": "victor.framework.lsp",
    "Hover": "victor.framework.lsp",
    "DocumentSymbol": "victor.framework.lsp",
    "SymbolInformation": "victor.framework.lsp",
    "TextEdit": "victor.framework.lsp",
    "TextDocumentIdentifier": "victor.framework.lsp",
    "VersionedTextDocumentIdentifier": "victor.framework.lsp",
    "TextDocumentEdit": "victor.framework.lsp",
    "LSPServiceProtocol": "victor.framework.lsp_protocols",
    "LSPPoolProtocol": "victor.framework.lsp_protocols",
}


def __getattr__(name: str):
    """Resolve LSP helpers lazily from the Victor host runtime."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'victor_sdk.lsp_runtime' has no attribute {name!r}")

    module = importlib.import_module(module_name)
    return getattr(module, name)
