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

"""Stable public API for LSP types used by external verticals.

External verticals should import LSP protocol types through this module
rather than from victor.protocols.lsp_types directly.

Usage:
    from victor.framework.lsp import (
        Position,
        Range,
        Diagnostic,
        CompletionItemKind,
    )
"""

from __future__ import annotations

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
]


def __getattr__(name: str):
    """Lazy imports to avoid circular dependencies and unnecessary loading."""
    _ALL_NAMES = set(__all__)

    if name in _ALL_NAMES:
        import importlib

        module = importlib.import_module("victor.protocols.lsp_types")
        return getattr(module, name)

    raise AttributeError(f"module 'victor.framework.lsp' has no attribute {name!r}")
