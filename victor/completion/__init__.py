# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""This module has moved to victor.processing.completion.

This module is maintained for backward compatibility only.
Please update your imports to use the new location:

    # OLD:
    from victor.completion import CompletionManager, CompletionParams

    # NEW (preferred):
    from victor.processing.completion import CompletionManager, CompletionParams
"""

# Re-export everything from the new location for backward compatibility
from victor.processing.completion import (
    # Protocol types
    CompletionCapabilities,
    CompletionContext,
    CompletionItem,
    CompletionItemKind,
    CompletionItemLabelDetails,
    CompletionList,
    CompletionMetrics,
    CompletionParams,
    CompletionTriggerKind,
    InlineCompletionItem,
    InlineCompletionList,
    InlineCompletionParams,
    InsertTextFormat,
    Position,
    Range,
    TextEdit,
    # Provider interface
    BaseCompletionProvider,
    CompletionProvider,
    # Registry and manager
    CompletionProviderRegistry,
    CompletionManager,
    # Built-in providers
    AICompletionProvider,
    LSPCompletionProvider,
    SnippetCompletionProvider,
)

__all__ = [
    # Protocol types
    "CompletionCapabilities",
    "CompletionContext",
    "CompletionItem",
    "CompletionItemKind",
    "CompletionItemLabelDetails",
    "CompletionList",
    "CompletionMetrics",
    "CompletionParams",
    "CompletionTriggerKind",
    "InlineCompletionItem",
    "InlineCompletionList",
    "InlineCompletionParams",
    "InsertTextFormat",
    "Position",
    "Range",
    "TextEdit",
    # Provider interface
    "BaseCompletionProvider",
    "CompletionProvider",
    # Registry and manager
    "CompletionProviderRegistry",
    "CompletionManager",
    # Built-in providers
    "AICompletionProvider",
    "LSPCompletionProvider",
    "SnippetCompletionProvider",
]
