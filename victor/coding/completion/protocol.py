# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Completion protocol types - re-exported from canonical location.

This module re-exports all completion protocol types from the canonical
source at victor.completion.protocol for backward compatibility.

All new code should import directly from victor.completion.protocol.
"""

# Re-export everything from canonical location
from victor.completion.protocol import (
    # LSP base types (originally from victor.protocols.lsp_types)
    Position,
    Range,
    TextEdit,
    CompletionItemKind,
    # Completion-specific types
    InsertTextFormat,
    CompletionTriggerKind,
    CompletionContext,
    CompletionParams,
    CompletionItemLabelDetails,
    CompletionItem,
    InlineCompletionItem,
    InlineCompletionParams,
    CompletionList,
    InlineCompletionList,
    CompletionCapabilities,
    CompletionMetrics,
)

__all__ = [
    "Position",
    "Range",
    "TextEdit",
    "CompletionItemKind",
    "InsertTextFormat",
    "CompletionTriggerKind",
    "CompletionContext",
    "CompletionParams",
    "CompletionItemLabelDetails",
    "CompletionItem",
    "InlineCompletionItem",
    "InlineCompletionParams",
    "CompletionList",
    "InlineCompletionList",
    "CompletionCapabilities",
    "CompletionMetrics",
]
