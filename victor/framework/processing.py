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

"""Stable public API for processing utilities used by external verticals.

External verticals should import editing, completion, and native processing
utilities through this module rather than from victor.processing.* directly.

Usage:
    from victor.framework.processing import (
        FileEditor,
        EditTransaction,
        get_default_text_chunker,
        CompletionItem,
    )
"""

from __future__ import annotations

__all__ = [
    # Editing
    "OperationType",
    "EditOperation",
    "EditTransaction",
    "FileEditor",
    # Native processing
    "get_default_text_chunker",
    # Completion protocol
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


def __getattr__(name: str):
    """Lazy imports to avoid circular dependencies and unnecessary loading."""
    _LAZY_IMPORTS = {
        # Editing
        "OperationType": "victor.processing.editing",
        "EditOperation": "victor.processing.editing",
        "EditTransaction": "victor.processing.editing",
        "FileEditor": "victor.processing.editing",
        # Native processing
        "get_default_text_chunker": "victor.processing.native",
        # Completion protocol
        "InsertTextFormat": "victor.processing.completion.protocol",
        "CompletionTriggerKind": "victor.processing.completion.protocol",
        "CompletionContext": "victor.processing.completion.protocol",
        "CompletionParams": "victor.processing.completion.protocol",
        "CompletionItemLabelDetails": "victor.processing.completion.protocol",
        "CompletionItem": "victor.processing.completion.protocol",
        "InlineCompletionItem": "victor.processing.completion.protocol",
        "InlineCompletionParams": "victor.processing.completion.protocol",
        "CompletionList": "victor.processing.completion.protocol",
        "InlineCompletionList": "victor.processing.completion.protocol",
        "CompletionCapabilities": "victor.processing.completion.protocol",
        "CompletionMetrics": "victor.processing.completion.protocol",
    }

    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)

    raise AttributeError(f"module 'victor.framework.processing' has no attribute {name!r}")
