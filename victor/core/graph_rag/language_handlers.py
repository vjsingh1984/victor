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

"""Language-specific edge detection handlers for Graph RAG indexing.

This module establishes the SEAM between core graph indexing and
language-specific edge detection logic.

ARCHITECTURAL PRINCIPLE:
- Core indexing pipeline (indexing.py) should be language-agnostic
- Each language implements edge detection via LanguageEdgeHandler protocol
- Handlers are discovered via registry pattern (similar to victor_coding)

MIGRATION PATH (2026-04-29):
- Current: CALLS edge detection is hardcoded in indexing.py
- Target: Each language in victor-coding provides its own edge handler
- This module provides the protocol and registry for that migration

Future sessions should:
1. Implement handlers for additional languages (JS, TS, Go, Rust, etc.)
2. Move handler implementations to victor-coding package
3. Remove hardcoded edge detection from indexing.py once all languages migrate
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Node, Tree

logger = logging.getLogger(__name__)


@dataclass
class CallEdge:
    """A detected function call edge.

    Attributes:
        caller_name: Name of the calling function/symbol
        callee_name: Name of the called function/symbol
        caller_line: Line number where call occurs (for debugging)
    """

    caller_name: str
    callee_name: str
    caller_line: Optional[int] = None


@dataclass
class EdgeDetectionResult:
    """Result from edge detection for a file.

    Attributes:
        calls: List of CALLS edges detected
        metadata: Additional metadata (language-specific)
    """

    calls: List[CallEdge]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LanguageEdgeHandler(Protocol):
    """Protocol for language-specific edge detection.

    Each language implements this protocol to provide:
    - CALLS edge detection (function/method calls)
    - REFERENCES edge detection (variable/type references)
    - Other language-specific relationships

    Implementation pattern:
        1. Create handler class in handlers/ subdirectory
        2. Register via register_edge_handler()
        3. Core indexing pipeline discovers and uses handler by language name
    """

    def get_supported_languages(self) -> List[str]:
        """Get list of language identifiers this handler supports.

        Returns:
            List of language names (e.g., ["python", "py"])
        """
        ...

    async def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in parsed source code.

        Args:
            tree: Parsed tree-sitter tree
            source_code: Raw source code text
            file_path: Path to source file

        Returns:
            EdgeDetectionResult with detected calls
        """
        ...

    def can_handle(self, language: str) -> bool:
        """Check if this handler can process the given language.

        Args:
            language: Language identifier

        Returns:
            True if handler supports this language
        """
        return language.lower() in [lang.lower() for lang in self.get_supported_languages()]


class EdgeHandlerRegistry:
    """Registry for language edge handlers.

    Provides:
    - Handler registration by language name
    - Handler discovery via language identifier
    - Fallback to generic handler when no specific handler exists

    Pattern follows victor_coding.languages.registry for consistency.
    """

    _handlers: Dict[str, LanguageEdgeHandler] = {}
    _fallback_handler: Optional[LanguageEdgeHandler] = None

    @classmethod
    def register(
        cls,
        handler: LanguageEdgeHandler,
        languages: List[str],
    ) -> None:
        """Register an edge handler for one or more languages.

        Args:
            handler: Edge handler instance
            languages: List of language identifiers this handler supports

        Example:
            registry.register(PythonEdgeHandler(), ["python", "py"])
        """
        for lang in languages:
            lang_lower = lang.lower()
            cls._handlers[lang_lower] = handler
            logger.debug(f"Registered edge handler for language: {lang_lower}")

    @classmethod
    def get_handler(cls, language: str) -> Optional[LanguageEdgeHandler]:
        """Get edge handler for a specific language.

        Args:
            language: Language identifier

        Returns:
            Handler instance or None if not registered
        """
        return cls._handlers.get(language.lower())

    @classmethod
    def get_or_fallback(cls, language: str) -> Optional[LanguageEdgeHandler]:
        """Get handler or fallback to generic handler.

        Args:
            language: Language identifier

        Returns:
            Specific handler, fallback handler, or None
        """
        handler = cls.get_handler(language)
        if handler is not None:
            return handler
        return cls._fallback_handler

    @classmethod
    def set_fallback(cls, handler: LanguageEdgeHandler) -> None:
        """Set the fallback handler for unsupported languages.

        Args:
            handler: Generic edge handler
        """
        cls._fallback_handler = handler
        logger.debug("Registered fallback edge handler")

    @classmethod
    def supported_languages(cls) -> List[str]:
        """Get list of all supported languages.

        Returns:
            List of language identifiers
        """
        return list(set(cls._handlers.keys()))


def register_edge_handler(
    handler: LanguageEdgeHandler,
    languages: List[str],
) -> None:
    """Register an edge handler.

    Convenience function for handler registration.

    Args:
        handler: Edge handler instance
        languages: List of language identifiers

    Example:
        register_edge_handler(PythonEdgeHandler(), ["python", "py"])
    """
    EdgeHandlerRegistry.register(handler, languages)


def get_edge_handler(language: str) -> Optional[LanguageEdgeHandler]:
    """Get edge handler for a language.

    Uses victor_coding language plugins as the canonical source
    for language-specific edge detection.

    Args:
        language: Language identifier

    Returns:
        Handler instance or None if not found
    """
    return _get_victor_coding_handler(language)


def _get_victor_coding_handler(language: str) -> Optional[LanguageEdgeHandler]:
    """Get edge handler from victor_coding language plugins.

    This provides the proper seam: victor-ai core discovers and uses
    victor_coding language plugins for edge detection.

    Args:
        language: Language identifier

    Returns:
        Handler instance or None if not found
    """
    try:
        from victor_coding.languages.registry import (
            get_language_registry,
            get_plugin_by_language,
        )
        from victor_coding.languages.base import LanguagePlugin

        # Ensure plugins are discovered
        registry = get_language_registry()
        if not registry.list_languages():
            registry.discover_plugins()

        plugin: Optional[LanguagePlugin] = get_plugin_by_language(language)
        if plugin is not None:
            # Wrap the language plugin as an edge handler
            return _VictorCodingPluginAdapter(plugin)
    except ImportError:
        logger.debug("victor_coding not available for edge detection")

    return None


class _VictorCodingPluginAdapter:
    """Adapter to use victor_coding LanguagePlugin as LanguageEdgeHandler.

    This bridges the gap between:
    - victor_coding LanguagePlugin protocol (detect_calls_edges synchronous)
    - victor-ai LanguageEdgeHandler protocol (detect_calls_edges async)
    """

    def __init__(self, plugin: Any):
        """Initialize adapter with a victor_coding language plugin.

        Args:
            plugin: LanguagePlugin instance from victor_coding
        """
        self._plugin = plugin
        self._supported_languages = [plugin.config.name] + plugin.config.aliases

    def get_supported_languages(self) -> List[str]:
        """Get list of language identifiers this handler supports."""
        return self._supported_languages

    async def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges using victor_coding plugin.

        Args:
            tree: Parsed tree-sitter tree
            source_code: Raw source code text
            file_path: Path to source file

        Returns:
            EdgeDetectionResult with detected calls
        """
        # victor_coding plugins use synchronous detect_calls_edges
        # Run in thread pool to avoid blocking
        import asyncio

        return await asyncio.to_thread(
            self._plugin.detect_calls_edges,
            tree,
            source_code,
            file_path,
        )

    def can_handle(self, language: str) -> bool:
        """Check if this handler can process the given language."""
        return language.lower() in [lang.lower() for lang in self._supported_languages]


__all__ = [
    "CallEdge",
    "EdgeDetectionResult",
    "LanguageEdgeHandler",
    "EdgeHandlerRegistry",
    "register_edge_handler",
    "get_edge_handler",
    "_VictorCodingPluginAdapter",  # Exported for testing
]
