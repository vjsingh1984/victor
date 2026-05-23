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


def _get_registry_plugin(registry: Any, language: str) -> Optional[Any]:
    """Return a victor_coding plugin from the current registry API."""
    try:
        return registry.get(language)
    except KeyError:
        return None


def _ensure_registry_discovered(registry: Any) -> Any:
    """Populate victor_coding's registry when it has not discovered plugins yet."""
    if not registry.list_languages():
        registry.discover_plugins()
    return registry


@dataclass
class CallEdge:
    """A detected function call edge.

    Attributes:
        caller_name: Name of the calling function/symbol
        callee_name: Name of the called function/symbol
        caller_line: Line number where call occurs (for debugging)
        receiver_type: Inferred static type of the receiver for method calls
            (e.g. ``obj.method()`` → ``Foo``). ``None`` when the language
            plugin or analysis provider could not determine it.
        is_method_call: True when the call is dot-dispatch
            (``obj.method()``) regardless of whether ``receiver_type`` is
            known. Indexing uses this to drop name-only fallback for
            method calls that would otherwise bind to unrelated methods
            with the same leaf name.
    """

    caller_name: str
    callee_name: str
    caller_line: Optional[int] = None
    receiver_type: Optional[str] = None
    is_method_call: bool = False


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

    Resolution order:
    1. Enhanced ``TreeSitterAnalysisProtocol`` provider via the capability
       registry — the preferred path; works for every language the
       provider's plugins declare.
    2. Direct victor_coding ``LanguagePlugin`` adapter fallback —
       activated only when no enhanced analysis provider is registered
       (e.g. an older host that hasn't completed plugin discovery yet).

    Args:
        language: Language identifier

    Returns:
        Handler instance or None if no provider handles this language.
    """
    handler = _get_analysis_provider_handler(language)
    if handler is not None:
        return handler
    return _get_victor_coding_handler(language)


def _get_analysis_provider_handler(language: str) -> Optional[LanguageEdgeHandler]:
    """Return an analysis-provider-backed handler when one is registered.

    Returns ``None`` when only the null stub is registered (no enhanced
    provider), or when the provider does not support ``language``. This
    is the preferred path: it does not import ``victor_coding`` directly
    and works for any language the provider's plugins declare.
    """
    try:
        from victor.core.capability_registry import CapabilityRegistry
        from victor.framework.vertical_protocols import TreeSitterAnalysisProtocol

        registry = CapabilityRegistry.get_instance()
        if not registry.is_enhanced(TreeSitterAnalysisProtocol):
            return None
        provider = registry.get(TreeSitterAnalysisProtocol)
    except Exception:
        return None

    if provider is None or not provider.supports_language(language):
        return None
    return _AnalysisProviderEdgeHandler(provider, language)


def _get_victor_coding_handler(language: str) -> Optional[LanguageEdgeHandler]:
    """Fallback: get edge handler from victor_coding ``LanguagePlugin``.

    ARCHITECTURAL VIOLATION (tracked):
    This function imports ``victor_coding`` directly, which the core →
    external boundary normally forbids. It is the *fallback* path for
    cases where the preferred ``TreeSitterAnalysisProtocol`` provider is
    not registered (only the null stub) but ``victor_coding`` is still
    importable — typically during early bootstrap, in tests that disable
    capability registration, or via an older host that hasn't migrated.
    Tracked in ``tests/unit/contracts/test_core_vertical_import_boundary.py``
    KNOWN_VIOLATIONS with a pointer back to this function.

    Args:
        language: Language identifier

    Returns:
        Handler instance or None if not found
    """
    try:
        from victor_coding.languages.registry import get_language_registry

        # Ensure plugins are discovered
        registry = _ensure_registry_discovered(get_language_registry())

        plugin = _get_registry_plugin(registry, language)
        if plugin is not None:
            # Wrap the language plugin as an edge handler
            return _VictorCodingPluginAdapter(plugin)
    except ImportError:
        logger.debug("victor_coding not available for edge detection")

    return None


class _AnalysisProviderEdgeHandler:
    """Adapter: ``TreeSitterAnalysisProtocol`` → ``LanguageEdgeHandler``.

    Wraps the analysis provider so that core indexing code can ask for
    CALLS edges through the existing ``handler.detect_calls_edges(tree,
    source_code, file_path)`` interface. The pre-parsed ``tree`` is
    intentionally ignored — the provider does its own parsing through
    :class:`TreeSitterService` (which shares the per-thread Parser cache),
    so the cost is one extra parse per file at most while the indexer is
    still building edges separately from the symbol pass. Once edge
    extraction joins symbol extraction in a single provider call, the
    duplicate parse goes away.
    """

    def __init__(self, provider: Any, language: str) -> None:
        self._provider = provider
        self._language = language

    def get_supported_languages(self) -> List[str]:
        return [self._language]

    def can_handle(self, language: str) -> bool:
        return language.lower() == self._language.lower()

    async def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        import asyncio

        edges = await asyncio.to_thread(
            self._provider.extract_edges,
            source_code.encode("utf-8"),
            self._language,
            file_path=str(file_path),
        )
        calls: List[CallEdge] = []
        for edge in edges:
            if edge.get("edge_type") != "CALLS":
                continue
            caller = edge.get("source")
            callee = edge.get("target")
            if not caller or not callee:
                continue
            calls.append(
                CallEdge(
                    caller_name=caller,
                    callee_name=callee,
                    caller_line=edge.get("line_number"),
                    receiver_type=edge.get("receiver_type"),
                    is_method_call=bool(edge.get("is_method_call", False)),
                )
            )
        return EdgeDetectionResult(calls=calls)


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
    "_AnalysisProviderEdgeHandler",  # Exported for testing
    "_VictorCodingPluginAdapter",  # Exported for testing (legacy fallback)
]
