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

"""Base codebase index implementation for core.

Provides the generic file-walking and indexing logic that can be
used for any directory-based agentic task (Coding, Research, RAG).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.framework.vertical_protocols import CodebaseIndexFactoryProtocol

logger = logging.getLogger(__name__)


class BaseCodebaseIndex:
    """Generic index for codebase-like directory structures.

    This class handles:
    - Recursive file walking
    - Filtering via ignore patterns
    - Change detection (staleness)
    - Coordination with chunkers and vector stores
    """

    def __init__(
        self,
        root_path: Path,
        vector_store: Optional[Any] = None,
        chunking_strategy: Optional[str] = "text",
        **kwargs: Any,
    ):
        self.root_path = root_path
        self.vector_store = vector_store
        self.chunking_strategy = chunking_strategy
        self.metadata: Dict[str, Any] = kwargs

    async def index(self, force: bool = False) -> Dict[str, Any]:
        """Perform full indexing of the root path.

        Subclasses or specific implementations should override this
        to add domain-specific parsing (e.g., AST extraction in coding).
        """
        logger.info(f"Indexing {self.root_path} with strategy={self.chunking_strategy}")
        # Base implementation just walks and chunks as text
        files_processed = 0
        for path in self.root_path.rglob("*"):
            if path.is_file() and not self._should_ignore(path):
                # chunk and add to store
                files_processed += 1

        return {"files_processed": files_processed}

    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored.

        Uses the capability registry to find the IgnorePatternsProtocol.
        """
        from victor.core.capability_registry import CapabilityRegistry
        from victor.framework.vertical_protocols import IgnorePatternsProtocol

        registry = CapabilityRegistry.get_instance()
        ignorer = registry.get(IgnorePatternsProtocol)
        if ignorer:
            return ignorer.should_ignore_path(str(path))

        # Fallback to simple dot-file ignore
        return any(part.startswith(".") for part in path.parts)


class CodebaseIndexFactory(CodebaseIndexFactoryProtocol):
    """Default factory for codebase indexers."""

    def create(self, root_path: str, **kwargs: Any) -> BaseCodebaseIndex:
        """Create a new BaseCodebaseIndex instance."""
        return BaseCodebaseIndex(Path(root_path), **kwargs)


class EnhancedCodebaseIndexFactory(CodebaseIndexFactoryProtocol):
    """Factory that delegates to an external codebase indexing provider.

    External vertical packages (e.g., victor-coding) register their
    CodebaseIndex factory as a capability via CapabilityRegistry at
    bootstrap time. This factory discovers and delegates to that provider.
    """

    def create(self, root_path: str, **kwargs: Any) -> Any:
        """Create a CodebaseIndex from an installed codebase indexing provider.

        Discovery order:
        1. CapabilityRegistry (if a different factory was registered)
        2. Direct import of victor_coding.codebase.indexer.CodebaseIndex
        3. ImportError with install instructions
        """
        # 1. Check registry for a different (non-self) factory
        from victor.core.capability_registry import CapabilityRegistry

        registry = CapabilityRegistry.get_instance()
        factory = registry.get(CodebaseIndexFactoryProtocol)
        if factory is not None and factory is not self:
            return factory.create(root_path, **kwargs)

        # 2. Direct import fallback — handles the case where the plugin
        #    system registers EnhancedCodebaseIndexFactory (self-referencing)
        #    but the actual CodebaseIndex class is importable.
        try:
            from victor_coding.codebase.indexer import CodebaseIndex

            return CodebaseIndex(Path(root_path), **kwargs)
        except ImportError:
            pass

        raise ImportError(
            "CodebaseIndex requires a codebase indexing provider "
            "(e.g., pip install victor-coding). The provider registers "
            "its factory via CapabilityRegistry at bootstrap."
        )


def detect_enhanced_index_factory() -> Optional[CodebaseIndexFactoryProtocol]:
    """Detect if a codebase indexing provider is registered.

    Returns None if no provider is available via CapabilityRegistry.
    """
    from victor.core.capability_registry import CapabilityRegistry

    registry = CapabilityRegistry.get_instance()
    factory = registry.get(CodebaseIndexFactoryProtocol)
    if factory is not None:
        return EnhancedCodebaseIndexFactory()
    return None
