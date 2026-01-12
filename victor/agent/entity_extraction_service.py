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

"""Entity extraction service for conversation-aware context.

This service extracts entities from messages and maintains them in memory
for improved context awareness across conversations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.providers.base import Message
    from victor.storage.memory.entity_memory import EntityMemory
    from victor.storage.memory.entity_types import EntityType

logger = logging.getLogger(__name__)


@dataclass
class EntityExtractionConfig:
    """Configuration for entity extraction."""

    enable_extraction: bool = True
    enable_code_aware_extraction: bool = True
    enable_relation_extraction: bool = True
    min_entity_length: int = 2
    max_entities_per_message: int = 50
    confidence_threshold: float = 0.5

    # Code-specific patterns
    extract_file_references: bool = True
    extract_function_references: bool = True
    extract_class_references: bool = True
    extract_module_references: bool = True


class EntityExtractor:
    """Extracts entities from messages for context-aware conversations.

    This service integrates with the conversation flow to automatically
    extract and track entities mentioned in messages.

    Example:
        extractor = EntityExtractor(entity_memory=memory)
        await extractor.extract_from_message(user_message)
        await extractor.extract_from_message(assistant_message)
    """

    def __init__(
        self,
        entity_memory: Optional["EntityMemory"] = None,
        config: Optional[EntityExtractionConfig] = None,
    ):
        """Initialize the entity extractor.

        Args:
            entity_memory: Entity memory instance for storing extracted entities
            config: Extraction configuration
        """
        from victor.storage.memory.entity_memory import EntityMemory, EntityMemoryConfig

        self._memory = entity_memory or EntityMemory(config=EntityMemoryConfig())
        self._config = config or EntityExtractionConfig()

        # Compile regex patterns for code entity detection
        self._file_pattern = re.compile(
            r'\b(?:[\w-]+/)*[\w-]+\.(?:py|js|ts|java|cpp|c|h|go|rs|rb|php|swift|kt|scala|rs|yaml|yml|json|xml|html|css|md|txt|sh|bash|zsh)\b'
        )
        self._function_pattern = re.compile(r'\b[a-z_][a-z0-9_]*(?:\s*\(|\s*\()')
        self._class_pattern = re.compile(r'\b[A-Z][a-zA-Z0-9]*\b')
        self._module_pattern = re.compile(r'\b(?:from|import)\s+([a-z_][a-z0-9_.]*)')

    async def extract_from_message(
        self,
        message: "Message",
        conversation_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract entities from a message and store them in memory.

        Args:
            message: The message to extract entities from
            conversation_context: Optional context about the conversation
        """
        if not self._config.enable_extraction:
            return

        if not message.content:
            return

        # Extract entities based on message role
        if message.role == "user":
            await self._extract_from_user_message(message, conversation_context)
        elif message.role == "assistant":
            await self._extract_from_assistant_message(message, conversation_context)
        elif message.role == "system":
            # System messages usually don't contain user-mentioned entities
            pass

    async def _extract_from_user_message(
        self,
        message: "Message",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract entities from user messages.

        User messages often explicitly mention entities they care about.
        """
        content = message.content

        # Extract code entities if enabled
        if self._config.enable_code_aware_extraction:
            await self._extract_code_entities(content, context)

        # Extract natural language entities
        await self._extract_natural_language_entities(content, context)

    async def _extract_from_assistant_message(
        self,
        message: "Message",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract entities from assistant messages.

        Assistant messages may reference files, functions, and code entities
        they're working with.
        """
        content = message.content

        # Focus on code entities from assistant
        if self._config.enable_code_aware_extraction:
            await self._extract_code_entities(content, context)

    async def _extract_code_entities(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract code-related entities from content.

        Args:
            content: Text content to extract from
            context: Optional conversation context
        """
        from victor.storage.memory.entity_types import EntityType

        entities_found = 0

        # Extract file references
        if self._config.extract_file_references:
            files = self._file_pattern.findall(content)
            for file_path in set(files):
                if entities_found >= self._config.max_entities_per_message:
                    break

                await self._memory.store(
                    name=file_path,
                    entity_type=EntityType.FILE,
                    attributes={"source": "code_reference", "context": "mentioned_in_message"},
                )
                entities_found += 1

        # Extract function references
        if self._config.extract_function_references:
            # Find function calls and definitions
            functions = self._function_pattern.findall(content)
            for func_name in set(functions):
                if entities_found >= self._config.max_entities_per_message:
                    break

                # Clean up the function name
                func_name = func_name.replace("(", "").strip()
                if len(func_name) >= self._config.min_entity_length:
                    await self._memory.store(
                        name=func_name,
                        entity_type=EntityType.FUNCTION,
                        attributes={"source": "code_reference"},
                    )
                    entities_found += 1

        # Extract class references
        if self._config.extract_class_references:
            # Find class names (PascalCase)
            classes = self._class_pattern.findall(content)
            for class_name in set(classes):
                if entities_found >= self._config.max_entities_per_message:
                    break

                # Filter out common non-class words
                if class_name.lower() not in {"the", "this", "that", "when", "then", "with"}:
                    await self._memory.store(
                        name=class_name,
                        entity_type=EntityType.CLASS,
                        attributes={"source": "code_reference"},
                    )
                    entities_found += 1

        # Extract module references
        if self._config.extract_module_references:
            modules = self._module_pattern.findall(content)
            for module_name in set(modules):
                if entities_found >= self._config.max_entities_per_message:
                    break

                await self._memory.store(
                    name=module_name,
                    entity_type=EntityType.MODULE,
                    attributes={"source": "import_statement"},
                )
                entities_found += 1

    async def _extract_natural_language_entities(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract natural language entities from content.

        This is a simplified implementation. A production system would use
        NLP models or LLM-based extraction for better accuracy.

        Args:
            content: Text content to extract from
            context: Optional conversation context
        """
        from victor.storage.memory.entity_types import EntityType

        entities_found = 0

        # Simple pattern-based extraction (can be enhanced with LLM)
        # Look for quoted strings (potential proper nouns)
        quoted_pattern = re.compile(r'"([^"]+)"')
        quotes = quoted_pattern.findall(content)

        for quote in quotes:
            if entities_found >= self._config.max_entities_per_message:
                break

            # Skip short quotes
            if len(quote) < self._config.min_entity_length:
                continue

            # Store as CONCEPT entity (could be refined with LLM)
            await self._memory.store(
                name=quote,
                entity_type=EntityType.CONCEPT,
                attributes={"source": "quoted_text"},
            )
            entities_found += 1

    async def extract_relations(
        self,
        source_entity_name: str,
        target_entity_name: str,
        relation_type: str,
    ) -> None:
        """Extract and store a relationship between two entities.

        Args:
            source_entity_name: Name of the source entity
            target_entity_name: Name of the target entity
            relation_type: Type of relationship (e.g., "IMPORTS", "CONTAINS")
        """
        if not self._config.enable_relation_extraction:
            return

        from victor.storage.memory.entity_types import RelationType

        try:
            relation_enum = RelationType[relation_type.upper()]
            await self._memory.store_relation(
                source_entity_name=source_entity_name,
                target_entity_name=target_entity_name,
                relation_type=relation_enum,
            )
        except KeyError:
            logger.warning(f"Unknown relation type: {relation_type}")

    def get_entity_memory(self) -> "EntityMemory":
        """Get the underlying entity memory instance.

        Returns:
            EntityMemory instance
        """
        return self._memory

    async def query_relevant_entities(
        self,
        query: str,
        limit: int = 10,
    ) -> List[str]:
        """Query entities relevant to a query.

        Args:
            query: Query string
            limit: Maximum number of entities to return

        Returns:
            List of relevant entity descriptions
        """
        entities = self._memory.search(query, limit=limit)

        return [
            f"{e.name} ({e.entity_type.value})" for e in entities
        ]

    async def get_context_summary(self, limit: int = 20) -> str:
        """Generate a summary of key entities in context.

        Args:
            limit: Maximum number of entities to include

        Returns:
            Summary string of key entities
        """
        # Get most mentioned entities
        all_entities = self._memory.search("", limit=limit)

        if not all_entities:
            return "No entities tracked in this conversation."

        # Group by type
        by_type: Dict[str, List[str]] = {}
        for entity in all_entities:
            entity_type = entity.entity_type.value
            if entity_type not in by_type:
                by_type[entity_type] = []
            by_type[entity_type].append(entity.name)

        # Generate summary
        summary_parts = []
        for entity_type, names in sorted(by_type.items()):
            names_str = ", ".join(sorted(names)[:5])
            if len(names) > 5:
                names_str += f" and {len(names) - 5} more"
            summary_parts.append(f"{entity_type}: {names_str}")

        return "Key entities in context:\n" + "\n".join(f"- {part}" for part in summary_parts)


__all__ = [
    "EntityExtractor",
    "EntityExtractionConfig",
]
