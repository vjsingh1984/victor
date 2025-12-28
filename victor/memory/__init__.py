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

"""Entity memory system for Victor.

Provides 4-tier entity memory architecture:
- Short-term: Current session entities
- Working memory: LRU cache of active entities
- Long-term: Persistent SQLite + vector storage
- Entity graph: Relationship tracking and traversal

Example:
    from victor.memory import EntityMemory, EntityType, CompositeExtractor

    # Create memory with extraction
    memory = EntityMemory()
    extractor = CompositeExtractor.create_default()

    # Extract and store entities from text
    result = await extractor.extract(text)
    for entity in result.entities:
        await memory.store(entity)

    # Query entities
    entities = await memory.search("auth", entity_types=[EntityType.CLASS])

    # Get related entities via graph
    related = await memory.get_related("ent_abc123")
"""

from victor.memory.entity_types import (
    EntityType,
    Entity,
    EntityRelation,
    RelationType,
    CODE_ENTITY_TYPES,
    PROJECT_ENTITY_TYPES,
    CONCEPT_ENTITY_TYPES,
    PEOPLE_ENTITY_TYPES,
)
from victor.memory.entity_memory import (
    EntityMemory,
    EntityMemoryConfig,
    LRUCache,
)
from victor.memory.entity_graph import (
    EntityGraph,
    GraphPath,
    GraphStats,
)
from victor.memory.extractors import (
    EntityExtractor,
    ExtractionResult,
    CodeEntityExtractor,
    TextEntityExtractor,
    CompositeExtractor,
    TreeSitterEntityExtractor,
    TreeSitterFileExtractor,
    has_tree_sitter,
    create_extractor,
)
from victor.memory.integration import (
    EntityMemoryIntegration,
    create_entity_integration,
)

__all__ = [
    # Entity types
    "EntityType",
    "Entity",
    "EntityRelation",
    "RelationType",
    # Type categories
    "CODE_ENTITY_TYPES",
    "PROJECT_ENTITY_TYPES",
    "CONCEPT_ENTITY_TYPES",
    "PEOPLE_ENTITY_TYPES",
    # Memory
    "EntityMemory",
    "EntityMemoryConfig",
    "LRUCache",
    # Graph
    "EntityGraph",
    "GraphPath",
    "GraphStats",
    # Extractors
    "EntityExtractor",
    "ExtractionResult",
    "CodeEntityExtractor",
    "TextEntityExtractor",
    "CompositeExtractor",
    "TreeSitterEntityExtractor",
    "TreeSitterFileExtractor",
    "has_tree_sitter",
    "create_extractor",
    # Integration
    "EntityMemoryIntegration",
    "create_entity_integration",
]
