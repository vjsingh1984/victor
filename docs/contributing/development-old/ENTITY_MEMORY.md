# Entity Memory System

> **Archived**: This document is kept for historical context and may be outdated. See `docs/contributing/index.md` for current guidance.


**Victor Framework**

Victor's entity memory system provides context-aware conversations by extracting, storing, and retrieving entities mentioned across conversations.

## Overview

The entity memory system is a **4-tier architecture**:

1. **Short-term memory** (in-memory): Current session entities
2. **Working memory** (LRU cache): Frequently accessed entities
3. **Long-term memory** (SQLite): Persistent entity storage
4. **Entity graph** (relationships): Tracked entity relationships

### Key Features

- **25+ Entity Types**: PERSON, ORGANIZATION, FILE, FUNCTION, CLASS, MODULE, etc.
- **15+ Relation Types**: IMPORTS, CONTAINS, DEPENDS_ON, RELATED_TO, etc.
- **Automatic Extraction**: Extract entities from user and assistant messages
- **Code-Aware**: Specialized handling for code entities (files, functions, classes)
- **Relationship Tracking**: Maintain relationships between entities
- **CLI Management**: Inspect and manage entities via command-line tools

## Quick Start

### CLI Usage

```bash
# List all entities in memory
victor memory list

# List entities by type
victor memory list --type FILE
victor memory list --type FUNCTION

# Show details for a specific entity
victor memory show UserService

# Search for entities
victor memory search authentication
victor memory search "UserService" --type CLASS

# Show entity graph statistics
victor memory graph

# List all available entity types
victor memory types
```

### Python API

#### Basic Entity Storage

```python
from victor.storage.memory.entity_memory import EntityMemory, EntityMemoryConfig
from victor.storage.memory.entity_types import EntityType

# Create entity memory instance
config = EntityMemoryConfig()
memory = EntityMemory(config=config)

# Store an entity
await memory.store(
    name="UserService",
    entity_type=EntityType.CLASS,
    attributes={"file": "src/auth/user_service.py", "language": "python"}
)

# Retrieve an entity
entity = memory.get("UserService", EntityType.CLASS)
print(f"Entity: {entity.name}, Type: {entity.entity_type}")
print(f"Mentions: {entity.mention_count}")
```

#### Entity Search

```python
# Search entities by name
entities = memory.search("user", limit=10)

# Get entities by type
from victor.storage.memory.entity_types import EntityType
files = memory.get_by_type(EntityType.FILE, limit=50)

# Get related entities
related = memory.get_related("UserService", depth=2)
```

#### Relationship Tracking

```python
from victor.storage.memory.entity_types import RelationType

# Store relationships
await memory.store_relation(
    source_entity_name="UserService",
    target_entity_name="AuthenticationService",
    relation_type=RelationType.DEPENDS_ON
)

# Get all relations for an entity
relations = memory.get_relations("UserService")
```

### Entity Extraction from Messages

```python
from victor.agent.entity_extraction_service import EntityExtractor, EntityExtractionConfig

# Create extractor
config = EntityExtractionConfig(
    enable_code_aware_extraction=True,
    extract_file_references=True,
    extract_function_references=True,
)
extractor = EntityExtractor(entity_memory=memory, config=config)

# Extract entities from a message
from victor.providers.base import Message

message = Message(
    role="user",
    content="Please fix the authentication bug in UserService.py"
)
await extractor.extract_from_message(message)

# Get relevant entities for context
relevant = await extractor.query_relevant_entities("authentication")
print(f"Relevant entities: {relevant}")

# Get context summary
summary = await extractor.get_context_summary()
print(summary)
```

## Entity Types

### People & Organizations

| Type | Description | Example |
|------|-------------|---------|
| `PERSON` | Individual person | "John Doe" |
| `ORGANIZATION` | Company or organization | "Acme Corp" |
| `TEAM` | Development team | "Platform Team" |

### Code Entities

| Type | Description | Example |
|------|-------------|---------|
| `FILE` | Source code file | "UserService.py" |
| `FUNCTION` | Function or method | "authenticate_user" |
| `CLASS` | Class definition | "UserService" |
| `MODULE` | Python module or package | "src.auth" |
| `VARIABLE` | Variable name | "user_token" |
| `INTERFACE` | Interface or protocol | "IAuthenticator" |

### Project Entities

| Type | Description | Example |
|------|-------------|---------|
| `PROJECT` | Project name | "Victor Framework" |
| `REPOSITORY` | Git repository | "victor-ai/victor" |
| `PACKAGE` | NPM/Python package | "victor-ai" |
| `DEPENDENCY` | External dependency | "anthropic" |

### Concepts

| Type | Description | Example |
|------|-------------|---------|
| `CONCEPT` | Abstract concept | "Authentication flow" |
| `TECHNOLOGY` | Technology stack | "React" |
| `PATTERN` | Design pattern | "Factory Pattern" |
| `REQUIREMENT` | Business requirement | "User must login" |
| `BUG` | Bug or issue | "Token expiration bug" |
| `FEATURE` | Feature request | "Multi-factor auth" |

### Infrastructure

| Type | Description | Example |
|------|-------------|---------|
| `SERVICE` | Microservice | "Auth Service" |
| `ENDPOINT` | API endpoint | "/api/auth/login" |
| `DATABASE` | Database | "PostgreSQL" |
| `CONFIG` | Configuration | "config.yaml" |

## Relation Types

### Structural Relations

| Relation | Description | Example |
|----------|-------------|---------|
| `CONTAINS` | Parent-child relationship | UserService contains authenticate |
| `BELONGS_TO` | Membership | authenticate belongs to UserService |
| `IMPORTS` | Import statement | UserService imports Database |
| `IMPLEMENTS` | Interface implementation | UserService implements IAuth |
| `EXTENDS` | Inheritance | AdminUser extends User |
| `DEPENDS_ON` | Dependency relationship | UserService depends on Database |

### Semantic Relations

| Relation | Description | Example |
|----------|-------------|---------|
| `RELATED_TO` | Generic relationship | UserService related to Authentication |
| `SIMILAR_TO` | Similar entities | UserService similar to AdminService |
| `REFERENCES` | Code reference | UserService references UserConfig |
| `USED_BY` | Inverse dependency | Database used by UserService |

### Ownership Relations

| Relation | Description | Example |
|----------|-------------|---------|
| `CREATED_BY` | Creator relationship | UserService created by John |
| `OWNED_BY` | Ownership | UserService owned by Platform Team |
| `MAINTAINED_BY` | Maintenance responsibility | UserService maintained by Auth Team |

### Temporal Relations

| Relation | Description | Example |
|----------|-------------|---------|
| `PRECEDED_BY` | Predecessor | UserService preceded by LegacyAuthService |
| `FOLLOWED_BY` | Successor | LegacyAuthService followed by UserService |
| `REPLACED_BY` | Replacement | LegacyAuthService replaced by UserService |

## Configuration

### Entity Memory Configuration

```python
from victor.storage.memory.entity_memory import EntityMemoryConfig

config = EntityMemoryConfig(
    enable_embeddings=True,  # Enable semantic search
    enable_graph=True,  # Enable relationship tracking
    working_memory_size=100,  # LRU cache size
    auto_merge=True,  # Merge duplicate entities
    persist_to_disk=True,  # Save to SQLite
)

memory = EntityMemory(config=config)
```

### Entity Extraction Configuration

```python
from victor.agent.entity_extraction_service import EntityExtractionConfig

config = EntityExtractionConfig(
    enable_extraction=True,
    enable_code_aware_extraction=True,
    enable_relation_extraction=True,
    min_entity_length=2,
    max_entities_per_message=50,
    extract_file_references=True,
    extract_function_references=True,
    extract_class_references=True,
)

extractor = EntityExtractor(entity_memory=memory, config=config)
```

## Entity Graph

The entity graph tracks relationships between entities:

```python
from victor.storage.memory.entity_graph import EntityGraph

# Create graph instance
graph = EntityGraph()

# Get statistics
stats = graph.get_statistics()
print(f"Total entities: {stats['total_entities']}")
print(f"Total relations: {stats['total_relations']}")
print(f"Most connected: {stats['most_connected']}")

# Find path between entities
path = graph.find_shortest_path("UserService", "Database")
print(f"Path: {' -> '.join(path)}")

# Get neighbors
neighbors = graph.get_neighbors("UserService", depth=2)
for neighbor in neighbors:
    print(f"  - {neighbor.name} ({neighbor.entity_type.value})")
```

## Integration with Orchestration Layer

To integrate entity extraction into the conversation flow:

```python
from victor.agent.entity_extraction_service import EntityExtractor
from victor.agent.orchestrator import AgentOrchestrator

# In your orchestrator initialization:
class EnhancedOrchestrator(AgentOrchestrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._entity_extractor = None

    async def initialize_entity_extractor(self):
        """Initialize entity extraction service."""
        from victor.storage.memory.entity_memory import EntityMemory

        memory = EntityMemory()
        self._entity_extractor = EntityExtractor(entity_memory=memory)

    async def process_message(self, message: Message):
        """Process message with entity extraction."""
        # Extract entities from message
        if self._entity_extractor:
            await self._entity_extractor.extract_from_message(message)

        # Continue with normal processing
        return await super().process_message(message)

    async def get_entity_context(self) -> str:
        """Get entity context for prompt enhancement."""
        if self._entity_extractor:
            return await self._entity_extractor.get_context_summary()
        return ""
```

## Best Practices

### When to Use Entity Memory

**Use entity memory when:**
- Conversations reference multiple files, functions, or classes
- You need to track relationships between code entities
- You want context-aware follow-up questions
- Debugging complex codebases with many components

**Avoid entity memory when:**
- Single-file tasks with no cross-references
- One-off conversations with no reuse
- Performance-critical paths (adds ~50-100ms per message)

### Entity Extraction Tips

1. **Code-Heavy Conversations**: Enable `extract_file_references`, `extract_function_references`, and `extract_class_references`
2. **Architecture Discussions**: Enable `enable_relation_extraction` to track dependencies
3. **Long Conversations**: Use `get_context_summary()` to provide entity context to the LLM
4. **Performance**: Adjust `max_entities_per_message` to balance accuracy vs. speed

### Entity Memory Management

```bash
# Monitor entity memory usage
victor memory list --limit 1000

# Clear specific entity types
victor memory clear --type CONCEPT --confirm

# View entity relationships
victor memory graph --entity UserService --depth 3
```

## Architecture

### 4-Tier Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Entity Memory System                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Tier 1: Short-term Memory (in-memory dict)                  │
│  ├─ Fast access for current session                          │
│  └─ Cleared on session end                                   │
│                                                               │
│  Tier 2: Working Memory (LRU cache)                          │
│  ├─ Frequently accessed entities                             │
│  ├─ Configurable size (default: 100 entities)                │
│  └─ Automatic eviction of least-recently-used                │
│                                                               │
│  Tier 3: Long-term Memory (SQLite)                           │
│  ├─ Persistent storage across sessions                       │
│  ├─ Indexed by entity ID and name                            │
│  └─ Supports complex queries                                 │
│                                                               │
│  Tier 4: Entity Graph (relationships)                       │
│  ├─ Adjacency list representation                            │
│  ├─ BFS-based path finding                                   │
│  └─ Graph statistics and analysis                            │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Message
    │
    ▼
┌──────────────────┐
│ Entity Extractor │
│ - Parse content  │
│ - Identify types │
│ - Extract attrs  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Entity Memory    │
│ - Deduplicate    │
│ - Update counts  │
│ - Store in tier  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Entity Graph     │
│ - Track relations│
│ - Update paths   │
└──────────────────┘
```

## Performance Considerations

- **Memory Overhead**: ~1-2KB per entity in memory
- **Extraction Time**: ~50-100ms per message (code-heavy)
- **SQLite Queries**: ~10-20ms for indexed queries
- **Graph Operations**: ~5-10ms for neighbor lookup (depth=2)

## Troubleshooting

### Entity Extraction Not Working

```python
# Check if extraction is enabled
config.enable_extraction  # Should be True

# Check entity memory is initialized
extractor.get_entity_memory()  # Should return EntityMemory instance
```

### Missing Entities

```bash
# Check if entities exist
victor memory search "entity_name"

# List all entities
victor memory list --limit 100

# Check extraction logs
# (Enable DEBUG logging for entity extraction details)
```

### Performance Issues

```python
# Reduce max entities per message
config = EntityExtractionConfig(max_entities_per_message=20)

# Disable relation extraction
config.enable_relation_extraction = False

# Disable code-aware extraction
config.enable_code_aware_extraction = False
```

## See Also

- [Agent and Workflow Presets](AGENT_AND_WORKFLOW_PRESETS.md)
- [Conversation Management](../architecture/CONVERSATION.md)
- [CLI Reference](../cli/README.md)
- [Storage Architecture](../architecture/STORAGE.md)
