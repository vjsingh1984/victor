# ADR 0003: Tool Metadata System

Date: 2025-12-16

## Status

Accepted

## Context

As Victor grew from 10 tools to 45+ enterprise tools, we faced significant challenges in tool selection, categorization, and discovery:

### Problems with Static Configuration

1. **Static Tool Lists**: Hardcoded sets like `WRITE_TOOL_NAMES = frozenset(["write", "edit", ...])` scattered across 12+ files
2. **Configuration Duplication**: Same tool classifications repeated in:
   - `action_authorizer.py` (write tools)
   - `safety.py` (dangerous tools)
   - `tool_executor.py` (cacheable tools)
   - `conversation_state.py` (stage-based tools)
   - `tool_selection.py` (category keywords)
3. **Maintenance Burden**: Adding a new tool required updating 5-10 different files
4. **Inconsistency Risk**: Easy to add tool to one list but forget others
5. **No Single Source of Truth**: No central registry of tool properties
6. **Discovery Difficulty**: No way to query "all tools in category X" or "tools matching keyword Y"

### Alternative Approaches Considered

- **Centralized YAML Config**: Single tool_config.yaml with all metadata
  - **Con**: Separates metadata from tool implementation
  - **Con**: Risk of drift between code and config
  - **Con**: No type checking

- **Database Backend**: Store tool metadata in DB
  - **Con**: Overkill for static metadata
  - **Con**: Runtime dependency on DB
  - **Con**: Deployment complexity

- **Annotation/Decorator Only**: Store all metadata in @tool decorator args
  - **Con**: No efficient querying
  - **Con**: Must instantiate tools to read metadata

## Decision

We implement a **Tool Metadata System** with three key components:

### 1. Metadata Enums (victor/tools/enums.py)

Five enums define tool properties:

```python
class Priority(Enum):
    """Tool selection priority"""
    CRITICAL = 0  # Always include (read, write, ls)
    HIGH = 1      # Include for most tasks
    MEDIUM = 2    # Include for relevant tasks
    LOW = 3       # Include only if specifically needed

class AccessMode(Enum):
    """What the tool can do"""
    READONLY = "readonly"   # Safe to cache
    WRITE = "write"         # Modifies state
    EXECUTE = "execute"     # Runs code
    MIXED = "mixed"         # Multiple modes

class DangerLevel(Enum):
    """Risk level for safety checks"""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"          # Requires confirmation
    CRITICAL = "critical"  # Strong warnings

class ExecutionCategory(Enum):
    """For parallel execution planning"""
    READ_ONLY = "read_only"    # Can parallelize
    WRITE = "write"            # Conflicts with other writes
    EXECUTE = "execute"        # Conflicts with writes
    COMPUTE = "compute"        # CPU-bound, parallelizable
    NETWORK = "network"        # I/O-bound, parallelizable
    MIXED = "mixed"            # Sequential only

class CostTier(Enum):
    """Cost tier for budget enforcement"""
    FREE = "free"              # Local operations
    LOW = "low"                # Compute-only
    MEDIUM = "medium"          # External API calls
    HIGH = "high"              # Resource-intensive
```

### 2. Tool Decorator with Metadata (victor/tools/base.py)

The `@tool` decorator captures metadata at definition time:

```python
@tool(
    name="read_file",
    description="Read file contents",
    # Core metadata
    priority=Priority.CRITICAL,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    cost_tier=CostTier.FREE,
    execution_category=ExecutionCategory.READ_ONLY,

    # Semantic selection
    category="filesystem",
    keywords=["read", "file", "content", "show"],

    # Stage-based selection
    stages=["initial", "reading", "analysis"],

    # Advanced selection
    mandatory_keywords=["show file", "read content"],
    task_types=["analysis", "search"],
    progress_params=["path", "offset", "limit"],
)
async def read_file(path: str, offset: int = 0, limit: int = 1000):
    ...
```

### 3. Metadata Registry (victor/tools/metadata_registry.py)

The `ToolMetadataRegistry` provides efficient querying:

**Indexed Lookups**:
- By priority: `registry.get_by_priority(Priority.CRITICAL)`
- By access mode: `registry.get_by_access_mode(AccessMode.READONLY)`
- By danger level: `registry.get_by_danger_level(DangerLevel.HIGH)`
- By category: `registry.get_by_category("git")`
- By stage: `registry.get_by_stage("reading")`
- By task type: `registry.get_by_task_type("analysis")`
- By execution category: `registry.get_by_execution_category(ExecutionCategory.READ_ONLY)`

**Text Matching**:
```python
# Keyword-based matching with scores
matches = registry.get_tools_matching_text_scored(
    "run tests",
    min_score=0.3,
    max_results=10
)
# Returns: [KeywordMatchResult(tool_name="test", score=0.85, ...)]

# Mandatory keyword matching (guaranteed inclusion)
tools = registry.get_tools_matching_mandatory_keywords("show diff")
# Returns: {"git_diff", "diff"}
```

**Dynamic Discovery**:
```python
# Auto-discover categories from tools
categories = registry.get_all_categories()
# Returns: {"git", "filesystem", "testing", "docker", ...}

# Get all keywords for a category
keywords = registry.get_category_keywords("git")
# Returns: {"commit", "push", "pull", "branch", ...}

# Detect categories from text
detected = registry.detect_categories_from_text("commit changes")
# Returns: {"git"}
```

**Derived Queries** (replaces static sets):
```python
# Replaces WRITE_TOOL_NAMES frozenset
write_tools = registry.get_write_tools()

# Replaces DEFAULT_CACHEABLE_TOOLS
cacheable = registry.get_idempotent_tools()

# Replaces STAGE_TOOL_MAPPING dict
reading_tools = registry.get_tools_by_stage("reading")

# Replaces TASK_TYPE_CATEGORIES dict
analysis_tools = registry.get_tools_by_task_type("analysis")
```

### Migration Pattern

Replace static configurations with registry queries:

```python
# BEFORE (static)
WRITE_TOOL_NAMES = frozenset(["write", "edit", "git_commit", ...])
if tool_name in WRITE_TOOL_NAMES:
    require_approval()

# AFTER (registry-driven)
from victor.tools.metadata_registry import get_write_tools
write_tools = get_write_tools()
if tool_name in write_tools:
    require_approval()
```

### Key Design Principles

1. **Single Source of Truth**: Metadata lives with tool implementation
2. **Decorator-Driven**: Metadata declared at tool definition
3. **Efficient Querying**: Pre-built indexes for O(1) lookups
4. **Backward Compatible**: Fallback to defaults if metadata missing
5. **Dynamic Discovery**: Categories and keywords auto-discovered
6. **Type-Safe**: Enums prevent typos and invalid values

## Consequences

### Positive Consequences

- **Centralization**: One place to update tool properties
- **Consistency**: Impossible to have conflicting classifications
- **Discoverability**: Query tools by any metadata attribute
- **Maintainability**: Adding tools requires updating only decorator
- **Performance**: Indexed lookups faster than list scans
- **Type Safety**: Enums catch mistakes at development time
- **Dynamic Adaptation**: System learns new categories from tool additions
- **Metrics**: Track which keywords/categories are most used

### Negative Consequences

- **Decorator Complexity**: More parameters to understand
- **Migration Effort**: 45+ tools need decorator updates
- **Memory Overhead**: Registry maintains indexes (~50KB for 45 tools)
- **Initialization Cost**: Registry population takes ~10ms

### Risks and Mitigations

- **Risk**: Developers might forget to add metadata to new tools
  - **Mitigation**: Auto-generation from tool properties as fallback
  - **Mitigation**: Linting rule to check for missing critical metadata

- **Risk**: Metadata could drift out of sync with tool behavior
  - **Mitigation**: Tests verify metadata accuracy
  - **Mitigation**: CI checks for tools missing metadata

- **Risk**: Registry state could become stale if tools added after init
  - **Mitigation**: Global singleton pattern ensures consistency
  - **Mitigation**: Runtime tool registration supported

## Implementation Notes

### Metadata Registration Flow

1. **Tool Definition**: Developer adds @tool decorator with metadata
2. **Tool Import**: When module loads, decorator captures metadata
3. **Registration**: Tool registered with ToolRegistry and ToolMetadataRegistry
4. **Indexing**: Metadata indexed by priority, category, keywords, etc.
5. **Querying**: Consumers query registry instead of static lists

### Registry Architecture

```python
class ToolMetadataRegistry:
    def __init__(self):
        # Primary storage
        self._entries: Dict[str, ToolMetadataEntry] = {}

        # Indexes for fast lookup
        self._by_priority: Dict[Priority, Set[str]] = {}
        self._by_access_mode: Dict[AccessMode, Set[str]] = {}
        self._by_danger_level: Dict[DangerLevel, Set[str]] = {}
        self._by_category: Dict[str, Set[str]] = {}
        self._by_keyword: Dict[str, Set[str]] = {}
        self._by_stage: Dict[str, Set[str]] = {}
        self._by_task_type: Dict[str, Set[str]] = {}
        self._by_execution_category: Dict[ExecutionCategory, Set[str]] = {}

        # Metrics for observability
        self._metrics: MatchingMetrics = MatchingMetrics()
```

### Keyword Matching Scoring

```python
def calculate_score(tool: ToolMetadataEntry, keywords_matched: Set[str]) -> float:
    # Base score: ratio of matched to total keywords
    base = len(keywords_matched) / len(tool.keywords)

    # Specificity bonus: longer keywords more specific
    specificity = sum(len(k) for k in keywords_matched) / (len(keywords_matched) * 10)
    specificity = min(specificity, 0.2)  # Cap at 0.2

    # Priority boost
    priority_boost = PRIORITY_WEIGHTS[tool.priority]

    return base + specificity + priority_boost
```

### Key Files

**Core System**:
- `victor/tools/enums.py` - Metadata enums (Priority, AccessMode, etc.)
- `victor/tools/metadata_registry.py` - ToolMetadataRegistry (1,812 lines)
- `victor/tools/base.py` - BaseTool, @tool decorator, ToolMetadata

**Consumers** (12+ files migrated):
- `victor/agent/action_authorizer.py` - Uses get_write_tools()
- `victor/agent/safety.py` - Uses get_dangerous_tools()
- `victor/agent/tool_executor.py` - Uses get_idempotent_tools()
- `victor/agent/conversation_state.py` - Uses get_tools_by_stage()
- `victor/agent/tool_selection.py` - Uses keyword matching
- `victor/agent/parallel_executor.py` - Uses execution categories

### Performance Characteristics

- **Registration**: O(n) for n tools, ~10ms for 45 tools
- **Query by property**: O(1) via index lookup
- **Keyword matching**: O(k) where k = keywords, ~1-2ms per query
- **Memory usage**: ~1KB per tool, ~50KB total for 45 tools

## References

- Related: [ADR-0002: Vertical System Design](0002-vertical-system-design.md) (verticals use tiered tool configs)
- Implementation: victor/tools/metadata_registry.py
- Enums: victor/tools/enums.py
- CLAUDE.md: Search for "Tool System" section
- Migration guide in metadata_registry.py docstring
