# Tools API Reference

Tool registry and configuration framework for Victor agents. Provides preset tool sets, categories, and extensible tool registration.

## Overview

The Tools API provides:
- **Tool presets** for common configurations (`default`, `minimal`, `full`, `airgapped`)
- **Tool categories** for organized selection (filesystem, git, web, testing, etc.)
- **Dynamic registry** for plugin and vertical extensions
- **Type-safe configuration** with Python type hints

## Quick Example

```python
from victor.framework.tools import ToolSet

# Use presets
tools = ToolSet.default()     # Core + filesystem + git
tools = ToolSet.minimal()     # Just core tools
tools = ToolSet.full()        # Everything

# Custom selection
tools = ToolSet.from_categories(["filesystem", "git", "testing"])
tools = ToolSet.from_tools(["read", "write", "git"])

# Modify existing sets
tools = ToolSet.default().include("docker").exclude_tools("shell")

# Use with Agent
from victor.framework import Agent

agent = Agent(tools=tools)
```

## ToolCategory Enum

Built-in tool categories for organized tool selection.

```python
class ToolCategory(str, Enum):
    """Built-in tool categories."""

    CORE = "core"                # Essential tools: read, write, edit, shell, search
    FILESYSTEM = "filesystem"    # File operations: list, glob, find, mkdir, rm, mv, cp
    GIT = "git"                  # Version control: status, diff, commit, branch, log
    SEARCH = "search"            # Search tools: grep, glob, code_search, semantic_code_search
    WEB = "web"                  # Network tools: web_search, web_fetch, http_request
    DATABASE = "database"        # Database tools: query, schema inspection
    DOCKER = "docker"            # Container operations: run, build, compose
    TESTING = "testing"          # Test execution: pytest, unittest, coverage
    REFACTORING = "refactoring"  # Code refactoring: rename, extract, inline
    DOCUMENTATION = "documentation"  # Documentation: docstring generation, README updates
    ANALYSIS = "analysis"        # Code analysis: complexity, metrics, static analysis
    COMMUNICATION = "communication"  # Communication tools: slack, teams, jira
    CUSTOM = "custom"            # Custom/user-defined tools
```

## ToolSet Class

Configuration for which tools are available to an agent.

```python
@dataclass
class ToolSet:
    """Configuration for which tools are available to an agent.

    Attributes:
        tools: Specific tool names to include
        categories: Tool categories to include
        exclude: Tool names to exclude
    """
```

### Constructor

```python
ToolSet(
    tools: set[str] = field(default_factory=set),
    categories: set[str] = field(default_factory=set),
    exclude: set[str] = field(default_factory=set),
) -> ToolSet
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `tools` | `set[str]` | `set()` | Specific tool names to include |
| `categories` | `set[str]` | `set()` | Tool categories to include |
| `exclude` | `set[str]` | `set()` | Tool names to explicitly exclude |

### Class Methods

#### default()

```python
@classmethod
def default(cls) -> ToolSet:
    """Default tool set - core + filesystem + git.

    This is the recommended starting point for most use cases.
    Includes file operations, shell access, and git integration.

    Returns:
        ToolSet with sensible defaults
    """
```

**Example**:

```python
tools = ToolSet.default()
# Includes: CORE, FILESYSTEM, GIT categories
```

#### minimal()

```python
@classmethod
def minimal(cls) -> ToolSet:
    """Minimal tool set - only core operations.

    Use for simple tasks that don't need git or advanced file operations.

    Returns:
        ToolSet with minimal tools

    Includes:
        - read, write, edit (file operations)
        - shell (command execution)
        - search (text search)
    """
```

**Example**:

```python
tools = ToolSet.minimal()
# Just CORE category
```

#### full()

```python
@classmethod
def full(cls) -> ToolSet:
    """Full tool set - all available tools.

    Use with caution - gives the agent access to everything.

    Returns:
        ToolSet with all categories
    """
```

**Example**:

```python
tools = ToolSet.full()
# All categories enabled
```

#### airgapped()

```python
@classmethod
def airgapped(cls) -> ToolSet:
    """Tool set for air-gapped environments (no network).

    Includes core, filesystem, and git but excludes all
    network-dependent tools.

    Returns:
        ToolSet without network tools

    Excludes:
        - web_search, web_fetch, http_request, fetch_url
    """
```

**Example**:

```python
tools = ToolSet.airgapped()
# CORE, FILESYSTEM, GIT without WEB tools
```

#### coding()

```python
@classmethod
def coding(cls) -> ToolSet:
    """Tool set optimized for coding tasks.

    Includes core, filesystem, git, testing, and refactoring.

    Returns:
        ToolSet for coding workflows
    """
```

**Example**:

```python
tools = ToolSet.coding()
# CORE, FILESYSTEM, GIT, TESTING, REFACTORING
```

#### research()

```python
@classmethod
def research(cls) -> ToolSet:
    """Tool set optimized for research tasks.

    Includes core, web, and filesystem tools.

    Returns:
        ToolSet for research workflows
    """
```

**Example**:

```python
tools = ToolSet.research()
# CORE, FILESYSTEM, WEB
```

#### from_categories()

```python
@classmethod
def from_categories(cls, categories: list[str]) -> ToolSet:
    """Create ToolSet from category names.

    Args:
        categories: List of category names (e.g., ["core", "git"])

    Returns:
        ToolSet with specified categories
    """
```

**Examples**:

```python
# Single category
tools = ToolSet.from_categories(["filesystem"])

# Multiple categories
tools = ToolSet.from_categories(["filesystem", "git", "testing"])

# With custom category (if registered)
tools = ToolSet.from_categories(["filesystem", "my_custom_category"])
```

#### from_tools()

```python
@classmethod
def from_tools(cls, tools: list[str]) -> ToolSet:
    """Create ToolSet from specific tool names.

    Args:
        tools: List of tool names (e.g., ["read", "write", "git"])

    Returns:
        ToolSet with specified tools
    """
```

**Examples**:

```python
# Specific tools
tools = ToolSet.from_tools(["read", "write", "git"])

# Mix tool types
tools = ToolSet.from_tools(["read", "git", "web_search", "pytest"])
```

### Instance Methods

#### include()

```python
def include(self, *tools: str) -> ToolSet:
    """Add tools to the set.

    Args:
        *tools: Tool names to add

    Returns:
        New ToolSet with added tools
    """
```

**Examples**:

```python
# Start with default and add tools
tools = ToolSet.default().include("docker", "web_search")

# Chain includes
tools = (ToolSet.default()
    .include("docker")
    .include("web_search")
    .include("pytest"))
```

#### include_category()

```python
def include_category(self, category: str) -> ToolSet:
    """Add a category to the set.

    Args:
        category: Category name to add

    Returns:
        New ToolSet with added category
    """
```

**Examples**:

```python
# Add testing category
tools = ToolSet.default().include_category("testing")

# Chain category additions
tools = (ToolSet.default()
    .include_category("testing")
    .include_category("docker"))
```

#### exclude_tools()

```python
def exclude_tools(self, *tools: str) -> ToolSet:
    """Exclude tools from the set.

    Args:
        *tools: Tool names to exclude

    Returns:
        New ToolSet with exclusions
    """
```

**Examples**:

```python
# Remove shell from default (safer)
tools = ToolSet.default().exclude_tools("shell")

# Remove multiple tools
tools = ToolSet.full().exclude_tools("shell", "docker", "web_search")
```

#### get_tool_names()

```python
def get_tool_names(self) -> set[str]:
    """Get all tool names in this set.

    Returns the pre-resolved tool names. The cache is built during
    __post_init__ for performance (eager resolution).

    Returns:
        Set of tool names
    """
```

**Examples**:

```python
tools = ToolSet.default()
tool_names = tools.get_tool_names()
print(tool_names)
# {'read', 'write', 'edit', 'shell', 'search', 'list', 'glob', ...}
```

## ToolCategoryRegistry

Registry for tool categories supporting dynamic extension (OCP compliant).

```python
class ToolCategoryRegistry:
    """Registry for tool categories supporting dynamic extension.

    The registry provides a three-tier lookup:
    1. ToolMetadataRegistry (decorator-driven, highest priority)
    2. Custom registrations (from plugins/verticals)
    3. Built-in defaults (fallback)
    """
```

### Methods

#### get_instance()

```python
@classmethod
def get_instance(cls) -> ToolCategoryRegistry:
    """Get singleton instance of the registry.

    Returns:
        The ToolCategoryRegistry singleton
    """
```

#### register_category()

```python
def register_category(
    self,
    name: str,
    tools: set[str],
    description: str | None = None,
) -> None:
    """Register a new custom category.

    Use this to add vertical-specific or plugin-specific categories
    that aren't part of the built-in ToolCategory enum.

    Args:
        name: Category name (lowercase recommended)
        tools: Set of tool names in this category
        description: Optional description of the category

    Raises:
        ValueError: If category name conflicts with built-in enum
    """
```

**Examples**:

```python
registry = ToolCategoryRegistry.get_instance()

# Register custom RAG category
registry.register_category(
    "rag",
    {"rag_search", "rag_query", "rag_ingest", "rag_delete"},
    description="RAG vector database operations"
)

# Register custom CI/CD category
registry.register_category(
    "cicd",
    {"jenkins_build", "github_actions", "deploy"},
    description="CI/CD pipeline tools"
)
```

#### extend_category()

```python
def extend_category(self, name: str, tools: set[str]) -> None:
    """Extend an existing category with additional tools.

    Use this to add vertical-specific tools to built-in categories
    without modifying core code.

    Args:
        name: Category name (can be built-in or custom)
        tools: Set of tool names to add
    """
```

**Examples**:

```python
registry = ToolCategoryRegistry.get_instance()

# Extend built-in search category
registry.extend_category("search", {"semantic_rag_search", "vector_search"})

# Extend custom category
registry.extend_category("rag", {"rag_hybrid_search"})
```

#### get_tools()

```python
def get_tools(self, category: str) -> set[str]:
    """Get all tools in a category (merged from all sources).

    Lookup priority:
    1. ToolMetadataRegistry (decorator-driven)
    2. Custom registrations
    3. Built-in defaults
    4. Extensions (added to result)

    Args:
        category: Category name (built-in enum value or custom)

    Returns:
        Set of tool names in this category
    """
```

**Examples**:

```python
registry = ToolCategoryRegistry.get_instance()

# Get built-in category tools
git_tools = registry.get_tools("git")
# {'status', 'diff', 'commit', 'log', 'branch', ...}

# Get custom category tools
rag_tools = registry.get_tools("rag")
# {'rag_search', 'rag_query', 'rag_ingest', 'rag_hybrid_search'}
```

#### get_all_categories()

```python
def get_all_categories(self) -> set[str]:
    """Get all available category names.

    Includes built-in categories, custom registrations, and
    categories discovered from ToolMetadataRegistry.

    Returns:
        Set of all category names
    """
```

**Examples**:

```python
registry = ToolCategoryRegistry.get_instance()
all_categories = registry.get_all_categories()
# {'core', 'filesystem', 'git', 'search', 'web', 'rag', 'cicd', ...}
```

## Built-in Tools

### Core Tools

| Tool | Description |
|------|-------------|
| `read` | Read file contents |
| `write` | Write content to file |
| `edit` | Edit file with string replacement |
| `shell` | Execute shell commands |
| `search` | Search text in files |

### Filesystem Tools

| Tool | Description |
|------|-------------|
| `list` | List directory contents |
| `glob` | Find files by pattern |
| `find` | Find files by name/content |
| `mkdir` | Create directory |
| `rm` | Remove file/directory |
| `mv` | Move/rename file |
| `cp` | Copy file |

### Git Tools

| Tool | Description |
|------|-------------|
| `status` | Show git status |
| `diff` | Show git diff |
| `commit` | Commit changes |
| `log` | Show commit history |
| `branch` | Branch operations |

### Web Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web |
| `web_fetch` | Fetch webpage content |
| `http_request` | Make HTTP request |

### Testing Tools

| Tool | Description |
|------|-------------|
| `pytest` | Run pytest tests |
| `unittest` | Run unittest tests |
| `coverage` | Run coverage report |

## Custom Tool Registration

### Using Tool Metadata Decorator

```python
from victor.tools.metadata import tool

@tool(
    name="my_custom_tool",
    category="custom",
    description="My custom tool description"
)
def my_custom_tool(param: str) -> str:
    """Tool implementation."""
    return f"Processed: {param}"

# Tool is automatically registered
# Can be accessed via ToolSet
tools = ToolSet.from_categories(["custom"])
```

### Using Registry Directly

```python
from victor.framework.tools import get_category_registry

registry = get_category_registry()

# Register category
registry.register_category(
    "my_tools",
    {"tool1", "tool2", "tool3"}
)

# Use in ToolSet
tools = ToolSet.from_categories(["my_tools"])

# Or include in existing set
tools = ToolSet.default().include_category("my_tools")
```

## Tool Presets Comparison

| Preset | Categories | Tool Count | Use Case |
|--------|-----------|------------|----------|
| `minimal()` | CORE | ~5 | Simple read/write tasks |
| `default()` | CORE, FILESYSTEM, GIT | ~15 | General development |
| `coding()` | CORE, FILESYSTEM, GIT, TESTING, REFACTORING | ~25 | Full coding workflow |
| `research()` | CORE, FILESYSTEM, WEB | ~20 | Research and data gathering |
| `airgapped()` | CORE, FILESYSTEM, GIT | ~15 | Offline environments |
| `full()` | All categories | ~33 | Maximum capabilities |

## Performance Optimization

### Eager Resolution

`ToolSet` pre-resolves tool names at creation time for O(1) lookups:

```python
# Tool names resolved once at creation
tools = ToolSet.default()

# Fast O(1) containment check
if "read" in tools:
    print("Read tool available")

# No lock contention (cache pre-built)
names = tools.get_tool_names()  # Returns cached result
```

### Cache Invalidation

After modifying categories in the registry, invalidate cache if needed:

```python
registry = get_category_registry()
registry.extend_category("search", {"new_search_tool"})

# Create new ToolSet to pick up changes
tools = ToolSet.from_categories(["search"])
```

## Best Practices

### 1. Use Presets When Possible

```python
# Good - Use preset
tools = ToolSet.coding()

# Avoid - Manual category selection (unless needed)
tools = ToolSet.from_categories(["core", "filesystem", "git", "testing", "refactoring"])
```

### 2. Exclude Rather Than Include

```python
# Good - Start with preset and exclude
tools = ToolSet.full().exclude_tools("docker", "shell")

# Avoid - Manually include many tools
tools = ToolSet.from_tools(["read", "write", "git", "pytest", ...])
```

### 3. Use Custom Categories for Verticals

```python
# Good - Register custom category for vertical
registry = get_category_registry()
registry.register_category("myapp", {"app_deploy", "app_config", "app_monitor"})

# Use category
tools = ToolSet.from_categories(["myapp"])

# Avoid - Always listing tools individually
tools = ToolSet.from_tools(["app_deploy", "app_config", "app_monitor"])
```

### 4. Combine Presets and Modifications

```python
# Good - Start with preset, add/remove specific tools
tools = (ToolSet.coding()
    .exclude_tools("shell")  # Safer
    .include_category("web")  # Add research capability
)

# Avoid - Building from scratch
tools = ToolSet.from_categories([
    "core", "filesystem", "git", "testing", "refactoring", "web"
]).exclude_tools("shell")
```

### 5. Use Type Hints

```python
from victor.framework.tools import ToolSet

# Good - Type-annotated
def configure_agent(tools: ToolSet) -> Agent:
    return Agent(tools=tools)

tools: ToolSet = ToolSet.default()
agent = configure_agent(tools)
```

## Type Aliases

```python
# Type alias for convenience
Tools = ToolSet

# Type for tools parameter that accepts multiple formats
ToolsInput = ToolSet | list[str] | None
```

**Examples**:

```python
from victor.framework.tools import Tools, ToolsInput

# Use alias
def create_agent(tools: Tools) -> Agent:
    return Agent(tools=tools)

# Accept multiple formats
def configure_tools(tools: ToolsInput) -> ToolSet:
    if tools is None:
        return ToolSet.default()
    if isinstance(tools, list):
        return ToolSet.from_tools(tools)
    return tools
```

## Tool Deduplication

Victor automatically deduplicates tools across multiple sources (native, LangChain, MCP, plugins) to prevent redundant tools and save tokens.

### ToolSource Enum

```python
class ToolSource(str, Enum):
    """Tool source with priority for deduplication resolution."""
    
    NATIVE = "native"          # Priority: 100 (highest)
    LANGCHAIN = "langchain"    # Priority: 75
    MCP = "mcp"                # Priority: 50
    PLUGIN = "plugin"          # Priority: 25 (lowest)
```

### DeduplicationConfig

```python
from victor.tools.deduplication import DeduplicationConfig

config = DeduplicationConfig(
    enabled=True,
    priority_order=["native", "langchain", "mcp", "plugin"],
    whitelist=[],              # Tools to always allow
    blacklist=[],              # Tools to always skip
    strict_mode=False,         # Fail on conflicts vs. log and skip
    naming_enforcement=True,   # Enforce lgc_*, mcp_*, plg_* prefixes
    semantic_similarity_threshold=0.85
)
```

### ToolDeduplicator

```python
from victor.tools.deduplication import ToolDeduplicator

deduplicator = ToolDeduplicator(config)
result = deduplicator.deduplicate(tools)

# Result contains:
# - kept_tools: List of tools to register
# - skipped_tools: List of tools skipped due to conflicts
# - conflicts_resolved: Number of conflicts detected
# - logs: List of log messages
```

### Unified Naming Convention

Tools from different sources use consistent prefixes:

| Source | Prefix | Example | Priority |
|--------|--------|---------|----------|
| Native | none | `read`, `write`, `search` | 1 (highest) |
| LangChain | `lgc_` | `lgc_wikipedia`, `lgc_wolfram_alpha` | 2 |
| MCP | `mcp_` | `mcp_github_search`, `mcp_filesystem_read` | 3 |
| Plugin | `plg_` | `plg_custom_tool` | 4 (lowest) |

### Configuration

```python
from victor.config.tool_settings import ToolSettings

settings = ToolSettings(
    enable_tool_deduplication=True,
    deduplication_priority_order=["native", "langchain", "mcp", "plugin"],
    deduplication_whitelist=[],
    deduplication_blacklist=[],
    deduplication_strict_mode=False,
    deduplication_naming_enforcement=True,
    deduplication_semantic_threshold=0.85
)
```

### Usage Example

```python
from victor.tools.registry import ToolRegistry
from victor.tools.deduplication import ToolDeduplicator, DeduplicationConfig

# Create registry with deduplication enabled
config = DeduplicationConfig(enabled=True)
registry = ToolRegistry(deduplication_config=config)

# Register tools - deduplication happens automatically
registry.register(native_tool)
registry.register(langchain_adapter)  # Skipped if conflicts with native
registry.register(mcp_adapter)        # Skipped if conflicts with native/langchain

# Get deduplicated tool set
tools = registry.list_tools(only_enabled=True)
# Only includes highest priority tools from each conflict group
```

### Programmatic Usage

```python
from victor.tools.deduplication import ToolDeduplicator, DeduplicationConfig

# Create deduplicator
config = DeduplicationConfig(
    enabled=True,
    priority_order=["native", "langchain", "mcp"],
    whitelist=["special_tool"],  # Always allow this tool
    naming_enforcement=True
)
deduplicator = ToolDeduplicator(config)

# Deduplicate tool list
tools = [native_search, langchain_search, mcp_search]
result = deduplicator.deduplicate(tools)

# Check results
print(f"Kept {len(result.kept_tools)} tools")
print(f"Skipped {len(result.skipped_tools)} tools")
print(f"Resolved {result.conflicts_resolved} conflicts")

# Use kept tools
for tool in result.kept_tools:
    print(f"Registered: {tool.name}")
```

### Conflict Detection

Tools are compared by **normalized name**:
1. Convert to lowercase
2. Remove source prefixes (`lgc_`, `mcp_`, `plg_`)
3. Normalize separators (`_`, `-` → space)
4. Remove extra whitespace

**Examples**:
- `search` ↔ `web_search` ↔ `lgc_search` → **Conflict** (same base name after normalization)
- `read_file` ↔ `read` → **No conflict** (different base names)

### Priority Resolution

When conflicts are detected:
1. Extract source of both tools (native, langchain, mcp, plugin)
2. Compare priority weights
3. Keep higher priority tool
4. Skip lower priority tool
5. Log resolution with reason

**Example**:
```python
# Native 'search' (priority 100) vs LangChain 'search' (priority 75)
# Result: Native tool kept, LangChain tool skipped

# Logs show:
# "Conflict resolved: kept search (source=native), skipped 1 lower-priority tools"
# "Skipped lgc_search (source=langchain) in favor of search"
```

### NamingEnforcer

```python
from victor.tools.deduplication import NamingEnforcer, ToolSource

enforcer = NamingEnforcer(enforce=True)

# Enforce naming convention
new_name = enforcer.enforce_name(tool, ToolSource.LANGCHAIN)
# Input: "wikipedia"
# Output: "lgc_wikipedia"

# Strip prefix
original = enforcer.strip_prefix("lgc_wikipedia")
# Output: "wikipedia"

# Detect source from name
source = enforcer.detect_source_from_name("lgc_wikipedia")
# Output: ToolSource.LANGCHAIN
```

### Troubleshooting Deduplication

**Tool not appearing in list**:
```python
# Check if tool was skipped due to deduplication
import logging
logging.basicConfig(level=logging.DEBUG)
# Logs will show: "Skipped lgc_search (source=langchain) in favor of search"
```

**Both tools needed**:
```python
# Add to whitelist to bypass deduplication
config = DeduplicationConfig(
    whitelist=["special_tool", "custom_search"]
)
```

**Check tool source**:
```python
# Tool has _tool_source attribute
print(tool._tool_source)  # "native", "langchain", "mcp", or "plugin"
```

## See Also

- [Agent API](agent.md) - Agent creation with tools
- [Configuration API](../users/reference/config.md) - Tool configuration in profiles
- [Core APIs](core.md) - Tool execution framework
- [Tool Deduplication Architecture](../architecture/tool-deduplication.md) - Detailed design documentation
