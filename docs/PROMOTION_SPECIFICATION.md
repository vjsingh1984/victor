# Victor Framework: Core Promotion Specification

## Implementation Status (December 2025)

**Status: ✅ COMPLETE**

All promotions have been implemented:

| Promotion | Description | Status | Files Created |
|-----------|-------------|--------|---------------|
| 1 | Safety Infrastructure Extension | ✅ | `victor/safety/{code_patterns,infrastructure,types}.py` |
| 2 | Mode Configuration Registry | ✅ | `victor/core/mode_config.py` |
| 3 | Tool Dependency Graph | ✅ | `victor/tools/tool_graph.py` |
| 4 | Tool Selection Strategy Protocol | ✅ | `victor/tools/selection/{protocol,registry}.py` |
| 5 | Typed Tool Execution Context | ✅ | `victor/tools/context.py` |

**Tests:**
- ✅ `tests/unit/safety/` - Safety module tests
- ✅ `tests/unit/core/` - Core module tests
- ✅ `tests/unit/tools/test_tool_graph.py` - Tool graph tests
- ✅ `tests/unit/tools/test_selection.py` - Tool selection tests
- ✅ `tests/unit/tools/test_context.py` - Context tests

---

## Overview

This document specifies the exact modules, interfaces, and implementations to be promoted from verticals to victor-core for cross-vertical benefit.

---

## Promotion 1: Safety Infrastructure Extension

### Current State

```
victor/safety/           # EXISTING (Partial)
├── __init__.py
├── secrets.py           # 19+ credential patterns
└── pii.py               # 20+ PII types

victor/verticals/coding/safety.py        # To consolidate
victor/verticals/devops/safety.py        # To consolidate
victor/verticals/data_analysis/safety.py # To consolidate
victor/verticals/research/safety.py      # To consolidate
```

### Target State

```
victor/safety/
├── __init__.py              # Unified exports
├── base.py                  # Base classes and protocols
├── secrets.py               # [EXISTING] Credential detection
├── pii.py                   # [EXISTING] PII detection
├── code_patterns.py         # [NEW] From coding/safety.py
├── infrastructure.py        # [NEW] From devops/safety.py
├── misinformation.py        # [NEW] From research/safety.py
├── scanner.py               # [NEW] Unified scanner
└── registry.py              # [NEW] Domain pattern registry
```

### New Interfaces

```python
# victor/safety/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Pattern
import re


class SafetySeverity(Enum):
    """Severity levels for safety findings."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyCategory(Enum):
    """Categories of safety patterns."""
    CREDENTIAL = "credential"
    PII = "pii"
    CODE = "code"
    INFRASTRUCTURE = "infrastructure"
    MISINFORMATION = "misinformation"
    CUSTOM = "custom"


@dataclass
class SafetyPattern:
    """Definition of a safety pattern."""
    name: str
    pattern: str  # Regex pattern
    severity: SafetySeverity
    category: SafetyCategory
    description: str
    remediation: Optional[str] = None

    @property
    def compiled_pattern(self) -> Pattern:
        return re.compile(self.pattern, re.IGNORECASE | re.MULTILINE)


@dataclass
class SafetyMatch:
    """A match found by safety scanning."""
    pattern_name: str
    matched_text: str
    severity: SafetySeverity
    category: SafetyCategory
    line_number: Optional[int] = None
    column: Optional[int] = None
    context: Optional[str] = None
    remediation: Optional[str] = None


class SafetyScanner(ABC):
    """Protocol for safety scanners."""

    @abstractmethod
    def get_patterns(self) -> List[SafetyPattern]:
        """Get all patterns this scanner checks."""
        ...

    @abstractmethod
    def scan(self, content: str) -> List[SafetyMatch]:
        """Scan content for safety issues."""
        ...

    def scan_file(self, file_path: str) -> List[SafetyMatch]:
        """Scan a file for safety issues."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return self.scan(f.read())


# victor/safety/registry.py
class SafetyPatternRegistry:
    """Registry for safety patterns across domains."""

    _instance: Optional["SafetyPatternRegistry"] = None
    _patterns: Dict[SafetyCategory, List[SafetyPattern]] = {}
    _scanners: Dict[SafetyCategory, SafetyScanner] = {}

    @classmethod
    def get_instance(cls) -> "SafetyPatternRegistry":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_builtin()
        return cls._instance

    def register_pattern(self, pattern: SafetyPattern) -> None:
        """Register a safety pattern."""
        if pattern.category not in self._patterns:
            self._patterns[pattern.category] = []
        self._patterns[pattern.category].append(pattern)

    def register_scanner(self, category: SafetyCategory, scanner: SafetyScanner) -> None:
        """Register a scanner for a category."""
        self._scanners[category] = scanner

    def get_patterns(self, category: Optional[SafetyCategory] = None) -> List[SafetyPattern]:
        """Get patterns, optionally filtered by category."""
        if category:
            return self._patterns.get(category, [])
        return [p for patterns in self._patterns.values() for p in patterns]

    def scan_all(self, content: str) -> List[SafetyMatch]:
        """Scan content with all registered scanners."""
        matches = []
        for scanner in self._scanners.values():
            matches.extend(scanner.scan(content))
        return matches

    def _register_builtin(self) -> None:
        """Register built-in patterns and scanners."""
        from victor.safety.secrets import SecretScanner
        from victor.safety.pii import PIIScanner

        self.register_scanner(SafetyCategory.CREDENTIAL, SecretScanner())
        self.register_scanner(SafetyCategory.PII, PIIScanner())
```

### Patterns to Promote

#### From `coding/safety.py`:
```python
# victor/safety/code_patterns.py
GIT_DANGEROUS_PATTERNS = [
    SafetyPattern(
        name="git_force_push",
        pattern=r"git\s+push\s+.*--force",
        severity=SafetySeverity.HIGH,
        category=SafetyCategory.CODE,
        description="Force push can overwrite remote history",
        remediation="Use --force-with-lease instead",
    ),
    SafetyPattern(
        name="git_reset_hard",
        pattern=r"git\s+reset\s+--hard",
        severity=SafetySeverity.MEDIUM,
        category=SafetyCategory.CODE,
        description="Hard reset discards uncommitted changes",
        remediation="Stash changes before reset",
    ),
    # ... more patterns
]
```

#### From `devops/safety.py`:
```python
# victor/safety/infrastructure.py
INFRA_DANGEROUS_PATTERNS = [
    SafetyPattern(
        name="privileged_container",
        pattern=r"privileged:\s*true",
        severity=SafetySeverity.CRITICAL,
        category=SafetyCategory.INFRASTRUCTURE,
        description="Privileged containers have full host access",
        remediation="Use specific capabilities instead",
    ),
    SafetyPattern(
        name="root_user",
        pattern=r"USER\s+root",
        severity=SafetySeverity.MEDIUM,
        category=SafetyCategory.INFRASTRUCTURE,
        description="Running as root increases attack surface",
        remediation="Create non-root user in Dockerfile",
    ),
    # ... more patterns
]
```

---

## Promotion 2: Mode Configuration Registry

### Current State

Each vertical implements:
```python
# In each vertical's mode_config.py
class ModeConfigProvider(ModeConfigProviderProtocol):
    def get_mode_configs(self) -> Dict[str, ModeConfig]:
        return {
            "fast": ModeConfig(name="fast", tool_budget=10, ...),
            "thorough": ModeConfig(name="thorough", tool_budget=50, ...),
            ...
        }
```

### Target State

```python
# victor/core/mode_config.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class ModeType(Enum):
    """Standard mode types."""
    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"
    EXPLORE = "explore"
    PLAN = "plan"


@dataclass
class ModeConfig:
    """Configuration for an operational mode."""
    name: str
    tool_budget: int = 25
    max_iterations: int = 40
    exploration_multiplier: float = 1.0
    allowed_tools: Optional[List[str]] = None
    blocked_tools: Optional[List[str]] = None
    description: str = ""

    # Execution settings
    enable_parallel: bool = True
    max_concurrent_tools: int = 5
    enable_caching: bool = True

    # Quality settings
    quality_threshold: float = 0.80
    require_grounding: bool = True


class ModeConfigRegistry:
    """Central registry for operational modes."""

    _instance: Optional["ModeConfigRegistry"] = None

    # Default modes applicable to all verticals
    DEFAULT_MODES: Dict[str, ModeConfig] = {
        "fast": ModeConfig(
            name="fast",
            tool_budget=10,
            max_iterations=20,
            exploration_multiplier=0.5,
            description="Quick responses, minimal exploration",
        ),
        "balanced": ModeConfig(
            name="balanced",
            tool_budget=25,
            max_iterations=40,
            exploration_multiplier=1.0,
            description="Default balanced mode",
        ),
        "thorough": ModeConfig(
            name="thorough",
            tool_budget=50,
            max_iterations=60,
            exploration_multiplier=1.5,
            description="Deep exploration, comprehensive analysis",
        ),
        "explore": ModeConfig(
            name="explore",
            tool_budget=30,
            max_iterations=80,
            exploration_multiplier=3.0,
            blocked_tools=["write", "edit", "rm"],
            description="Read-only exploration mode",
        ),
        "plan": ModeConfig(
            name="plan",
            tool_budget=40,
            max_iterations=60,
            exploration_multiplier=2.5,
            description="Planning and analysis mode",
        ),
    }

    def __init__(self):
        self._vertical_modes: Dict[str, Dict[str, ModeConfig]] = {}

    @classmethod
    def get_instance(cls) -> "ModeConfigRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_modes(self, vertical: str, modes: Dict[str, ModeConfig]) -> None:
        """Register modes for a vertical (overrides defaults)."""
        self._vertical_modes[vertical] = modes

    def get_mode(self, mode_name: str, vertical: Optional[str] = None) -> ModeConfig:
        """Get mode config, checking vertical-specific then defaults."""
        # Check vertical-specific first
        if vertical and vertical in self._vertical_modes:
            if mode_name in self._vertical_modes[vertical]:
                return self._vertical_modes[vertical][mode_name]

        # Fall back to defaults
        if mode_name in self.DEFAULT_MODES:
            return self.DEFAULT_MODES[mode_name]

        raise ValueError(f"Unknown mode: {mode_name}")

    def get_all_modes(self, vertical: Optional[str] = None) -> Dict[str, ModeConfig]:
        """Get all available modes for a vertical."""
        modes = dict(self.DEFAULT_MODES)
        if vertical and vertical in self._vertical_modes:
            modes.update(self._vertical_modes[vertical])
        return modes

    def list_mode_names(self, vertical: Optional[str] = None) -> List[str]:
        """List available mode names."""
        return list(self.get_all_modes(vertical).keys())


# Convenience functions
def get_mode(mode_name: str, vertical: Optional[str] = None) -> ModeConfig:
    """Get a mode configuration."""
    return ModeConfigRegistry.get_instance().get_mode(mode_name, vertical)


def register_vertical_modes(vertical: str, modes: Dict[str, ModeConfig]) -> None:
    """Register modes for a vertical."""
    ModeConfigRegistry.get_instance().register_modes(vertical, modes)
```

### Vertical Simplification

```python
# Before: victor/verticals/coding/mode_config.py (60 lines)
class CodingModeConfigProvider(ModeConfigProviderProtocol):
    def get_mode_configs(self) -> Dict[str, ModeConfig]:
        return {
            "fast": ModeConfig(name="fast", tool_budget=10, max_iterations=20),
            "balanced": ModeConfig(name="balanced", tool_budget=25, max_iterations=40),
            "thorough": ModeConfig(name="thorough", tool_budget=50, max_iterations=60),
            "explore": ModeConfig(name="explore", tool_budget=30, max_iterations=80),
            "plan": ModeConfig(name="plan", tool_budget=40, max_iterations=60),
        }

# After: victor/verticals/coding/mode_config.py (10 lines)
from victor.core.mode_config import register_vertical_modes, ModeConfig

# Only register coding-specific overrides
register_vertical_modes("coding", {
    "debug": ModeConfig(
        name="debug",
        tool_budget=30,
        max_iterations=100,
        description="Extended debugging mode",
    ),
})
```

---

## Promotion 3: Tool Dependency Graph

### Current State

Two incompatible formats:
```python
# victor/verticals/coding/tool_dependencies.py
@dataclass
class ToolDependency:
    tool_name: str
    depends_on: List[str]
    enables: List[str]
    weight: float

# victor/verticals/devops/tool_dependencies.py
TOOL_TRANSITIONS: Dict[str, List[Tuple[str, float]]] = {
    "read": [("edit", 0.6), ("grep", 0.3)],
    "grep": [("read", 0.5), ("semantic_search", 0.3)],
}
```

### Target State

```python
# victor/tools/tool_graph.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum


class DependencyType(Enum):
    """Types of tool dependencies."""
    REQUIRES = "requires"      # Must run before
    ENABLES = "enables"        # Makes possible
    SUGGESTS = "suggests"      # Often follows
    CONFLICTS = "conflicts"    # Should not run together


@dataclass
class ToolTransition:
    """A transition between tools with probability."""
    from_tool: str
    to_tool: str
    probability: float  # 0.0 to 1.0
    dependency_type: DependencyType = DependencyType.SUGGESTS
    context: Optional[str] = None  # e.g., "after_error", "success"


@dataclass
class ToolNode:
    """A node in the tool dependency graph."""
    name: str
    category: str = ""
    outgoing: List[ToolTransition] = field(default_factory=list)
    incoming: List[ToolTransition] = field(default_factory=list)


class ToolExecutionGraph:
    """Unified graph of tool execution patterns.

    Supports both:
    - ToolDependency format (coding vertical)
    - Transition probability format (devops vertical)
    """

    def __init__(self):
        self._nodes: Dict[str, ToolNode] = {}
        self._transitions: List[ToolTransition] = []

    def add_node(self, name: str, category: str = "") -> ToolNode:
        """Add or get a tool node."""
        if name not in self._nodes:
            self._nodes[name] = ToolNode(name=name, category=category)
        return self._nodes[name]

    def add_transition(
        self,
        from_tool: str,
        to_tool: str,
        probability: float,
        dependency_type: DependencyType = DependencyType.SUGGESTS,
        context: Optional[str] = None,
    ) -> None:
        """Add a transition between tools."""
        transition = ToolTransition(
            from_tool=from_tool,
            to_tool=to_tool,
            probability=probability,
            dependency_type=dependency_type,
            context=context,
        )
        self._transitions.append(transition)

        # Update nodes
        from_node = self.add_node(from_tool)
        to_node = self.add_node(to_tool)
        from_node.outgoing.append(transition)
        to_node.incoming.append(transition)

    def add_dependency(
        self,
        tool_name: str,
        depends_on: List[str],
        enables: List[str],
        weight: float = 1.0,
    ) -> None:
        """Add dependencies in ToolDependency format (coding style)."""
        for dep in depends_on:
            self.add_transition(dep, tool_name, weight, DependencyType.REQUIRES)
        for enabled in enables:
            self.add_transition(tool_name, enabled, weight, DependencyType.ENABLES)

    def add_transitions(
        self,
        transitions: Dict[str, List[Tuple[str, float]]]
    ) -> None:
        """Add transitions in dict format (devops style)."""
        for from_tool, targets in transitions.items():
            for to_tool, prob in targets:
                self.add_transition(from_tool, to_tool, prob)

    def get_next_tools(
        self,
        current_tool: str,
        context: Optional[str] = None,
        min_probability: float = 0.1,
    ) -> List[Tuple[str, float]]:
        """Get likely next tools with probabilities."""
        if current_tool not in self._nodes:
            return []

        node = self._nodes[current_tool]
        results = []
        for t in node.outgoing:
            if t.probability >= min_probability:
                if context is None or t.context is None or t.context == context:
                    results.append((t.to_tool, t.probability))

        return sorted(results, key=lambda x: -x[1])

    def get_required_tools(self, tool_name: str) -> Set[str]:
        """Get tools that must run before this tool."""
        if tool_name not in self._nodes:
            return set()

        required = set()
        for t in self._nodes[tool_name].incoming:
            if t.dependency_type == DependencyType.REQUIRES:
                required.add(t.from_tool)
        return required

    def validate_sequence(self, sequence: List[str]) -> Tuple[bool, List[str]]:
        """Validate a tool call sequence.

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        for i, tool in enumerate(sequence):
            required = self.get_required_tools(tool)
            preceding = set(sequence[:i])

            missing = required - preceding
            if missing:
                issues.append(
                    f"Tool '{tool}' requires {missing} to run first"
                )

        return len(issues) == 0, issues

    def get_boost_score(
        self,
        candidate_tool: str,
        recent_tools: List[str],
        decay: float = 0.8,
    ) -> float:
        """Calculate selection boost based on recent tool history.

        Used by ToolSequenceTracker for 15-20% selection improvement.
        """
        boost = 0.0
        weight = 1.0

        for recent in reversed(recent_tools[-5:]):  # Last 5 tools
            next_tools = self.get_next_tools(recent)
            for tool, prob in next_tools:
                if tool == candidate_tool:
                    boost += prob * weight
                    break
            weight *= decay

        return min(boost, 0.3)  # Cap at 30% boost


# Global instance
_graph: Optional[ToolExecutionGraph] = None


def get_tool_graph() -> ToolExecutionGraph:
    """Get the global tool execution graph."""
    global _graph
    if _graph is None:
        _graph = ToolExecutionGraph()
        _register_default_patterns(_graph)
    return _graph


def _register_default_patterns(graph: ToolExecutionGraph) -> None:
    """Register default tool execution patterns."""
    # Common patterns across all verticals
    graph.add_transitions({
        "read": [
            ("edit", 0.6),
            ("grep", 0.4),
            ("semantic_search", 0.3),
        ],
        "grep": [
            ("read", 0.7),
            ("semantic_search", 0.4),
        ],
        "semantic_search": [
            ("read", 0.8),
            ("grep", 0.3),
        ],
        "edit": [
            ("read", 0.5),  # Verify changes
            ("shell", 0.4),  # Run tests
            ("git", 0.3),   # Commit
        ],
        "write": [
            ("read", 0.6),
            ("shell", 0.4),
        ],
        "shell": [
            ("read", 0.5),
            ("edit", 0.3),
        ],
        "git": [
            ("shell", 0.4),  # Run tests before commit
            ("read", 0.3),
        ],
    })
```

---

## Promotion 4: Tool Selection Strategy Protocol

### Target Interface

```python
# victor/tools/selection/protocol.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass
class PerformanceProfile:
    """Performance characteristics of a selection strategy."""
    avg_latency_ms: float
    requires_embeddings: bool
    requires_model_inference: bool
    memory_usage_mb: float


@dataclass
class ToolSelectionContext:
    """Context for tool selection."""
    prompt: str
    conversation_history: List[dict]
    current_stage: str
    task_type: Optional[str]
    provider_name: str
    model_name: str
    cost_budget: float
    enabled_tools: Optional[List[str]]
    disabled_tools: Optional[List[str]]


class ToolSelectionStrategy(Protocol):
    """Protocol for tool selection strategies."""

    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        ...

    def get_performance_profile(self) -> PerformanceProfile:
        """Get performance characteristics."""
        ...

    async def select_tools(
        self,
        context: ToolSelectionContext,
        max_tools: int = 10,
    ) -> List[str]:
        """Select relevant tools for the context.

        Returns:
            List of tool names, ordered by relevance
        """
        ...

    def supports_context(self, context: ToolSelectionContext) -> bool:
        """Check if this strategy can handle the context."""
        ...


# victor/tools/selection/registry.py
class ToolSelectionStrategyRegistry:
    """Registry of tool selection strategies."""

    _strategies: Dict[str, ToolSelectionStrategy] = {}

    @classmethod
    def register(cls, name: str, strategy: ToolSelectionStrategy) -> None:
        cls._strategies[name] = strategy

    @classmethod
    def get(cls, name: str) -> Optional[ToolSelectionStrategy]:
        return cls._strategies.get(name)

    @classmethod
    def get_best_strategy(cls, context: ToolSelectionContext) -> ToolSelectionStrategy:
        """Get the best strategy for the given context."""
        # Prefer semantic if embeddings available
        if "semantic" in cls._strategies:
            strategy = cls._strategies["semantic"]
            if strategy.supports_context(context):
                return strategy

        # Fall back to hybrid
        if "hybrid" in cls._strategies:
            return cls._strategies["hybrid"]

        # Fall back to keyword
        return cls._strategies.get("keyword")
```

---

## Promotion 5: Typed Tool Execution Context

### Target Interface

```python
# victor/tools/context.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from enum import Enum, auto


class Permission(Enum):
    """Permissions for tool execution."""
    READ_FILES = auto()
    WRITE_FILES = auto()
    EXECUTE_COMMANDS = auto()
    NETWORK_ACCESS = auto()
    GIT_OPERATIONS = auto()
    ADMIN_OPERATIONS = auto()


@dataclass
class ToolExecutionContext:
    """Typed context for tool execution.

    Replaces Dict[str, Any] _exec_ctx parameter.
    """
    # Session info
    session_id: str
    workspace_root: Path

    # Conversation state
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_stage: str = "INITIAL"

    # Budget tracking
    tool_budget_total: int = 25
    tool_budget_used: int = 0

    # Provider info
    provider_name: str = ""
    model_name: str = ""
    provider_capabilities: Dict[str, Any] = field(default_factory=dict)

    # Permissions
    user_permissions: Set[Permission] = field(
        default_factory=lambda: {Permission.READ_FILES}
    )

    # Tool-specific state
    open_files: Dict[str, str] = field(default_factory=dict)  # path -> content
    modified_files: Set[str] = field(default_factory=set)
    created_files: Set[str] = field(default_factory=set)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def tool_budget_remaining(self) -> int:
        return max(0, self.tool_budget_total - self.tool_budget_used)

    @property
    def can_write(self) -> bool:
        return Permission.WRITE_FILES in self.user_permissions

    @property
    def can_execute(self) -> bool:
        return Permission.EXECUTE_COMMANDS in self.user_permissions

    def use_budget(self, amount: int = 1) -> bool:
        """Use tool budget. Returns False if insufficient."""
        if self.tool_budget_used + amount > self.tool_budget_total:
            return False
        self.tool_budget_used += amount
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for backward compatibility."""
        return {
            "session_id": self.session_id,
            "workspace_root": str(self.workspace_root),
            "conversation_history": self.conversation_history,
            "current_stage": self.current_stage,
            "tool_budget_remaining": self.tool_budget_remaining,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "can_write": self.can_write,
            "can_execute": self.can_execute,
            **self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolExecutionContext":
        """Create from dict for backward compatibility."""
        return cls(
            session_id=data.get("session_id", ""),
            workspace_root=Path(data.get("workspace_root", ".")),
            conversation_history=data.get("conversation_history", []),
            current_stage=data.get("current_stage", "INITIAL"),
            tool_budget_total=data.get("tool_budget_total", 25),
            tool_budget_used=data.get("tool_budget_used", 0),
            provider_name=data.get("provider_name", ""),
            model_name=data.get("model_name", ""),
            metadata={k: v for k, v in data.items() if k not in cls.__dataclass_fields__},
        )
```

---

## Migration Guide

### For Vertical Maintainers

1. **Safety**: Import patterns from `victor.safety` instead of defining locally
2. **Mode Config**: Call `register_vertical_modes()` for custom modes only
3. **Tool Dependencies**: Use `get_tool_graph().add_transitions()` to register patterns
4. **Tool Execution**: Accept `ToolExecutionContext` instead of `Dict[str, Any]`

### Backward Compatibility

All promoted modules include:
- `from_dict()` / `to_dict()` methods for legacy code
- Deprecation warnings when using old patterns
- Gradual migration path (old code continues to work)

---

*Specification version: 1.0*
*Last updated: December 2025*
