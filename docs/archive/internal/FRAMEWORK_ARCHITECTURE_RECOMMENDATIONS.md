# Framework Architecture Recommendations

Analysis of mode workflow issues (BUILD, PLAN, EXPLORE) across providers revealed systemic architectural problems. This document proposes well-architected framework-level fixes that benefit all verticals.

## Root Cause Analysis

The identified issues share common root causes:

| Issue | Symptom | Root Cause |
|-------|---------|------------|
| shell_readonly in BUILD | Can't create files | Fragmented tool access control |
| Path resolution | Wrong file paths | No centralized path handling |
| Mode controller integration | Tools blocked unexpectedly | Multiple access control systems |
| Exploration limit | Forced completion too early | Scattered budget management |
| Vertical override | Mode settings ignored | Unclear precedence rules |
| Stage filtering | Write tools removed | Ad-hoc mode checks |

### Architectural Anti-Patterns Identified

1. **Fragmented Authority**: Tool access controlled by 5+ independent systems
2. **Scattered State**: Budgets/limits managed in different locations
3. **Ad-hoc Integration**: Mode checks added reactively in multiple places
4. **Missing Abstractions**: No unified path handling or access control protocol

---

## Proposed Architecture

### 1. Unified Tool Access Control (UTAC)

**Problem**: Tool access is controlled by multiple independent systems:
- `ModeController.is_tool_allowed()`
- `orchestrator.is_tool_enabled()`
- `orchestrator.get_enabled_tools()`
- `tool_selection._filter_tools_for_stage()`
- `TieredToolConfig` in verticals

**Solution**: Single `ToolAccessController` protocol with clear precedence.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ToolAccessController                          │
│  (Single source of truth for tool availability)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Precedence (highest to lowest):                                 │
│  1. Safety restrictions (always enforced)                        │
│  2. Mode settings (BUILD allows all, PLAN restricts writes)      │
│  3. Session restrictions (user-specified)                        │
│  4. Vertical defaults (domain-specific toolsets)                 │
│  5. Stage recommendations (soft suggestions)                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Protocol Definition**:

```python
# victor/protocols/tool_access.py

from typing import Protocol, Set, Optional
from dataclasses import dataclass
from enum import Enum, auto

class AccessLevel(Enum):
    """Tool access levels with clear semantics."""
    BLOCKED = auto()      # Safety/security blocked - cannot override
    DISALLOWED = auto()   # Mode/policy blocked - can be overridden
    ALLOWED = auto()      # Available for use
    REQUIRED = auto()     # Must be available (vertical core tools)

@dataclass
class ToolAccessDecision:
    """Decision about tool availability with reasoning."""
    tool_name: str
    level: AccessLevel
    reason: str
    source: str  # Which system made the decision

class IToolAccessController(Protocol):
    """Unified tool access control protocol."""

    def get_access_level(self, tool_name: str) -> ToolAccessDecision:
        """Get access level for a tool with full reasoning."""
        ...

    def get_available_tools(self) -> Set[str]:
        """Get all currently available tools."""
        ...

    def is_tool_available(self, tool_name: str) -> bool:
        """Quick check if tool is available."""
        ...

    def get_blocked_tools(self) -> Set[str]:
        """Get tools blocked by safety or mode."""
        ...

class ToolAccessController:
    """Concrete implementation with layered access control."""

    def __init__(
        self,
        safety_controller: "SafetyController",
        mode_controller: "ModeController",
        session_config: Optional["SessionConfig"] = None,
        vertical_config: Optional["VerticalConfig"] = None,
    ):
        self._safety = safety_controller
        self._mode = mode_controller
        self._session = session_config
        self._vertical = vertical_config

    def get_access_level(self, tool_name: str) -> ToolAccessDecision:
        # Layer 1: Safety (absolute)
        if self._safety.is_blocked(tool_name):
            return ToolAccessDecision(
                tool_name=tool_name,
                level=AccessLevel.BLOCKED,
                reason=self._safety.get_block_reason(tool_name),
                source="safety"
            )

        # Layer 2: Mode (BUILD overrides most restrictions)
        mode_config = self._mode.config
        if tool_name in mode_config.disallowed_tools:
            return ToolAccessDecision(
                tool_name=tool_name,
                level=AccessLevel.DISALLOWED,
                reason=f"Blocked by {mode_config.name} mode",
                source="mode"
            )

        if mode_config.allow_all_tools:
            return ToolAccessDecision(
                tool_name=tool_name,
                level=AccessLevel.ALLOWED,
                reason=f"{mode_config.name} mode allows all tools",
                source="mode"
            )

        # Layer 3: Session restrictions
        if self._session and tool_name in self._session.blocked_tools:
            return ToolAccessDecision(
                tool_name=tool_name,
                level=AccessLevel.DISALLOWED,
                reason="Blocked by session configuration",
                source="session"
            )

        # Layer 4: Vertical requirements
        if self._vertical:
            if tool_name in self._vertical.required_tools:
                return ToolAccessDecision(
                    tool_name=tool_name,
                    level=AccessLevel.REQUIRED,
                    reason=f"Required by {self._vertical.name} vertical",
                    source="vertical"
                )
            if tool_name not in self._vertical.allowed_tools:
                return ToolAccessDecision(
                    tool_name=tool_name,
                    level=AccessLevel.DISALLOWED,
                    reason=f"Not in {self._vertical.name} tool set",
                    source="vertical"
                )

        # Default: allowed
        return ToolAccessDecision(
            tool_name=tool_name,
            level=AccessLevel.ALLOWED,
            reason="No restrictions",
            source="default"
        )
```

**Benefits**:
- Single place to check tool availability
- Clear precedence rules
- Debugging: each decision includes reason and source
- All verticals benefit from consistent access control

---

### 2. Unified Budget Manager

**Problem**: Iteration limits, tool budgets, and exploration limits are scattered:
- `unified_task_tracker.py`: exploration_iterations, action_iterations, max_exploration_iterations
- `orchestrator.py`: tool_budget, max_iterations
- `intelligent_prompt_builder.py`: recommended_tool_budget
- `mode_controller.py`: exploration_multiplier

**Solution**: Single `BudgetManager` that centralizes all budget/limit logic.

```
┌─────────────────────────────────────────────────────────────────┐
│                      BudgetManager                               │
│  (Centralized budget and limit management)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Budgets:                                                        │
│  - tool_budget: Max tool calls per conversation                  │
│  - iteration_budget: Max LLM iterations                          │
│  - exploration_budget: Max read/search operations                │
│  - action_budget: Max write/modify operations                    │
│                                                                  │
│  Multipliers:                                                    │
│  - mode_multiplier: PLAN=2.5x, EXPLORE=3.0x, BUILD=2.0x         │
│  - model_multiplier: Based on model capabilities                 │
│  - task_multiplier: Based on task complexity                     │
│                                                                  │
│  Effective Budget = base * mode * model * task                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Protocol Definition**:

```python
# victor/protocols/budget.py

from typing import Protocol, Optional
from dataclasses import dataclass
from enum import Enum, auto

class BudgetType(Enum):
    TOOL_CALLS = auto()
    ITERATIONS = auto()
    EXPLORATION = auto()
    ACTION = auto()
    CONTEXT_TOKENS = auto()

@dataclass
class BudgetStatus:
    """Current budget status with utilization metrics."""
    budget_type: BudgetType
    used: int
    limit: int
    remaining: int
    utilization_pct: float
    is_exhausted: bool
    effective_multiplier: float

@dataclass
class BudgetConfig:
    """Budget configuration with multipliers."""
    base_tool_budget: int = 50
    base_iteration_budget: int = 50
    base_exploration_budget: int = 15
    base_action_budget: int = 100  # Actions usually unlimited

    mode_multiplier: float = 1.0
    model_multiplier: float = 1.0
    task_multiplier: float = 1.0

class IBudgetManager(Protocol):
    """Unified budget management protocol."""

    def get_status(self, budget_type: BudgetType) -> BudgetStatus:
        """Get current status for a budget type."""
        ...

    def consume(self, budget_type: BudgetType, amount: int = 1) -> bool:
        """Consume budget, returns False if exhausted."""
        ...

    def should_force_completion(self) -> tuple[bool, Optional[str]]:
        """Check if any budget requires forcing completion."""
        ...

    def get_recommended_budget_for_prompt(self) -> int:
        """Get budget to communicate to LLM in prompt."""
        ...

class BudgetManager:
    """Concrete budget manager implementation."""

    def __init__(self, config: BudgetConfig):
        self._config = config
        self._usage = {bt: 0 for bt in BudgetType}

    def _get_effective_limit(self, budget_type: BudgetType) -> int:
        """Calculate effective limit with all multipliers."""
        base = {
            BudgetType.TOOL_CALLS: self._config.base_tool_budget,
            BudgetType.ITERATIONS: self._config.base_iteration_budget,
            BudgetType.EXPLORATION: self._config.base_exploration_budget,
            BudgetType.ACTION: self._config.base_action_budget,
        }.get(budget_type, 50)

        multiplier = (
            self._config.mode_multiplier *
            self._config.model_multiplier *
            self._config.task_multiplier
        )
        return int(base * multiplier)

    def record_tool_call(self, tool_name: str, is_write: bool) -> None:
        """Record a tool call, categorizing as exploration or action."""
        self._usage[BudgetType.TOOL_CALLS] += 1

        if is_write:
            self._usage[BudgetType.ACTION] += 1
        else:
            self._usage[BudgetType.EXPLORATION] += 1

    def should_force_completion(self) -> tuple[bool, Optional[str]]:
        """Check if exploration budget exhausted (actions don't count)."""
        exploration_status = self.get_status(BudgetType.EXPLORATION)
        if exploration_status.is_exhausted:
            return True, "Exploration budget exhausted"
        return False, None
```

**Benefits**:
- All budget logic in one place
- Clear multiplier composition
- Exploration vs action tracking built-in
- Easy to add new budget types

---

### 3. Path Resolution Layer

**Problem**: LLMs often use incorrect paths (subdirectory prefixes, wrong relative paths). Fixes are scattered across individual tools.

**Solution**: Centralized `PathResolver` that all filesystem tools use.

```
┌─────────────────────────────────────────────────────────────────┐
│                       PathResolver                               │
│  (Centralized path normalization and resolution)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Resolution Steps:                                               │
│  1. Expand user (~) and environment variables                    │
│  2. Strip redundant cwd prefix (investor_homelab/utils → utils)  │
│  3. Try first path component removal as fallback                 │
│  4. Resolve to absolute path                                     │
│  5. Validate exists (with helpful error if not)                  │
│                                                                  │
│  Features:                                                       │
│  - Fuzzy matching for typos                                      │
│  - Suggest similar paths on failure                              │
│  - Track LLM path patterns for learning                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Protocol Definition**:

```python
# victor/protocols/path_resolver.py

from typing import Protocol, Optional, List
from pathlib import Path
from dataclasses import dataclass

@dataclass
class PathResolution:
    """Result of path resolution."""
    original_path: str
    resolved_path: Path
    was_normalized: bool
    normalization_applied: Optional[str] = None
    suggestions: List[str] = None  # Similar paths if not found

class IPathResolver(Protocol):
    """Path resolution protocol for filesystem tools."""

    def resolve(self, path: str, must_exist: bool = True) -> PathResolution:
        """Resolve a path, applying normalizations."""
        ...

    def resolve_file(self, path: str) -> PathResolution:
        """Resolve path that must be a file."""
        ...

    def resolve_directory(self, path: str) -> PathResolution:
        """Resolve path that must be a directory."""
        ...

    def suggest_similar(self, path: str, limit: int = 5) -> List[str]:
        """Suggest similar existing paths."""
        ...

class PathResolver:
    """Concrete path resolver with normalization strategies."""

    def __init__(self, cwd: Optional[Path] = None):
        self._cwd = cwd or Path.cwd()
        self._normalizers = [
            self._strip_cwd_prefix,
            self._strip_first_component,
            self._fix_common_typos,
        ]

    def _strip_cwd_prefix(self, path: str) -> Optional[str]:
        """Strip cwd name if path starts with it."""
        cwd_name = self._cwd.name
        if path.startswith(f"{cwd_name}/"):
            return path[len(cwd_name) + 1:]
        return None

    def _strip_first_component(self, path: str) -> Optional[str]:
        """Strip first path component as fallback."""
        parts = Path(path).parts
        if len(parts) > 1:
            return str(Path(*parts[1:]))
        return None

    def resolve(self, path: str, must_exist: bool = True) -> PathResolution:
        """Resolve path with automatic normalization."""
        # Try original path first
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = self._cwd / resolved

        if resolved.exists():
            return PathResolution(
                original_path=path,
                resolved_path=resolved,
                was_normalized=False
            )

        # Try normalizations
        for normalizer in self._normalizers:
            normalized = normalizer(path)
            if normalized and normalized != path:
                norm_resolved = (self._cwd / normalized).resolve()
                if norm_resolved.exists():
                    return PathResolution(
                        original_path=path,
                        resolved_path=norm_resolved,
                        was_normalized=True,
                        normalization_applied=normalizer.__name__
                    )

        # Not found - provide suggestions
        if must_exist:
            suggestions = self.suggest_similar(path)
            raise FileNotFoundError(
                f"Path not found: {path}. Did you mean: {suggestions}"
            )

        return PathResolution(
            original_path=path,
            resolved_path=resolved,
            was_normalized=False
        )
```

**Benefits**:
- All filesystem tools use consistent path handling
- Clear normalization pipeline
- Helpful error messages with suggestions
- Easy to add new normalization strategies

---

### 4. Mode-Aware Component Pattern

**Problem**: Components need to check mode settings but do so inconsistently with ad-hoc imports.

**Solution**: `ModeAware` mixin that provides consistent mode access.

```python
# victor/protocols/mode_aware.py

from typing import Protocol, Optional
from functools import cached_property

class IModeController(Protocol):
    """Mode controller protocol for dependency injection."""
    @property
    def config(self) -> "ModeConfig": ...
    @property
    def current_mode(self) -> "AgentMode": ...

class ModeAwareMixin:
    """Mixin for components that need mode awareness."""

    _mode_controller: Optional[IModeController] = None

    @cached_property
    def mode_controller(self) -> Optional[IModeController]:
        """Get mode controller with lazy initialization."""
        if self._mode_controller is None:
            try:
                from victor.agent.mode_controller import get_mode_controller
                self._mode_controller = get_mode_controller()
            except Exception:
                pass
        return self._mode_controller

    @property
    def is_build_mode(self) -> bool:
        """Check if in BUILD mode (allow_all_tools)."""
        mc = self.mode_controller
        return mc is not None and mc.config.allow_all_tools

    @property
    def is_exploration_mode(self) -> bool:
        """Check if in EXPLORE or PLAN mode."""
        mc = self.mode_controller
        if mc is None:
            return False
        return mc.current_mode.value in ("explore", "plan")

    @property
    def exploration_multiplier(self) -> float:
        """Get exploration multiplier for current mode."""
        mc = self.mode_controller
        if mc is None:
            return 1.0
        return getattr(mc.config, "exploration_multiplier", 1.0)
```

**Usage**:

```python
class ToolSelector(ModeAwareMixin):
    def _filter_tools_for_stage(self, tools, stage):
        # Clean mode check using mixin
        if self.is_build_mode:
            return tools  # BUILD mode keeps all tools
        # ... existing filtering logic
```

**Benefits**:
- Consistent mode access pattern
- No ad-hoc imports scattered through codebase
- Cached for performance
- Easy to test (inject mock mode controller)

---

## Implementation Plan

### Phase 1: Protocols (Week 1)
1. Create `victor/protocols/tool_access.py`
2. Create `victor/protocols/budget.py`
3. Create `victor/protocols/path_resolver.py`
4. Create `victor/protocols/mode_aware.py`

### Phase 2: Core Implementations (Week 2)
1. Implement `ToolAccessController`
2. Implement `BudgetManager`
3. Implement `PathResolver`
4. Add `ModeAwareMixin` to key components

### Phase 3: Integration (Week 3)
1. Refactor `orchestrator.py` to use `ToolAccessController`
2. Refactor `unified_task_tracker.py` to use `BudgetManager`
3. Refactor filesystem tools to use `PathResolver`
4. Add `ModeAwareMixin` to `ToolSelector`, `TaskTracker`

### Phase 4: Migration & Cleanup (Week 4)
1. Remove scattered mode checks
2. Remove duplicate tool access logic
3. Remove individual path normalization code
4. Update tests

---

## Migration Guide

### Before (scattered):
```python
# In orchestrator.py
def is_tool_enabled(self, tool_name):
    from victor.agent.mode_controller import get_mode_controller
    try:
        mc = get_mode_controller()
        if mc.config.allow_all_tools:
            return True
    except:
        pass
    # ... more logic scattered

# In tool_selection.py
def _filter_tools_for_stage(self, tools, stage):
    from victor.agent.mode_controller import get_mode_controller
    try:
        mc = get_mode_controller()
        if mc.config.allow_all_tools:
            return tools
    except:
        pass
    # ... similar logic duplicated
```

### After (unified):
```python
# In orchestrator.py
def __init__(self, tool_access: IToolAccessController, ...):
    self._tool_access = tool_access

def is_tool_enabled(self, tool_name):
    return self._tool_access.is_tool_available(tool_name)

# In tool_selection.py (with mixin)
class ToolSelector(ModeAwareMixin):
    def _filter_tools_for_stage(self, tools, stage):
        if self.is_build_mode:
            return tools
        # ...
```

---

## Benefits Summary

| Improvement | Benefit |
|-------------|---------|
| Unified Tool Access | Single source of truth, clear precedence, debuggable |
| Centralized Budgets | All limits in one place, proper multiplier composition |
| Path Resolution Layer | Consistent handling, better errors, easy to extend |
| Mode-Aware Pattern | Clean mode checks, testable, no scattered imports |
| Protocol-First | Dependency injection, mockable, SOLID principles |

All verticals benefit because:
1. **Coding vertical**: BUILD mode works correctly with shell/write tools
2. **Research vertical**: EXPLORE mode gets proper exploration limits
3. **DevOps vertical**: Mode settings respected over vertical defaults
4. **Data Analysis vertical**: Path resolution works in subdirectories
5. **Custom verticals**: Clear integration points via protocols
