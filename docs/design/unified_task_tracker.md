# Unified Task Tracker Design

## Status: Implemented ✅

**Implementation Date:** December 2024

**Key Commits:**
- `d952b71` - Initial fix for iteration limits
- `f2a3b8c` - Created UnifiedTaskTracker (~1250 lines)
- `8418f81` - Integrated into orchestrator
- `1ac5623` - Made unified_tracker PRIMARY decision maker
- `3fce7d9` - Removed legacy trackers, migration complete

## Problem Statement (Resolved)

Victor previously had two separate task tracking systems that overlapped in functionality:

1. **TaskMilestoneMonitor** (`milestone_monitor.py`) - Goal-aware milestone tracking _(DEPRECATED)_
2. **LoopDetector** (`loop_detector.py`) - Loop detection and budget enforcement _(DEPRECATED)_

This caused:
- Conflicting iteration limits (forcing at different counts)
- Task type mismatches (7 fine-grained vs 4 coarse types)
- Duplicate state tracking
- Complex synchronization code in orchestrator
- Harder to debug and maintain

## Proposed Solution: UnifiedTaskTracker

Consolidate both systems into a single `UnifiedTaskTracker` that provides:

### 1. Unified Task Types (Best of Both)

```python
class TaskType(Enum):
    """Unified task types combining fine-grained and coarse classifications."""

    # Action tasks (modify/create files)
    EDIT = "edit"           # Modify existing files
    CREATE = "create"       # Create new files with context
    CREATE_SIMPLE = "create_simple"  # Create files directly (no context needed)

    # Analysis tasks (read/search/understand)
    SEARCH = "search"       # Find/locate code or files
    ANALYZE = "analyze"     # Count, measure, analyze code
    RESEARCH = "research"   # Web research tasks

    # Other tasks
    DESIGN = "design"       # Conceptual/planning (no tools)
    GENERAL = "general"     # Ambiguous or mixed tasks
```

### 2. Unified Configuration

Single YAML config with all settings:

```yaml
# unified_task_config.yaml
task_types:
  edit:
    max_exploration_iterations: 8
    force_action_after_target_read: true
    tool_budget: 15
    loop_repeat_threshold: 4

  analyze:
    max_exploration_iterations: 20
    tool_budget: 30
    loop_repeat_threshold: 5

  # ... etc

model_overrides:
  deepseek*:
    exploration_multiplier: 1.5
    continuation_patience: 5

  claude*:
    exploration_multiplier: 1.0
    continuation_patience: 3

global:
  max_total_iterations: 50
  min_content_threshold: 150
  signature_history_size: 10
```

### 3. Core Class Design

```python
@dataclass
class UnifiedTaskProgress:
    """Single source of truth for task progress."""

    # Task classification
    task_type: TaskType = TaskType.GENERAL

    # Iteration tracking
    iteration_count: int = 0      # Productive iterations (tool calls)
    total_turns: int = 0          # All turns including continuations
    low_output_iterations: int = 0

    # Tool budget
    tool_calls: int = 0
    tool_budget: int = 50

    # Milestone tracking (from TaskMilestoneMonitor)
    milestones: Set[Milestone] = field(default_factory=set)
    target_files: Set[str] = field(default_factory=set)
    files_read: Set[str] = field(default_factory=set)
    files_modified: Set[str] = field(default_factory=set)

    # Loop detection (from LoopDetector)
    unique_resources: Set[str] = field(default_factory=set)
    file_read_ranges: Dict[str, List[FileReadRange]] = field(default_factory=dict)
    signature_history: deque = field(default_factory=lambda: deque(maxlen=10))
    loop_warning_given: bool = False

    # Stage tracking
    stage: ConversationStage = ConversationStage.INITIAL


class UnifiedTaskTracker:
    """Unified task tracking combining milestones and loop detection."""

    def __init__(
        self,
        config: Optional[UnifiedTaskConfig] = None,
        model_capabilities: Optional[ToolCallingCapabilities] = None,
    ):
        self.config = config or UnifiedTaskConfig.load()
        self.progress = UnifiedTaskProgress()
        self._exploration_multiplier = 1.0
        self._continuation_patience = 3

        if model_capabilities:
            self.set_model_capabilities(model_capabilities)

    # === Task Type & Configuration ===

    def set_task_type(self, task_type: TaskType) -> None:
        """Set task type from classification."""
        self.progress.task_type = task_type
        self._load_task_config(task_type)

    def set_model_capabilities(self, caps: ToolCallingCapabilities) -> None:
        """Set model-specific exploration settings."""
        self._exploration_multiplier = caps.exploration_multiplier
        self._continuation_patience = caps.continuation_patience

    # === Tool Call Recording ===

    def record_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Record a tool call - updates milestones, loops, and budgets."""
        self.progress.tool_calls += 1
        self.progress.iteration_count += 1
        self.progress.total_turns += 1

        # Update milestones
        self._update_milestones(tool_name, arguments)

        # Update loop detection
        self._update_loop_state(tool_name, arguments)

        # Track resources
        self._track_resource(tool_name, arguments)

    def increment_turn(self) -> None:
        """Record a turn without tool call (continuation prompt)."""
        self.progress.total_turns += 1

    # === Stop/Force Checking (Single Decision Point) ===

    def should_stop(self) -> StopDecision:
        """Single unified check for whether to stop.

        Checks in order:
        1. Manual force stop
        2. Tool budget exceeded
        3. True loop detected
        4. Max iterations exceeded
        5. Goal-based forcing (milestone-aware)
        """
        # Tool budget check
        if self.progress.tool_calls >= self.progress.tool_budget:
            return StopDecision(
                should_stop=True,
                reason="tool_budget_exceeded",
                hint=f"Tool budget exceeded ({self.progress.tool_calls}/{self.progress.tool_budget})"
            )

        # Loop check
        loop = self._check_loop()
        if loop:
            return StopDecision(should_stop=True, reason="loop_detected", hint=loop)

        # Iteration limit (with model multiplier and productivity ratio)
        effective_max = self._calculate_effective_max()
        if self.progress.iteration_count > effective_max:
            return StopDecision(
                should_stop=True,
                reason="max_iterations",
                hint=self._get_completion_hint()
            )

        # Goal-based forcing (milestone-aware)
        if self._should_force_for_goal():
            return StopDecision(
                should_stop=True,
                reason="goal_forcing",
                hint=self._get_goal_hint()
            )

        return StopDecision(should_stop=False)

    def _calculate_effective_max(self) -> int:
        """Calculate effective max iterations with all adjustments."""
        config = self._get_task_config()
        base_max = config.max_exploration_iterations

        # Apply model-specific multiplier
        model_adjusted = int(base_max * self._exploration_multiplier)

        # Apply productivity ratio adjustment
        if self.progress.total_turns > 0 and self.progress.iteration_count > 0:
            productivity = self.progress.iteration_count / self.progress.total_turns
            if productivity > 0:
                productivity_mult = min(2.0, max(1.0, 1.0 / productivity))
                return int(model_adjusted * productivity_mult)

        return model_adjusted
```

### 4. Migration Path

1. **Phase 1**: Add UnifiedTaskTracker alongside existing systems
2. **Phase 2**: Update orchestrator to use UnifiedTaskTracker
3. **Phase 3**: Deprecate TaskMilestoneMonitor and LoopDetector
4. **Phase 4**: Remove deprecated systems

### 5. Benefits

| Current | Unified |
|---------|---------|
| 2 task type enums | 1 enum with all types |
| 2 iteration trackers | 1 tracker |
| Mapping between systems | Single source of truth |
| 2 config files | 1 config file |
| Complex sync in orchestrator | Simple calls |
| Hard to debug | Clear logging |

### 6. Key Files to Create/Modify

**New Files:**
- `victor/agent/unified_task_tracker.py` - Main consolidated class
- `victor/config/unified_task_config.yaml` - Unified configuration

**Modified Files:**
- `victor/agent/orchestrator.py` - Use UnifiedTaskTracker
- `victor/agent/task_analyzer.py` - Update to use unified types

**Deprecated Files:**
- `victor/agent/milestone_monitor.py` → deprecated
- `victor/agent/loop_detector.py` → deprecated

## Implementation Estimate

| Phase | Work |
|-------|------|
| Phase 1 | Create UnifiedTaskTracker with combined features |
| Phase 2 | Update orchestrator to use new tracker |
| Phase 3 | Add deprecation warnings to old systems |
| Phase 4 | Remove old systems after validation |

## Open Questions

1. Should loop detection be optional (can disable for trusted models)?
2. Should we support per-conversation config overrides?
3. How to handle backward compatibility for existing YAML configs?
