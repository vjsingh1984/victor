# Victor CLI Gap Analysis: Grok vs DeepSeek Provider Testing

**Date:** 2025-12-16
**Test Environment:** webui/investor_homelab project
**Tested Providers:** Grok (xAI), DeepSeek
**Test Complexity Levels:** Simple, Medium, Complex (Agentic)

---

## Executive Summary

Comprehensive testing of Victor CLI with Grok and DeepSeek profiles revealed **6 critical gaps** and **4 design improvement opportunities**. The gaps primarily relate to tool execution patterns, parameter handling, and provider-specific quirks that the current architecture doesn't fully abstract.

### Key Findings

| Category | Grok | DeepSeek | Gap Severity |
|----------|------|----------|--------------|
| Simple Tasks | ✅ Completed | ⚠️ Excessive tool calls | Medium |
| Medium Tasks | ✅ Comprehensive output | ⚠️ Incomplete exploration | Medium |
| Complex Tasks | ✅ Full test suite generated | ❌ Tool execution errors | High |
| Tool Parameter Handling | Consistent | Inconsistent | High |
| Loop Detection | Not triggered | Should have triggered | High |

---

## Test Results Detail

### Test 1: Simple Complexity (List Python Files)

**Prompt:** "List all Python files in this project and describe what each module does"

#### Grok Results
- **Tool Calls:** 2 (`ls` with pattern twice)
- **Output Quality:** Complete, well-structured markdown table
- **Time:** 28.7s, ~857 tokens, 29.9 tok/s
- **Issues:** None

#### DeepSeek Results
- **Tool Calls:** 11 (`overview`, 5× `ls` with different paths/patterns)
- **Output Quality:** Incomplete - didn't finish listing all files
- **Time:** ~30s (estimated from tool calls)
- **Issues:**
  1. **GAP-5: Excessive Tool Calling** - DeepSeek made redundant `ls` calls with different paths
  2. **GAP-6: Missing Output Consolidation** - Multiple partial results not merged

### Test 2: Medium Complexity (Database Schema Analysis)

**Prompt:** "Analyze the database_schema.py file. Identify design issues, missing indexes, and suggest improvements."

#### Grok Results
- **Tool Calls:** 1 (`read`)
- **Output Quality:** Comprehensive analysis with:
  - 5 design issues (foreign keys, timestamps, data redundancy, constraints, unique keys)
  - 2 missing index categories
  - 7 specific improvement recommendations with code examples
- **Time:** 33.9s, ~1507 tokens, 44.5 tok/s
- **Issues:** None

#### DeepSeek Results
- **Tool Calls:** 9+ (`read`, `ls`, `read`×2, `graph`, `grep`, `refs`, `read`×2, `graph`)
- **Output Quality:** Partial - explored related files but didn't produce final analysis
- **Time:** Timeout (ran out without completing)
- **Issues:**
  1. **GAP-7: Over-exploration Without Synthesis** - Explored 6+ files but didn't synthesize findings
  2. **GAP-8: Missing Task Completion Signal** - No indication of analysis completion

### Test 3: Complex Complexity (Pytest Test Suite Generation)

**Prompt:** "Create a comprehensive pytest test suite for WebSearchClient class. Include fixtures, mocks for external APIs, tests for all public methods, error handling, and edge cases."

#### Grok Results
- **Tool Calls:** 1 (`read`)
- **Output Quality:** Complete 278-line test file with:
  - 4 fixtures (client, mock_requests, sample_result)
  - 18 test cases covering all public methods
  - Error handling tests
  - Edge case tests (empty query, None symbol)
- **Time:** ~45s (estimated)
- **Issues:** None

#### DeepSeek Results
- **Tool Calls:** 7 (`read`×2, `symbol`, `grep`, `read`×2)
- **Output Quality:** Incomplete - got stuck in tool exploration loop
- **Time:** Timeout
- **Issues:**
  1. **GAP-9: Symbol Tool Parameter Error** - `symbol() missing 1 required positional argument: 'file_path'`
  2. **GAP-10: Tool Error Recovery** - After tool error, didn't gracefully degrade
  3. **Loop Detection Should Have Triggered** - Read same file 2× with slight variations

---

## Gap Analysis Summary

### GAP-5: Excessive Tool Calling (DeepSeek)
**Severity:** Medium
**Description:** DeepSeek makes redundant tool calls for simple tasks that could be completed in 1-2 calls.
**Root Cause:** Lack of provider-specific tool guidance in system prompts.
**Impact:** Wasted tokens, slower responses, potential loop triggers.

### GAP-6: Missing Output Consolidation
**Severity:** Medium
**Description:** Multiple partial tool results aren't merged into coherent output.
**Root Cause:** No output aggregation strategy for multi-call scenarios.
**Impact:** Incomplete user-facing responses.

### GAP-7: Over-exploration Without Synthesis
**Severity:** High
**Description:** DeepSeek explores extensively but fails to synthesize findings.
**Root Cause:** Missing "synthesis checkpoint" in agentic workflows.
**Impact:** Tasks timeout without producing results.

### GAP-8: Missing Task Completion Signal
**Severity:** Medium
**Description:** No clear signal when agentic task is complete vs. stuck.
**Root Cause:** No completion detection heuristic.
**Impact:** User doesn't know if task is still running or stalled.

### GAP-9: Symbol Tool Parameter Error
**Severity:** High
**Description:** DeepSeek called `symbol(symbol_name='SearchResult')` without required `file_path`.
**Root Cause:** Tool calling adapter doesn't enforce required parameters.
**Impact:** Tool execution failure, broken workflows.

### GAP-10: Tool Error Recovery
**Severity:** High
**Description:** After tool error, model didn't recover gracefully.
**Root Cause:** No error recovery strategy in tool pipeline.
**Impact:** Single tool failure cascades to task failure.

---

## SOLID-Based Design Solutions

### Solution 1: Provider-Specific Tool Guidance (Strategy Pattern)

**Addresses:** GAP-5, GAP-7

```python
# victor/agent/tool_guidance/__init__.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class ToolGuidanceStrategy(ABC):
    """Strategy interface for provider-specific tool guidance."""

    @abstractmethod
    def get_guidance_prompt(self, task_type: str, available_tools: List[str]) -> str:
        """Return provider-specific tool usage guidance."""
        pass

    @abstractmethod
    def should_consolidate_calls(self, tool_history: List[Dict]) -> bool:
        """Determine if recent tool calls should be consolidated."""
        pass

    @abstractmethod
    def get_max_exploration_depth(self, task_complexity: str) -> int:
        """Return max tool calls before forcing synthesis."""
        pass


class GrokToolGuidance(ToolGuidanceStrategy):
    """Grok tends to be efficient - minimal guidance needed."""

    def get_guidance_prompt(self, task_type: str, available_tools: List[str]) -> str:
        return ""  # Grok handles tools well

    def should_consolidate_calls(self, tool_history: List[Dict]) -> bool:
        return False  # Grok already consolidates

    def get_max_exploration_depth(self, task_complexity: str) -> int:
        return {"simple": 3, "medium": 8, "complex": 15}.get(task_complexity, 10)


class DeepSeekToolGuidance(ToolGuidanceStrategy):
    """DeepSeek needs explicit guidance to avoid over-exploration."""

    def get_guidance_prompt(self, task_type: str, available_tools: List[str]) -> str:
        return """
IMPORTANT: Minimize tool calls. Before making a tool call:
1. Check if you already have the information from previous calls
2. Prefer broad queries over multiple narrow ones
3. After 5 tool calls, synthesize findings before continuing
"""

    def should_consolidate_calls(self, tool_history: List[Dict]) -> bool:
        # Trigger consolidation if same file read twice
        files_read = [h['args'].get('path') for h in tool_history if h['tool'] == 'read']
        return len(files_read) != len(set(files_read))

    def get_max_exploration_depth(self, task_complexity: str) -> int:
        return {"simple": 2, "medium": 5, "complex": 10}.get(task_complexity, 5)
```

### Solution 2: Output Aggregation Service (Observer Pattern)

**Addresses:** GAP-6, GAP-8

```python
# victor/agent/output_aggregator.py
from typing import Protocol, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class AggregationState(Enum):
    COLLECTING = "collecting"
    READY_TO_SYNTHESIZE = "ready_to_synthesize"
    COMPLETE = "complete"
    STUCK = "stuck"


@dataclass
class AggregatedResult:
    state: AggregationState
    results: List[Dict[str, Any]]
    synthesis_prompt: str
    confidence: float


class OutputAggregatorObserver(Protocol):
    """Observer interface for aggregation state changes."""

    def on_state_change(self, new_state: AggregationState) -> None: ...
    def on_result_added(self, result: Dict[str, Any]) -> None: ...
    def on_synthesis_ready(self, aggregated: AggregatedResult) -> None: ...


class OutputAggregator:
    """Aggregates tool outputs and detects completion state."""

    def __init__(self, max_results: int = 10, stale_threshold_seconds: float = 30):
        self._results: List[Dict] = []
        self._observers: List[OutputAggregatorObserver] = []
        self._state = AggregationState.COLLECTING
        self._last_result_time: float = 0
        self._max_results = max_results
        self._stale_threshold = stale_threshold_seconds

    def add_observer(self, observer: OutputAggregatorObserver) -> None:
        self._observers.append(observer)

    def add_result(self, tool_name: str, result: Any, metadata: Dict = None) -> None:
        import time
        self._results.append({
            "tool": tool_name,
            "result": result,
            "metadata": metadata or {},
            "timestamp": time.time()
        })
        self._last_result_time = time.time()
        self._update_state()
        for obs in self._observers:
            obs.on_result_added(self._results[-1])

    def _update_state(self) -> None:
        import time
        old_state = self._state

        # Check for stuck state (no results in threshold)
        if time.time() - self._last_result_time > self._stale_threshold:
            self._state = AggregationState.STUCK
        # Check for synthesis threshold
        elif len(self._results) >= self._max_results:
            self._state = AggregationState.READY_TO_SYNTHESIZE
        # Check for completion patterns
        elif self._detect_completion_pattern():
            self._state = AggregationState.COMPLETE

        if old_state != self._state:
            for obs in self._observers:
                obs.on_state_change(self._state)

    def _detect_completion_pattern(self) -> bool:
        """Detect if results form a complete answer."""
        if not self._results:
            return False
        # Pattern: Last result is a text generation (not a tool call)
        # Pattern: Decreasing tool call frequency
        return False  # Implement detection logic

    def get_synthesis_prompt(self) -> str:
        """Generate prompt to synthesize collected results."""
        result_summaries = []
        for r in self._results:
            result_summaries.append(f"- {r['tool']}: {str(r['result'])[:200]}...")

        return f"""
Based on the following {len(self._results)} tool results, provide a synthesized answer:

{chr(10).join(result_summaries)}

Synthesize these findings into a coherent response.
"""
```

### Solution 3: Required Parameter Enforcement (Decorator Pattern)

**Addresses:** GAP-9

```python
# victor/tools/parameter_enforcer.py
from functools import wraps
from typing import Callable, Dict, Any, Set
import logging

logger = logging.getLogger(__name__)


class MissingRequiredParameterError(Exception):
    """Raised when a required parameter is missing."""

    def __init__(self, tool_name: str, param_name: str, available_params: Set[str]):
        self.tool_name = tool_name
        self.param_name = param_name
        self.available_params = available_params
        super().__init__(
            f"Tool '{tool_name}' requires parameter '{param_name}'. "
            f"Available: {available_params}"
        )


def enforce_required_params(required: Set[str]):
    """Decorator to enforce required parameters on tool execution."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, **kwargs) -> Any:
            missing = required - set(kwargs.keys())
            if missing:
                # Try to infer missing params from context
                inferred = self._infer_missing_params(missing, kwargs)
                kwargs.update(inferred)

                # Check again after inference
                still_missing = required - set(kwargs.keys())
                if still_missing:
                    raise MissingRequiredParameterError(
                        tool_name=self.name,
                        param_name=list(still_missing)[0],
                        available_params=set(kwargs.keys())
                    )

            return await func(self, **kwargs)
        return wrapper
    return decorator


class ParameterInferenceStrategy:
    """Strategy for inferring missing parameters from context."""

    @staticmethod
    def infer_file_path(context: Dict[str, Any]) -> str | None:
        """Infer file_path from recent tool history."""
        # Look for recent read/grep operations
        tool_history = context.get("tool_history", [])
        for entry in reversed(tool_history[-5:]):
            if "path" in entry.get("args", {}):
                return entry["args"]["path"]
            if "file_path" in entry.get("args", {}):
                return entry["args"]["file_path"]
        return None

    @staticmethod
    def infer_symbol_file(symbol_name: str, context: Dict[str, Any]) -> str | None:
        """Infer file containing a symbol from codebase index."""
        # Query the symbol index
        symbol_index = context.get("symbol_index", {})
        if symbol_name in symbol_index:
            return symbol_index[symbol_name]["file_path"]
        return None
```

### Solution 4: Tool Error Recovery (Chain of Responsibility)

**Addresses:** GAP-10

```python
# victor/agent/error_recovery.py
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

class RecoveryAction(Enum):
    RETRY = "retry"
    RETRY_WITH_DEFAULTS = "retry_with_defaults"
    SKIP = "skip"
    FALLBACK_TOOL = "fallback_tool"
    ASK_USER = "ask_user"
    ABORT = "abort"


@dataclass
class RecoveryResult:
    action: RecoveryAction
    modified_args: Optional[Dict[str, Any]] = None
    fallback_tool: Optional[str] = None
    user_message: Optional[str] = None


class ErrorRecoveryHandler(ABC):
    """Abstract handler in the chain of responsibility."""

    def __init__(self):
        self._next_handler: Optional[ErrorRecoveryHandler] = None

    def set_next(self, handler: "ErrorRecoveryHandler") -> "ErrorRecoveryHandler":
        self._next_handler = handler
        return handler

    @abstractmethod
    def can_handle(self, error: Exception, tool_name: str, args: Dict) -> bool:
        pass

    @abstractmethod
    def handle(self, error: Exception, tool_name: str, args: Dict) -> RecoveryResult:
        pass

    def process(self, error: Exception, tool_name: str, args: Dict) -> RecoveryResult:
        if self.can_handle(error, tool_name, args):
            return self.handle(error, tool_name, args)
        elif self._next_handler:
            return self._next_handler.process(error, tool_name, args)
        else:
            return RecoveryResult(action=RecoveryAction.ABORT)


class MissingParameterHandler(ErrorRecoveryHandler):
    """Handle missing required parameter errors."""

    # Default values for common parameters
    DEFAULTS = {
        "file_path": ".",
        "path": ".",
        "limit": 100,
        "offset": 0,
    }

    def can_handle(self, error: Exception, tool_name: str, args: Dict) -> bool:
        return "missing" in str(error).lower() and "argument" in str(error).lower()

    def handle(self, error: Exception, tool_name: str, args: Dict) -> RecoveryResult:
        # Extract missing parameter from error message
        import re
        match = re.search(r"'(\w+)'", str(error))
        if match:
            param_name = match.group(1)
            if param_name in self.DEFAULTS:
                return RecoveryResult(
                    action=RecoveryAction.RETRY_WITH_DEFAULTS,
                    modified_args={**args, param_name: self.DEFAULTS[param_name]}
                )
        return RecoveryResult(action=RecoveryAction.SKIP)


class ToolNotFoundHandler(ErrorRecoveryHandler):
    """Handle tool not found errors with fallback."""

    FALLBACKS = {
        "symbol": "grep",  # Fallback from symbol lookup to grep
        "semantic_search": "grep",
        "tree": "ls",
    }

    def can_handle(self, error: Exception, tool_name: str, args: Dict) -> bool:
        return "not found" in str(error).lower() or "unknown tool" in str(error).lower()

    def handle(self, error: Exception, tool_name: str, args: Dict) -> RecoveryResult:
        if tool_name in self.FALLBACKS:
            return RecoveryResult(
                action=RecoveryAction.FALLBACK_TOOL,
                fallback_tool=self.FALLBACKS[tool_name]
            )
        return RecoveryResult(action=RecoveryAction.SKIP)


class NetworkErrorHandler(ErrorRecoveryHandler):
    """Handle network-related errors."""

    def can_handle(self, error: Exception, tool_name: str, args: Dict) -> bool:
        return any(x in str(error).lower() for x in ["timeout", "connection", "network"])

    def handle(self, error: Exception, tool_name: str, args: Dict) -> RecoveryResult:
        return RecoveryResult(
            action=RecoveryAction.RETRY,
            user_message="Network error, retrying..."
        )


# Build the chain
def build_recovery_chain() -> ErrorRecoveryHandler:
    chain = MissingParameterHandler()
    chain.set_next(ToolNotFoundHandler()).set_next(NetworkErrorHandler())
    return chain
```

### Solution 5: Synthesis Checkpoint (Template Method)

**Addresses:** GAP-7, GAP-8

```python
# victor/agent/synthesis_checkpoint.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class CheckpointResult:
    should_synthesize: bool
    reason: str
    suggested_prompt: Optional[str] = None


class SynthesisCheckpoint(ABC):
    """Abstract checkpoint that determines when to force synthesis."""

    @abstractmethod
    def check(self, tool_history: List[Dict], task_context: Dict) -> CheckpointResult:
        pass


class ToolCountCheckpoint(SynthesisCheckpoint):
    """Checkpoint based on tool call count."""

    def __init__(self, max_calls: int = 10):
        self.max_calls = max_calls

    def check(self, tool_history: List[Dict], task_context: Dict) -> CheckpointResult:
        if len(tool_history) >= self.max_calls:
            return CheckpointResult(
                should_synthesize=True,
                reason=f"Reached {self.max_calls} tool calls",
                suggested_prompt="Synthesize your findings so far before continuing."
            )
        return CheckpointResult(should_synthesize=False, reason="Under limit")


class DuplicateToolCheckpoint(SynthesisCheckpoint):
    """Checkpoint when same tool called repeatedly."""

    def check(self, tool_history: List[Dict], task_context: Dict) -> CheckpointResult:
        if len(tool_history) < 3:
            return CheckpointResult(should_synthesize=False, reason="Too few calls")

        last_three = [h["tool"] for h in tool_history[-3:]]
        if len(set(last_three)) == 1:
            return CheckpointResult(
                should_synthesize=True,
                reason=f"Same tool '{last_three[0]}' called 3 times",
                suggested_prompt="You've called the same tool repeatedly. Synthesize what you've learned."
            )
        return CheckpointResult(should_synthesize=False, reason="No duplicates")


class TimeoutApproachingCheckpoint(SynthesisCheckpoint):
    """Checkpoint when approaching time limit."""

    def __init__(self, warning_threshold: float = 0.7):
        self.warning_threshold = warning_threshold

    def check(self, tool_history: List[Dict], task_context: Dict) -> CheckpointResult:
        elapsed = task_context.get("elapsed_time", 0)
        timeout = task_context.get("timeout", 180)

        if elapsed / timeout > self.warning_threshold:
            return CheckpointResult(
                should_synthesize=True,
                reason=f"Used {int(elapsed/timeout*100)}% of time budget",
                suggested_prompt="Time is running short. Provide your best answer with current findings."
            )
        return CheckpointResult(should_synthesize=False, reason="Plenty of time")


class CompositeSynthesisCheckpoint(SynthesisCheckpoint):
    """Combines multiple checkpoints."""

    def __init__(self, checkpoints: List[SynthesisCheckpoint]):
        self.checkpoints = checkpoints

    def check(self, tool_history: List[Dict], task_context: Dict) -> CheckpointResult:
        for checkpoint in self.checkpoints:
            result = checkpoint.check(tool_history, task_context)
            if result.should_synthesize:
                return result
        return CheckpointResult(should_synthesize=False, reason="All checks passed")
```

---

## Implementation Status

### ✅ COMPLETED: GAP-9 - Parameter Enforcement Decorator

**Implementation:** `victor/agent/parameter_enforcer.py`
**Test Suite:** `tests/unit/test_parameter_enforcer.py` (35 tests)

Key components implemented:
- `ParameterType` enum for type validation (STRING, INTEGER, BOOLEAN, FLOAT, ARRAY, OBJECT)
- `InferenceStrategy` enum (NONE, FROM_CONTEXT, FROM_PREVIOUS_ARGS, FROM_DEFAULT, FROM_WORKING_DIR)
- `ParameterSpec` dataclass for parameter definitions
- `ParameterEnforcer` class with `validate()` and `enforce()` methods
- `@enforce_parameters` decorator for tool execution functions
- `create_enforcer_for_tool()` factory from JSON schema
- Pre-registered enforcers for `symbol`, `read`, `grep`, `ls`, `glob` tools

Features:
- Automatic parameter inference from context and previous tool args
- Type coercion (string→int, string→bool, etc.)
- Multiple inference keys support (`inference_keys=["path", "file_path", "file"]`)
- Graceful error handling with `ParameterInferenceError` and `ParameterValidationError`

### ✅ COMPLETED: GAP-10 - Error Recovery Chain

**Implementation:** `victor/agent/error_recovery.py`
**Test Suite:** `tests/unit/test_error_recovery.py` (69 tests)

Key components implemented:
- `RecoveryAction` enum (RETRY, RETRY_WITH_DEFAULTS, RETRY_WITH_INFERRED, SKIP, FALLBACK_TOOL, ASK_USER, ABORT)
- `RecoveryResult` dataclass with retry tracking
- `ErrorRecoveryHandler` abstract base class (Chain of Responsibility)
- 7 specialized handlers:
  - `MissingParameterHandler` - Provides defaults for common parameters
  - `TypeErrorHandler` - Converts string booleans/numbers
  - `FileNotFoundHandler` - Tries path variations
  - `ToolNotFoundHandler` - Falls back to alternative tools
  - `RateLimitHandler` - Exponential backoff
  - `NetworkErrorHandler` - Automatic retry
  - `PermissionErrorHandler` - User notification
- `build_recovery_chain()` factory function
- `recover_from_error()` convenience function

### ✅ COMPLETED: GAP-5/GAP-7 - Provider-Specific Tool Guidance

**Implementation:** `victor/agent/provider_tool_guidance.py`
**Test Suite:** `tests/unit/test_provider_tool_guidance.py` (39 tests)

Key components implemented:
- `ToolGuidanceStrategy` abstract base class (Strategy Pattern)
- 6 provider-specific strategies:
  - `GrokToolGuidance` - Minimal guidance (handles tools efficiently)
  - `DeepSeekToolGuidance` - Explicit guidance with synthesis checkpoints
  - `OllamaToolGuidance` - Strict guidance for local models
  - `AnthropicToolGuidance` - Minimal guidance (Claude handles tools well)
  - `OpenAIToolGuidance` - Moderate guidance
  - `DefaultToolGuidance` - Conservative defaults for unknown providers
- Provider method features:
  - `get_guidance_prompt(task_type, available_tools)` - Provider-specific prompts
  - `should_consolidate_calls(tool_history)` - Detect redundant tool usage
  - `get_max_exploration_depth(task_complexity)` - Limit exploration by complexity
  - `get_synthesis_checkpoint_prompt(tool_count)` - Force synthesis after N calls
- `get_tool_guidance_strategy()` factory function with caching
- Provider aliases (`xai` → Grok, `claude` → Anthropic, etc.)

Features:
- Automatic consolidation detection for DeepSeek (duplicate file reads, excessive ls calls)
- Task complexity-aware exploration limits (simple: 3, medium: 6, complex: 12 for DeepSeek)
- Synthesis checkpoints at threshold (5 tool calls for DeepSeek)
- Provider-aware guidance prompts injected into system prompts

---

## Implementation Roadmap

### ✅ Phase 1: Critical Fixes (GAP-9, GAP-10) - COMPLETE
1. ✅ Implement `ParameterEnforcer` decorator
2. ✅ Build error recovery chain of responsibility
3. ✅ Add unit tests for parameter validation (35 tests)
4. ✅ Add unit tests for error recovery (69 tests)

### ✅ Phase 2: Provider Optimization (GAP-5, GAP-7) - COMPLETE
1. ✅ Implement `ToolGuidanceStrategy` interface
2. ✅ Create Grok, DeepSeek, Ollama, Anthropic, OpenAI strategies
3. ✅ Add unit tests for provider tool guidance (39 tests)
4. ⏳ Integrate with `PromptBuilder` (pending pipeline integration)

### Phase 3: Output Management (GAP-6, GAP-8) - PENDING
1. Implement `OutputAggregator` with observer pattern
2. Add synthesis checkpoints
3. Create completion detection heuristics
4. Integration test with complex tasks

### Phase 4: Integration - PENDING
1. Integrate parameter enforcer into tool pipeline
2. Integrate error recovery into AgentOrchestrator
3. End-to-end tests with real providers
4. Performance benchmarks

---

## Appendix: Raw Test Logs

### Grok Simple Test
```
Tool Calls: ls(path='.', pattern='*.py'), ls(path='investor_homelab', pattern='*.py')
Output: 10 Python files listed with descriptions
Time: 28.7s | Tokens: ~857 | Speed: 29.9 tok/s
```

### DeepSeek Simple Test
```
Tool Calls: overview, ls×5 with varying paths
Output: Incomplete listing
Time: ~30s (timeout)
```

### Grok Medium Test
```
Tool Calls: read(path='database_schema.py')
Output: 7 design issues, 2 index categories, 7 improvements
Time: 33.9s | Tokens: ~1507 | Speed: 44.5 tok/s
```

### DeepSeek Medium Test
```
Tool Calls: read, ls, read×2, graph, grep, refs, read×2, graph
Output: Incomplete (explored but didn't synthesize)
Time: Timeout
```

### Grok Complex Test
```
Tool Calls: read(path='web_search_client.py')
Output: 278-line pytest test suite
Time: ~45s (estimated)
```

### DeepSeek Complex Test
```
Tool Calls: read×2, symbol (ERROR), grep, read×2
Output: Failed - tool error cascaded
Error: symbol() missing 1 required positional argument: 'file_path'
Time: Timeout
```
