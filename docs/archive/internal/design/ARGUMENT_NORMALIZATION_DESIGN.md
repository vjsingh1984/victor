# Argument Normalization System - Design Document

**Date**: November 27, 2025
**Issue**: Models output Python-style syntax (single quotes) instead of JSON (double quotes)
**Impact**: Tool calls fail with JSON parsing errors, breaking execution workflows
**Status**: Design phase

---

## Problem Statement

### Current Behavior
Models (especially Qwen) generate tool call arguments using Python dict syntax:
```python
'arguments': {'operations': '[{\'type\': \'modify\', \'path\': \'fibonacci.sh\', ...}]'}
```

Tools expect valid JSON:
```json
'arguments': {'operations': '[{"type": "modify", "path": "fibonacci.sh", ...}]'}
```

### Impact
- `edit_files` tool fails completely with JSON parsing errors
- Model cannot modify files, falls back to repeated `write_file` calls
- Execution workflows break (create but never execute)
- Poor user experience and wasted API calls

---

## Design Principles

1. **Defense in Depth**: Multiple fallback strategies, not single-point-of-failure
2. **Performance First**: Fast path for valid JSON (99% of cases)
3. **Transparency**: Log all normalizations for debugging and monitoring
4. **Extensibility**: Easy to add new normalization strategies
5. **Provider-Agnostic**: Works with any LLM provider
6. **Zero Breaking Changes**: Backward compatible with existing valid JSON

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Tool Call Argument Flow                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Argument Normalization Pipeline (orchestrator.py)       │
│     - Multi-layer validation and repair                     │
│     - Provider-specific strategies                           │
│     - Metrics collection                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Tool-Specific Validators (individual tools)              │
│     - Schema validation                                      │
│     - Type coercion                                          │
│     - Semantic validation                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Tool Execution (with validated arguments)                │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Design

### Component 1: Argument Normalization Pipeline

**Location**: `victor/agent/argument_normalizer.py` (new file)

```python
"""
Robust argument normalization for tool calls.
Handles malformed JSON from various LLM providers.
"""

from typing import Any, Dict, Optional, Tuple
import json
import ast
import re
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class NormalizationStrategy(Enum):
    """Strategies for normalizing malformed arguments."""
    DIRECT = "direct"              # Valid JSON, no changes
    PYTHON_AST = "python_ast"      # Python syntax → JSON via ast.literal_eval
    REGEX_QUOTES = "regex_quotes"  # Simple quote replacement
    MANUAL_REPAIR = "manual_repair"  # Complex repairs
    FAILED = "failed"              # All strategies failed


class ArgumentNormalizer:
    """
    Multi-layer argument normalization with fallback strategies.

    Design Philosophy:
    - Fast path first (valid JSON)
    - Graceful degradation (multiple fallback strategies)
    - Complete transparency (log all normalizations)
    - Provider-aware (can customize per provider)
    """

    def __init__(self, provider_name: str = "unknown"):
        self.provider_name = provider_name
        self.stats = {
            "total_calls": 0,
            "normalizations": {strategy.value: 0 for strategy in NormalizationStrategy},
            "failures": 0
        }

    def normalize_arguments(
        self,
        arguments: Dict[str, Any],
        tool_name: str
    ) -> Tuple[Dict[str, Any], NormalizationStrategy]:
        """
        Normalize tool call arguments through multi-layer pipeline.

        Returns:
            (normalized_arguments, strategy_used)
        """
        self.stats["total_calls"] += 1

        # Layer 1: Check if already valid (fast path)
        if self._is_valid_json_dict(arguments):
            self.stats["normalizations"][NormalizationStrategy.DIRECT.value] += 1
            return arguments, NormalizationStrategy.DIRECT

        # Layer 2: Try Python AST conversion for string values
        try:
            normalized = self._normalize_via_ast(arguments)
            if self._is_valid_json_dict(normalized):
                self.stats["normalizations"][NormalizationStrategy.PYTHON_AST.value] += 1
                logger.info(
                    f"Normalized {tool_name} arguments via AST conversion "
                    f"(provider={self.provider_name})"
                )
                return normalized, NormalizationStrategy.PYTHON_AST
        except Exception as e:
            logger.debug(f"AST normalization failed: {e}")

        # Layer 3: Regex-based quote replacement
        try:
            normalized = self._normalize_via_regex(arguments)
            if self._is_valid_json_dict(normalized):
                self.stats["normalizations"][NormalizationStrategy.REGEX_QUOTES.value] += 1
                logger.info(
                    f"Normalized {tool_name} arguments via regex "
                    f"(provider={self.provider_name})"
                )
                return normalized, NormalizationStrategy.REGEX_QUOTES
        except Exception as e:
            logger.debug(f"Regex normalization failed: {e}")

        # Layer 4: Manual repair for known patterns
        try:
            normalized = self._normalize_via_manual_repair(arguments, tool_name)
            if self._is_valid_json_dict(normalized):
                self.stats["normalizations"][NormalizationStrategy.MANUAL_REPAIR.value] += 1
                logger.info(
                    f"Normalized {tool_name} arguments via manual repair "
                    f"(provider={self.provider_name})"
                )
                return normalized, NormalizationStrategy.MANUAL_REPAIR
        except Exception as e:
            logger.debug(f"Manual repair failed: {e}")

        # All strategies failed
        self.stats["failures"] += 1
        logger.error(
            f"Failed to normalize {tool_name} arguments after all strategies "
            f"(provider={self.provider_name}). Original: {arguments}"
        )
        return arguments, NormalizationStrategy.FAILED

    def _is_valid_json_dict(self, obj: Any) -> bool:
        """Check if object is valid for JSON serialization."""
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False

    def _normalize_via_ast(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Python syntax to JSON via AST.

        Handles: '[{\'key\': \'value\'}]' → '[{"key": "value"}]'
        """
        normalized = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                # Try to parse as Python literal and convert to JSON
                try:
                    python_obj = ast.literal_eval(value)
                    # Convert to JSON string if it was a structure
                    if isinstance(python_obj, (list, dict)):
                        normalized[key] = json.dumps(python_obj)
                    else:
                        normalized[key] = value
                except (ValueError, SyntaxError):
                    normalized[key] = value
            else:
                normalized[key] = value
        return normalized

    def _normalize_via_regex(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple quote replacement via regex.

        Handles simple cases: {\'key\': \'value\'} → {"key": "value"}
        """
        normalized = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                # Replace escaped single quotes with double quotes
                # Pattern: \' → "
                repaired = value.replace("\\'", '"').replace("'", '"')
                normalized[key] = repaired
            else:
                normalized[key] = value
        return normalized

    def _normalize_via_manual_repair(
        self,
        arguments: Dict[str, Any],
        tool_name: str
    ) -> Dict[str, Any]:
        """
        Tool-specific manual repairs for known patterns.

        Extensible: Add new patterns as they're discovered.
        """
        # Tool-specific repairs
        if tool_name == "edit_files":
            return self._repair_edit_files_args(arguments)

        # Add more tool-specific repairs here

        return arguments

    def _repair_edit_files_args(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Repair common edit_files argument malformations.

        Known patterns:
        1. Python dict syntax in 'operations' field
        2. Missing quotes around field names
        3. Single quotes instead of double quotes
        """
        if "operations" in arguments:
            ops = arguments["operations"]
            if isinstance(ops, str):
                try:
                    # Try AST first
                    python_obj = ast.literal_eval(ops)
                    arguments["operations"] = json.dumps(python_obj)
                except:
                    # Fallback to regex
                    pass
        return arguments

    def get_stats(self) -> Dict[str, Any]:
        """Get normalization statistics for monitoring."""
        return {
            "provider": self.provider_name,
            "total_calls": self.stats["total_calls"],
            "normalizations": self.stats["normalizations"],
            "failures": self.stats["failures"],
            "success_rate": (
                (self.stats["total_calls"] - self.stats["failures"])
                / max(self.stats["total_calls"], 1)
            ) * 100
        }
```

### Component 2: Integration with Orchestrator

**Location**: `victor/agent/orchestrator.py` (modifications)

```python
# Add to imports
from victor.agent.argument_normalizer import ArgumentNormalizer, NormalizationStrategy

class AgentOrchestrator:
    def __init__(self, ...):
        # ... existing init code ...

        # Initialize argument normalizer
        provider_name = self.config.get("provider", "unknown")
        self.argument_normalizer = ArgumentNormalizer(provider_name=provider_name)

    async def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute tool calls with argument normalization."""
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})

            # NORMALIZE ARGUMENTS
            normalized_args, strategy = self.argument_normalizer.normalize_arguments(
                arguments,
                tool_name
            )

            # Log if normalization was applied
            if strategy != NormalizationStrategy.DIRECT:
                logger.warning(
                    f"Applied {strategy.value} normalization to {tool_name} arguments. "
                    f"Original: {arguments} → Normalized: {normalized_args}"
                )

            # Execute tool with normalized arguments
            try:
                result = await self._execute_single_tool(tool_name, normalized_args)
                results.append(result)
            except Exception as e:
                logger.error(f"Tool {tool_name} failed: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "tool": tool_name,
                    "normalization_strategy": strategy.value
                })

        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics including normalization stats."""
        return {
            "argument_normalization": self.argument_normalizer.get_stats(),
            # ... other metrics ...
        }
```

### Component 3: Tool-Specific Validators (Enhanced)

**Location**: `victor/tools/edit_files.py` (modifications)

```python
from pydantic import BaseModel, field_validator
from typing import List, Dict, Any
import json

class EditOperation(BaseModel):
    """Validated edit operation schema."""
    type: str  # "modify", "delete", "insert"
    path: str
    content: str = ""

    @field_validator('type')
    def validate_type(cls, v):
        allowed = {"modify", "delete", "insert"}
        if v not in allowed:
            raise ValueError(f"type must be one of {allowed}")
        return v

class EditFilesArgs(BaseModel):
    """Validated arguments for edit_files tool."""
    operations: List[EditOperation]

    @field_validator('operations', mode='before')
    def parse_operations(cls, v):
        """
        Parse operations with robust error handling.
        Accepts: JSON string, Python string, or list of dicts
        """
        if isinstance(v, str):
            # Already normalized by ArgumentNormalizer
            try:
                return json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid operations JSON: {e}")
        elif isinstance(v, list):
            return v
        else:
            raise ValueError(f"operations must be string or list, got {type(v)}")

async def edit_files(operations: str, **kwargs) -> Dict[str, Any]:
    """Edit files with validated arguments."""
    try:
        # Validate and parse arguments
        args = EditFilesArgs(operations=operations)

        # Execute edits
        results = []
        for op in args.operations:
            # ... perform edit ...
            pass

        return {"success": True, "results": results}

    except Exception as e:
        logger.error(f"edit_files failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "hint": "Check argument format - operations should be valid JSON array"
        }
```

---

## Configuration

**Location**: `victor/config/settings.py`

```python
# Argument normalization settings
ARGUMENT_NORMALIZATION = {
    "enabled": True,  # Global enable/disable
    "log_normalizations": True,  # Log when normalization is applied
    "fail_on_normalization_failure": False,  # Continue or fail on parse errors

    # Provider-specific overrides
    "provider_overrides": {
        "ollama": {
            "enabled": True,
            "strategies": ["ast", "regex", "manual"]  # Enabled strategies
        },
        "anthropic": {
            "enabled": False  # Anthropic outputs valid JSON
        }
    }
}
```

---

## Monitoring and Metrics

### Metrics to Track
1. **Normalization rate** by provider
2. **Strategy distribution** (which strategies are used most)
3. **Failure rate** (arguments that couldn't be normalized)
4. **Tool-specific patterns** (which tools need normalization most)

### Example Metrics Output
```python
{
    "argument_normalization": {
        "provider": "ollama",
        "total_calls": 1247,
        "normalizations": {
            "direct": 1180,      # 94.6% valid JSON
            "python_ast": 52,    # 4.2% needed AST conversion
            "regex_quotes": 10,  # 0.8% needed regex
            "manual_repair": 3,  # 0.2% needed manual repair
            "failed": 2          # 0.2% failed
        },
        "failures": 2,
        "success_rate": 99.8
    }
}
```

---

## Testing Strategy

### Unit Tests
- Test each normalization strategy independently
- Test with known malformed inputs
- Test performance (fast path should be <1μs)

### Integration Tests
- Test with real model outputs (Ollama, GPT, Claude)
- Test tool execution with normalized arguments
- Test failure handling

### Performance Tests
- Benchmark normalization overhead
- Ensure <1% impact on valid JSON (fast path)

---

## Rollout Plan

### Phase 1: Implementation (1 day)
1. Create `argument_normalizer.py` with all strategies
2. Integrate with `orchestrator.py`
3. Add unit tests

### Phase 2: Testing (1 day)
1. Test with Ollama/Qwen models (known to output Python syntax)
2. Test with other providers (ensure no regression)
3. Collect metrics

### Phase 3: Monitoring (ongoing)
1. Track normalization rates
2. Identify new patterns
3. Add tool-specific repairs as needed

---

## Benefits

1. **Immediate Fix**: Resolves edit_files JSON parsing failures
2. **Future-Proof**: Handles new models with different output formats
3. **Transparent**: Full logging and metrics for debugging
4. **Performance**: Zero overhead for valid JSON (99%+ of cases)
5. **Extensible**: Easy to add new strategies and tool-specific repairs
6. **Provider-Agnostic**: Works with any LLM provider

---

## Potential Issues and Mitigations

### Issue 1: Performance Overhead
**Mitigation**: Fast path first (direct JSON validation), only fallback on failure

### Issue 2: False Positives
**Mitigation**: Conservative strategies, extensive testing, logging

### Issue 3: Provider-Specific Edge Cases
**Mitigation**: Provider-aware configuration, easy to add overrides

### Issue 4: Security (Code Injection)
**Mitigation**: Use `ast.literal_eval()` (safe) instead of `eval()`, validate all repaired JSON

---

## Alternative Approaches Considered

### 1. Prompt Engineering (Rejected)
- **Approach**: Improve system prompt to force JSON output
- **Why Rejected**: Models still output Python syntax despite prompts, unreliable

### 2. Model Fine-Tuning (Rejected)
- **Approach**: Fine-tune models to output valid JSON
- **Why Rejected**: Not feasible for third-party models (Ollama), expensive

### 3. Post-Processing Only (Rejected)
- **Approach**: Fix JSON in tool execution layer only
- **Why Rejected**: Requires duplicating logic in every tool, not DRY

### 4. Pre-Processing in Provider (Considered)
- **Approach**: Normalize in Ollama provider before orchestrator
- **Why Rejected**: Not all providers need it, better as separate concern

---

## Future Enhancements

1. **LLM-Based Repair**: Use small LLM to repair complex malformations
2. **Auto-Learning**: Detect patterns and generate new repair strategies
3. **Tool Schema Enforcement**: Validate against tool schemas automatically
4. **Real-Time Metrics Dashboard**: Monitor normalization rates per provider/tool

---

## Files to Create/Modify

### New Files
- `victor/agent/argument_normalizer.py` (~300 lines)
- `tests/test_argument_normalizer.py` (~200 lines)

### Modified Files
- `victor/agent/orchestrator.py` (~30 lines added)
- `victor/tools/edit_files.py` (~40 lines modified)
- `victor/config/settings.py` (~15 lines added)

**Total Impact**: ~585 lines of code

---

## Success Metrics

1. ✅ edit_files tool success rate: 0% → 99%+
2. ✅ Execution workflow completion: Fix "create but don't execute" issue
3. ✅ Zero performance regression for valid JSON
4. ✅ Comprehensive logging for debugging
5. ✅ Extensible for future providers/models

---

**Status**: Ready for implementation
**Priority**: HIGH (blocks execution workflows)
**Estimated Effort**: 2 days (implementation + testing)
