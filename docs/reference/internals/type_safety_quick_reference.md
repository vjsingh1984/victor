# Type Safety Quick Reference for Developers

**Last Updated**: January 14, 2026
**Related Docs**: TYPE_SAFETY_PLAN.md, docs/type_ignore_audit.md

## Quick Status

- **Total `type: ignore` comments**: 128 (fully audited)
- **Modules with enhanced mypy**: 19 (10,541 lines)
- **Coordinator modules**: 17 files, 9,822 lines, 0 type ignores ✅
- **Current phase**: Phase 1 complete, Phase 2 ready to start

## Common Type Patterns

### 1. Adding Type Hints to Functions

**Before**:
```python
def process_data(data, max_items=10):
    result = []
    for item in data[:max_items]:
        result.append(transform(item))
    return result
```text

**After**:
```python
from typing import List, Any

def process_data(data: List[Any], max_items: int = 10) -> List[Any]:
    result: List[Any] = []
    for item in data[:max_items]:
        result.append(transform(item))
    return result
```

### 2. Using Protocol for Interfaces

**Before**:
```python
class Provider:
    def chat(self, messages):
        ...

def use_provider(provider: Provider):
    return provider.chat(messages)
```text

**After**:
```python
from typing import Protocol

class ProviderProtocol(Protocol):
    def chat(self, messages: list[Message]) -> str: ...

def use_provider(provider: ProviderProtocol) -> str:
    return provider.chat(messages)
```

### 3. Using TypedDict for Structured Data

**Before**:
```python
def create_tool_call(name, arguments):
    return {
        "name": name,
        "arguments": arguments,
        "timestamp": time.time()
    }
```text

**After**:
```python
from typing import TypedDict
import time

class ToolCall(TypedDict):
    name: str
    arguments: dict[str, Any]
    timestamp: float

def create_tool_call(name: str, arguments: dict[str, Any]) -> ToolCall:
    return {
        "name": name,
        "arguments": arguments,
        "timestamp": time.time()
    }
```

### 4. Using TypeVar for Generics

**Before**:
```python
class Result:
    def __init__(self, value):
        self.value = value
```text

**After**:
```python
from typing import TypeVar, Generic

T = TypeVar('T')

class Result(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value: T = value
```

### 5. Handling Optional Values

**Before**:
```python
def get_provider():
    if available:
        return provider
    return None
```text

**After**:
```python
from typing import Optional

def get_provider() -> Optional[ProviderProtocol]:
    if available:
        return provider
    return None
```

Or with Python 3.10+:
```python
def get_provider() -> ProviderProtocol | None:
    if available:
        return provider
    return None
```text

## When to Use `type: ignore`

### Acceptable Use Cases

**1. Optional Dependencies**
```python
try:
    import jira  # type: ignore  # Optional dependency: jira package
except ImportError:
    JIRA = None  # type: ignore  # Optional dependency
```

**2. Dynamic Attributes**
```python
wrapper._is_tool = True  # type: ignore[attr-defined]  # Dynamic tool marker
```text

**3. External Library Issues**
```python
import lancedb  # type: ignore  # External lib: lancedb has incomplete types
```

**4. Complex Decorators**
```python
return wrapper  # type: ignore  # Decorator pattern with complex typing
```text

### Unacceptable Use Cases

**❌ Missing Type Annotations**
```python
# Don't do this:
def bad_func(x):  # type: ignore  # Missing annotations
    return x + 1

# Do this instead:
def good_func(x: int) -> int:
    return x + 1
```

**❌ Ignoring Real Type Errors**
```python
# Don't do this:
result = process("string")  # type: ignore  # Wrong type!

# Do this instead:
result = process(123)  # Correct type
```text

## Running Type Checks

### Check All Modules
```bash
mypy victor
```

### Check Specific Module
```bash
mypy victor/agent/coordinators/chat_coordinator.py
```text

### Check with Strict Mode
```bash
mypy victor/agent/coordinators --strict
```

### Check with Error Codes
```bash
mypy victor/agent/coordinators --show-error-codes
```text

### Count Type Errors
```bash
mypy victor/agent/coordinators 2>&1 | grep "error:" | wc -l
```

## Common Mypy Error Codes

| Code | Meaning | Fix |
|------|---------|-----|
| `no-untyped-def` | Function missing type annotations | Add parameter and return types |
| `no-any-return` | Returning `Any` from typed function | Specify return type |
| `arg-type` | Argument type mismatch | Fix argument type |
| `attr-defined` | Accessing undefined attribute | Check attribute exists or add to Protocol |
| `assignment` | Assignment type mismatch | Fix variable type |
| `return-value` | Return type mismatch | Fix return type |
| `type-arg` | Missing type parameters | Add generic parameters, e.g., `list[int]` |
| `name-defined` | Name not defined | Import from typing |

## Phase 1 Modules (Enhanced Type Checking)

✅ **All coordinator modules** (17 files, 9,822 lines):
- victor/agent/coordinators/*
- victor/agent/factory_adapter.py

✅ **Already strict**:
- victor/config/*
- victor/storage/cache/*

## Review Checklist

Before submitting code with type changes:

- [ ] New code passes `mypy --strict` (for new modules)
- [ ] Modified code passes standard mypy
- [ ] No new `type: ignore` without justification comment
- [ ] Type annotations added to all functions
- [ ] Generic types properly parameterized
- [ ] Protocols used for interfaces
- [ ] TypedDict used for structured data

## Getting Help

**Documentation**:
- TYPE_SAFETY_PLAN.md - Comprehensive roadmap
- docs/type_ignore_audit.md - All `type: ignore` explanations
- docs/phase_2.3_summary.md - Phase 2.3 implementation details
- pyproject.toml - Mypy configuration

**Commands**:
```bash
# Check type errors in your module
mypy path/to/your/module.py --show-error-codes

# See all type ignores in your module
grep "type: ignore" path/to/your/module.py

# Count lines of code
wc -l path/to/your/module.py
```text

**Best Practices**:
1. Add type hints as you write code
2. Run mypy locally before pushing
3. Use Protocol for interfaces
4. Document `type: ignore` comments
5. Fix nearby type issues when modifying code

## Progress Tracking

**Phase 1** (Week 1): ✅ Complete
- Enhanced type checking for 19 modules
- Created comprehensive documentation
- Updated CI workflow

**Phase 2** (Week 2): Ready to start
- Protocol and interface modules
- victor.protocols.*
- victor.framework.coordinators.*

**Future Phases**: See TYPE_SAFETY_PLAN.md

---

**Remember**: We prefer gradual improvement with documentation over attempts at perfection. Not all code needs strict
  typing,
  but all exemptions should be justified.

---

**Last Updated:** February 01, 2026
**Reading Time:** 2 min
