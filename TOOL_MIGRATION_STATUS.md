# Tool Migration Status: Class-Based → Decorator Pattern

**Date:** November 24, 2025
**Goal:** Migrate all tools to consistent `@tool` decorator pattern

## Current Status

### ✅ Migrated to Decorator Pattern (6 tools)
1. ✅ `bash.py` - **JUST MIGRATED** (`execute_bash()`)
2. ✅ `code_executor_tool.py` - Already using decorators
3. ✅ `code_intelligence_tool.py` - Already using decorators
4. ✅ `filesystem.py` - Already using decorators (`read_file()`, `write_file()`, `list_directory()`)
5. ✅ `testing_tool.py` - Already using decorators (`run_tests()`)
6. ✅ `workflow_tool.py` - Already using decorators

### ⚠️ Still Using Class-Based Pattern (17 tools)

| Tool File | Lines | Complexity | Priority |
|-----------|-------|------------|----------|
| `cache_tool.py` | 225 | Medium | **HIGH** (small, high-impact) |
| `http_tool.py` | 290 | Medium | **HIGH** (commonly used) |
| `file_editor_tool.py` | 434 | High | **HIGH** (core functionality) |
| `git_tool.py` | 616 | High | **MEDIUM** (complex, many operations) |
| `batch_processor_tool.py` | 245 | Medium | MEDIUM |
| `cicd_tool.py` | 158 | Medium | MEDIUM |
| `code_review_tool.py` | 353 | High | MEDIUM |
| `database_tool.py` | 213 | Medium | MEDIUM |
| `dependency_tool.py` | 302 | Medium | MEDIUM |
| `docker_tool.py` | 227 | Medium | LOW |
| `documentation_tool.py` | 327 | Medium | LOW |
| `metrics_tool.py` | 240 | Medium | LOW |
| `refactor_tool.py` | 358 | High | LOW |
| `scaffold_tool.py` | 124 | Low | LOW |
| `security_scanner_tool.py` | 259 | Medium | LOW |
| `web_search_tool.py` | 135 | Medium | MEDIUM |

**Total:** 17 tools need migration (~4,700 lines of code)

## Migration Pattern

### Before (Class-Based):
```python
from victor.tools.base import BaseTool, ToolResult

class MyTool(BaseTool):
    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "My tool description"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "Parameter 1"}
            },
            "required": ["param1"]
        }

    async def execute(self, context: Dict[str, Any], **kwargs) -> ToolResult:
        param1 = kwargs.get("param1")

        # Tool logic here
        result = do_something(param1)

        return ToolResult(
            success=True,
            output=result,
            metadata={"processed": True}
        )
```

### After (Decorator-Based):
```python
from typing import Dict, Any
from victor.tools.decorators import tool

@tool
async def my_tool_operation(
    param1: str,
    optional_param: str = "default",
    timeout: int = 60
) -> Dict[str, Any]:
    """
    My tool description.

    Detailed explanation of what this tool does.

    Args:
        param1: Description of parameter 1.
        optional_param: Description of optional parameter.
        timeout: Operation timeout in seconds.

    Returns:
        Dictionary containing the operation results.
    """
    # Tool logic here
    result = do_something(param1)

    return {
        "success": True,
        "result": result,
        "processed": True
    }
```

## Key Migration Changes

### 1. Remove Class Boilerplate
- ❌ Remove `class XyzTool(BaseTool):`
- ❌ Remove `__init__()` method
- ❌ Remove `@property` decorators (name, description, parameters)
- ❌ Remove `ToolResult` wrapper
- ✅ Use simple `@tool` decorator

### 2. Simplify Function Signature
- ❌ `async def execute(self, context, **kwargs):`
- ✅ `async def operation_name(param1: type, param2: type = default):`

### 3. Return Simple Dictionaries
- ❌ `return ToolResult(success=True, output=data, metadata={})`
- ✅ `return {"success": True, "data": data, ...}`

### 4. Documentation via Docstrings
- Tool name: Inferred from function name
- Description: From function docstring
- Parameters: From function signature + type hints
- Auto-generated JSON schema from function signature

## Benefits of Decorator Pattern

1. **Less Boilerplate**: ~50% less code per tool
2. **Better Type Safety**: Explicit parameter types in signature
3. **Auto Documentation**: Docstrings → JSON schema automatically
4. **Easier Testing**: Functions are easier to test than classes
5. **Better IDE Support**: Type hints work better with functions
6. **Consistency**: Aligns with modern Python patterns

## Recommended Migration Order

### Phase 1: High-Priority (Quick Wins)
1. ✅ **bash.py** - COMPLETED
2. **cache_tool.py** (225 lines) - Simple, high-impact
3. **http_tool.py** (290 lines) - Commonly used
4. **web_search_tool.py** (135 lines) - Smaller tool

### Phase 2: Core Functionality
5. **file_editor_tool.py** (434 lines) - Critical tool
6. **git_tool.py** (616 lines) - Most complex, break into multiple @tool functions
7. **database_tool.py** (213 lines)

### Phase 3: Advanced Tools
8. **batch_processor_tool.py** (245 lines)
9. **code_review_tool.py** (353 lines)
10. **cicd_tool.py** (158 lines)
11. **dependency_tool.py** (302 lines)

### Phase 4: Specialized Tools
12. **docker_tool.py** (227 lines)
13. **documentation_tool.py** (327 lines)
14. **metrics_tool.py** (240 lines)
15. **refactor_tool.py** (358 lines)
16. **scaffold_tool.py** (124 lines)
17. **security_scanner_tool.py** (259 lines)

## Migration Checklist (Per Tool)

For each tool file:

- [ ] Read the current class implementation
- [ ] Identify all operations (methods besides execute())
- [ ] For simple tools: Create one `@tool` function
- [ ] For complex tools: Create multiple `@tool` functions (one per operation)
- [ ] Convert `ToolResult` returns to simple dictionaries
- [ ] Move docstrings to function level
- [ ] Add type hints to all parameters
- [ ] Test the migrated tool
- [ ] Update any tool registry/imports
- [ ] Remove old class code
- [ ] Update related tests

## Example: Git Tool Migration Strategy

**git_tool.py** is complex with many operations. Recommended approach:

**Before:** 1 class with multiple operations
```python
class GitTool(BaseTool):
    async def execute(self, context, **kwargs):
        operation = kwargs.get("operation")
        if operation == "commit":
            return self._commit(...)
        elif operation == "push":
            return self._push(...)
        # ... many more operations
```

**After:** Multiple focused `@tool` functions
```python
@tool
async def git_commit(message: str, files: List[str] = None) -> Dict[str, Any]:
    """Commit changes to git repository."""
    ...

@tool
async def git_push(remote: str = "origin", branch: str = "main") -> Dict[str, Any]:
    """Push commits to remote repository."""
    ...

@tool
async def git_status() -> Dict[str, Any]:
    """Get current git status."""
    ...
```

## Testing After Migration

After migrating each tool:

1. **Import Test**: `python3 -c "from victor.tools.toolname import *"`
2. **Type Check**: `mypy victor/tools/toolname.py`
3. **Run Tests**: `pytest tests/unit/test_toolname.py`
4. **Manual Test**: Test the tool in a simple script

## Impact on Coverage

Migrating tools will:
- Make tools easier to test (functions vs classes)
- Allow more focused unit tests (one test per @tool function)
- Improve coverage as decorator pattern is simpler to cover

Estimated coverage improvement after full migration: **+10-15%**

## Next Steps

1. ✅ Bash tool migrated
2. Migrate cache_tool.py (high-priority, small)
3. Migrate http_tool.py (high-priority, commonly used)
4. Migrate remaining 15 tools systematically
5. Update tool registry to handle both patterns during transition
6. Create integration tests for migrated tools
7. Update documentation

## Notes

- **Backward Compatibility**: During migration, support both patterns
- **Tool Registry**: May need updates to handle @tool decorated functions
- **Agent Integration**: Verify agents can call both old and new patterns
- **Documentation**: Update all examples to use new pattern
