# Victor Package Migration Guide

This document describes the restructuring of Victor into independent packages and provides migration guidance.

## Package Structure

Victor is now organized as a monorepo with three independent packages:

```
packages/
├── victor-core/      # Core framework (providers, tools, protocols)
├── victor-coding/    # Coding vertical (Tree-sitter, LSP, 25 tools)
└── victor-ai/        # Meta-package (backward compatibility)
```

### Package Descriptions

| Package | Version | Description |
|---------|---------|-------------|
| `victor-core` | 0.3.0 | Multi-provider LLM orchestration, base tools, protocols |
| `victor-coding` | 0.1.0 | Coding vertical with Tree-sitter, LSP, code analysis |
| `victor-ai` | 0.3.0 | Meta-package that installs both core + coding |

## Installation Options

### Full Installation (Recommended)
```bash
pip install victor-ai
```
This installs everything, equivalent to the original `victor` package.

### Core Framework Only
```bash
pip install victor-core
```
For applications that only need the LLM orchestration without coding-specific features.

### Core + Coding
```bash
pip install victor-core victor-coding
```
Same as `victor-ai` but explicit about dependencies.

## Import Changes

### New Import Paths

The coding-specific modules have moved to `victor_coding`:

| Old Import | New Import |
|------------|------------|
| `from victor.codebase import ...` | `from victor_coding.codebase import ...` |
| `from victor.lsp import ...` | `from victor_coding.lsp import ...` |
| `from victor.languages import ...` | `from victor_coding.languages import ...` |
| `from victor.editing import ...` | `from victor_coding.editing import ...` |
| `from victor.testgen import ...` | `from victor_coding.testgen import ...` |
| `from victor.refactor import ...` | `from victor_coding.refactor import ...` |
| `from victor.review import ...` | `from victor_coding.review import ...` |
| `from victor.security import ...` | `from victor_coding.security import ...` |
| `from victor.docgen import ...` | `from victor_coding.docgen import ...` |
| `from victor.completion import ...` | `from victor_coding.completion import ...` |
| `from victor.deps import ...` | `from victor_coding.deps import ...` |
| `from victor.coverage import ...` | `from victor_coding.coverage import ...` |

### Core Imports (Unchanged)

These imports remain the same:

```python
from victor.config.settings import Settings, get_settings
from victor.core.container import ServiceContainer
from victor.verticals.base import VerticalBase
from victor.providers.registry import ProviderRegistry
from victor.tools.base import BaseTool
```

### Backward Compatibility

Old imports will continue to work with deprecation warnings until version 0.5.0:

```python
# This will work but emit a DeprecationWarning
from victor.codebase import CodebaseIndexer

# Recommended: Use the new import
from victor_coding.codebase import CodebaseIndexer
```

## CodingVertical Access

The `CodingVertical` class is now in `victor_coding`:

```python
# Old way (deprecated)
from victor.verticals.coding import CodingAssistant

# New way
from victor_coding import CodingVertical
from victor_coding.vertical import CodingVertical  # explicit
```

## Plugin Discovery

Verticals and tools are now discovered via entry points:

### Entry Points (pyproject.toml)

```toml
[project.entry-points."victor.verticals"]
coding = "victor_coding:CodingVertical"

[project.entry-points."victor.tools"]
code_search = "victor_coding.tools:CodeSearchTool"
semantic_code_search = "victor_coding.tools:SemanticCodeSearchTool"
# ... more tools
```

### Discovery API

```python
from victor.verticals.vertical_loader import (
    discover_vertical_plugins,
    discover_tool_plugins,
)

# Discover installed verticals
verticals = discover_vertical_plugins()
# {'coding': <class 'victor_coding.CodingVertical'>}

# Discover installed tools
tools = discover_tool_plugins()
# {'code_search': <class 'victor_coding.tools.CodeSearchTool'>, ...}
```

## VS Code Extension

The VS Code extension is fully compatible with the new package structure. It connects to the Victor server via HTTP API, which is unaffected by the internal package reorganization.

No changes required for VS Code extension users.

## Docker

### Using the Meta-Package

```dockerfile
FROM python:3.11-slim
RUN pip install victor-ai
```

### Minimal Core-Only Image

```dockerfile
FROM python:3.11-slim
RUN pip install victor-core
# No coding tools, no Tree-sitter, smaller image
```

## Migration Checklist

- [ ] Update imports from `victor.codebase` to `victor_coding.codebase`
- [ ] Update imports from `victor.lsp` to `victor_coding.lsp`
- [ ] Update any `CodingAssistant` imports to `CodingVertical`
- [ ] Run tests to catch any missed imports
- [ ] Update Dockerfile if using specific modules

## Timeline

| Version | Date | Changes |
|---------|------|---------|
| 0.3.0 | Current | Deprecation warnings for old imports |
| 0.4.0 | Future | Last version supporting old imports |
| 0.5.0 | Future | Remove backward compatibility shims |

## Questions?

If you encounter issues during migration, please file an issue at:
https://github.com/vijayksingh/victor/issues
