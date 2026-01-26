# Security and Optimization Module Migration Guide

This guide covers the migration of security and optimization modules introduced in Victor v0.6.0.

## Overview

Victor v0.6.0 introduces a significant reorganization of security and optimization modules to improve:
- **Separation of concerns**: Security infrastructure vs. security analysis tools
- **SOLID compliance**: Better Single Responsibility Principle adherence
- **Developer clarity**: Clear naming and module organization
- **Maintainability**: Easier to find and modify related code

## Timeline

| Version | State |
|---------|-------|
| v0.6.0 | New structure available, deprecation warnings for old paths |
| v0.7.0 | Optional auto-migration script available |
| v1.0.0 | Deprecated paths removed |

## Security Module Changes

### Before: Mixed Concerns in `victor/security/`

The old structure mixed security infrastructure with security analysis tools:

```
victor/security/
├── auth/rbac.py          # Infrastructure
├── audit/                # Infrastructure
├── authorization_enhanced.py  # Infrastructure
├── safety/               # Analysis patterns
├── scanner.py            # Analysis tools
├── cve_database.py       # Analysis tools
└── penetration_testing.py # Analysis tools
```

### After: Clear Separation

**Security Infrastructure** (`victor/core/security/`):
- RBAC, audit logging, authorization
- Cross-cutting concerns used throughout the framework

**Security Analysis Vertical** (`victor/security_analysis/`):
- Vulnerability scanning, CVE databases
- Safety patterns (secrets, PII, code patterns)
- Penetration testing tools

## Import Migration Table

### Security Infrastructure

| Old Import (Deprecated) | New Import |
|------------------------|------------|
| `from victor.security.auth import RBACManager, Permission` | `from victor.core.security.auth import RBACManager, Permission` |
| `from victor.security.audit import AuditManager` | `from victor.core.security.audit import AuditManager` |
| `from victor.security.audit import AuditEvent, AuditEventType` | `from victor.core.security.audit import AuditEvent, AuditEventType` |
| `from victor.security.authorization_enhanced import EnhancedAuthorizer` | `from victor.core.security.authorization import EnhancedAuthorizer` |

### Security Analysis (Patterns)

| Old Import (Deprecated) | New Import |
|------------------------|------------|
| `from victor.security.safety import detect_secrets` | `from victor.security_analysis.patterns import detect_secrets` |
| `from victor.security.safety import CodePatternScanner` | `from victor.security_analysis.patterns import CodePatternScanner` |
| `from victor.security.safety.pii import detect_pii_columns` | `from victor.security_analysis.patterns.pii import detect_pii_columns` |
| `from victor.security.safety.secrets import SecretScanner` | `from victor.security_analysis.patterns.secrets import SecretScanner` |
| `from victor.security.safety.infrastructure import InfrastructureScanner` | `from victor.security_analysis.patterns.infrastructure import InfrastructureScanner` |

### Security Analysis (Tools)

| Old Import (Deprecated) | New Import |
|------------------------|------------|
| `from victor.security import SecurityScanner, get_scanner` | `from victor.security_analysis.tools import SecurityScanner, get_scanner` |
| `from victor.security import SecurityManager, get_security_manager` | `from victor.security_analysis.tools import SecurityManager, get_security_manager` |
| `from victor.security.cve_database import OSVDatabase` | `from victor.security_analysis.tools import OSVDatabase` |
| `from victor.security.penetration_testing import SecurityTestSuite` | `from victor.security_analysis.tools import SecurityTestSuite` |

## Optimization Module Changes

### Before: Three Confusingly Named Modules

```
victor/optimization/          # Workflow optimization
victor/optimizations/         # Runtime performance (plural!)
victor/core/optimizations/    # Hot path utilities
```

### After: Unified Structure

```
victor/optimization/
├── __init__.py          # Unified facade
├── workflow/            # Workflow profiling and optimization
├── runtime/             # Lazy loading, parallel execution
└── core/                # Hot path utilities (JSON, lazy imports)
```

## Optimization Import Migration Table

### Workflow Optimization (unchanged)

```python
# These imports work the same as before
from victor.optimization import WorkflowOptimizer, WorkflowProfiler
```

### Runtime Optimization

| Old Import (Deprecated) | New Import |
|------------------------|------------|
| `from victor.optimizations import LazyComponentLoader` | `from victor.optimization.runtime import LazyComponentLoader` |
| `from victor.optimizations import AdaptiveParallelExecutor` | `from victor.optimization.runtime import AdaptiveParallelExecutor` |
| `from victor.optimizations import create_adaptive_executor` | `from victor.optimization.runtime import create_adaptive_executor` |

### Hot Path Utilities

| Old Import (Deprecated) | New Import |
|------------------------|------------|
| `from victor.core.optimizations import json_dumps, json_loads` | `from victor.optimization.core import json_dumps, json_loads` |
| `from victor.core.optimizations import LazyImport` | `from victor.optimization.core import LazyImport` |
| `from victor.core.optimizations import timed, retry` | `from victor.optimization.core import timed, retry` |

### Unified Import (New)

You can now import from the unified facade:

```python
# All optimization capabilities from one place
from victor.optimization import (
    # Workflow
    WorkflowOptimizer,
    WorkflowProfiler,
    # Runtime
    LazyComponentLoader,
    AdaptiveParallelExecutor,
    # Hot path
    json_dumps,
    json_loads,
    LazyImport,
)
```

## Code Examples

### Before (Deprecated)

```python
# Security - mixed imports
from victor.security.auth import RBACManager, Permission
from victor.security.audit import AuditManager
from victor.security.safety import detect_secrets, CodePatternScanner
from victor.security import SecurityScanner, get_security_manager

# Optimization - confusing plurals
from victor.optimizations import LazyComponentLoader
from victor.core.optimizations import json_dumps
```

### After (Recommended)

```python
# Security Infrastructure
from victor.core.security.auth import RBACManager, Permission
from victor.core.security.audit import AuditManager

# Security Analysis (vertical)
from victor.security_analysis.patterns import detect_secrets, CodePatternScanner
from victor.security_analysis.tools import SecurityScanner, get_security_manager

# Or use the vertical assistant
from victor.security_analysis import SecurityAnalysisAssistant

# Optimization (unified)
from victor.optimization import (
    LazyComponentLoader,    # from runtime/
    json_dumps,             # from core/
    WorkflowOptimizer,      # from workflow/
)
```

## Suppressing Deprecation Warnings

If you need to suppress warnings temporarily during migration:

```python
import warnings

# Suppress all deprecation warnings (not recommended for production)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Or suppress specific modules
warnings.filterwarnings(
    "ignore",
    message="victor.security.*",
    category=DeprecationWarning,
)
```

## Testing Deprecation Warnings

To ensure your code doesn't use deprecated imports:

```bash
# Run with warnings as errors
python -W error::DeprecationWarning your_script.py

# Or in pytest
pytest -W error::DeprecationWarning
```

## Vertical Registration

The new `security_analysis` vertical is registered via entry points:

```toml
# pyproject.toml
[project.entry-points."victor.verticals"]
security_analysis = "victor.security_analysis:SecurityAnalysisAssistant"
```

You can use it like any other vertical:

```python
from victor.security_analysis import SecurityAnalysisAssistant

# Get tools for security analysis
tools = SecurityAnalysisAssistant.get_tools()

# Get system prompt
prompt = SecurityAnalysisAssistant.get_system_prompt()
```

## Questions?

If you encounter issues during migration:

1. Check the deprecation warning message for the recommended import path
2. Review this guide for the mapping tables
3. File an issue at https://github.com/anthropics/victor/issues
