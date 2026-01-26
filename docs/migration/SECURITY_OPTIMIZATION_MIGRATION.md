# Security and Optimization Module Migration Guide

## Overview

This guide helps you migrate from deprecated module locations to their new canonical locations as part of Victor's architectural refactoring for better separation of concerns, SOLID compliance, and maintainability.

**Migration Timeline**:
- **v0.6.0** (Current): New canonical structure + deprecation warnings
- **v0.7.0**: Optional auto-migration script
- **v1.0.0**: Removal of deprecated paths

---

## Table of Contents

1. [Security Module Migration](#security-module-migration)
2. [Optimization Module Migration](#optimization-module-migration)
3. [Quick Reference](#quick-reference)
4. [Testing Your Migration](#testing-your-migration)

---

## Security Module Migration

### Architecture Overview

The security module has been reorganized into two distinct areas:

**Framework Infrastructure** (`victor/core/security/`):
- Cross-cutting security services used by all verticals
- RBAC, audit logging, authorization
- Safety patterns (secrets, PII, code safety)
- NOT domain-specific

**Security Analysis Vertical** (`victor/security_analysis/`):
- Domain-specific security analysis orchestration
- Vulnerability scanning, CVE databases
- Penetration testing tools

### Migration Map

#### Core Security (Framework Infrastructure)

| Old (Deprecated) | New (Canonical) | Notes |
|------------------|-----------------|-------|
| `from victor.security.auth import RBACManager` | `from victor.core.security.auth import RBACManager` | RBAC functionality |
| `from victor.security.audit import AuditManager` | `from victor.core.security.audit import AuditManager` | Audit logging |
| `from victor.security.authorization_enhanced import EnhancedAuthorizer` | `from victor.core.security.authorization import EnhancedAuthorizer` | RBAC + ABAC |
| `from victor.security.protocol import CVE, Severity` | `from victor.core.security.protocol import CVE, CVESeverity` | CVE/vulnerability types |
| `from victor.security.safety import SafetyPattern` | `from victor.core.security.patterns import SafetyPattern` | Safety patterns base |
| `from victor.security.safety.secrets import detect_secrets` | `from victor.core.security.patterns.secrets import detect_secrets` | Secret detection |
| `from victor.security.safety.pii import PIIScanner` | `from victor.core.security.patterns.pii import PIIScanner` | PII detection |
| `from victor.security.safety.code_patterns import CodePatternScanner` | `from victor.core.security.patterns.code_patterns import CodePatternScanner` | Code safety |
| `from victor.security.safety.infrastructure import InfrastructureScanner` | `from victor.core.security.patterns.infrastructure import InfrastructureScanner` | IaC security |
| `from victor.security.safety.source_credibility import SourceCredibilityScanner` | `from victor.core.security.patterns.source_credibility import SourceCredibilityScanner` | Source validation |

#### Security Analysis Tools (Vertical)

| Old (Deprecated) | New (Canonical) | Notes |
|------------------|-----------------|-------|
| `from victor.security.scanner import SecurityScanner` | `from victor.security_analysis.tools import SecurityScanner` | Dependency vulnerability scanning |
| `from victor.security.cve_database import CVEDatabase` | `from victor.security_analysis.tools import CVEDatabase` | CVE database integration |
| `from victor.security.manager import SecurityManager` | `from victor.security_analysis.tools import SecurityManager` | Security orchestration |
| `from victor.security.penetration_testing import SecurityTestSuite` | `from victor.security_analysis.tools import SecurityTestSuite` | Penetration testing |

### Code Examples

#### Before (Deprecated)

```python
# Old imports from deprecated locations
from victor.security.auth import RBACManager, Permission, Role
from victor.security.audit import AuditManager
from victor.security.protocol import CVE, CVESeverity, SecurityScanResult
from victor.security.safety.secrets import detect_secrets, CREDENTIAL_PATTERNS
from victor.security.safety.pii import PIIScanner
from victor.security.scanner import SecurityScanner
from victor.security.cve_database import CVEDatabase
from victor.security.manager import SecurityManager
from victor.security.penetration_testing import SecurityTestSuite
```

#### After (Canonical)

```python
# New imports from canonical locations
from victor.core.security.auth import RBACManager, Permission, Role
from victor.core.security.audit import AuditManager
from victor.core.security.protocol import CVE, CVESeverity, SecurityScanResult
from victor.core.security.patterns.secrets import detect_secrets, CREDENTIAL_PATTERNS
from victor.core.security.patterns.pii import PIIScanner
from victor.security_analysis.tools import SecurityScanner, CVEDatabase, SecurityManager
from victor.security_analysis.tools.penetration_testing import SecurityTestSuite

# Or use unified imports where available
from victor.core.security import (
    RBACManager,
    AuditManager,
    CVE,
    CVESeverity,
    detect_secrets,
    PIIScanner,
)
from victor.security_analysis import (
    SecurityScanner,
    CVEDatabase,
    SecurityManager,
)
```

### Safety Patterns in Verticals

All verticals now import safety patterns from the core location:

```python
# In victor/coding/safety.py
from victor.core.security.patterns.code_patterns import CodePatternScanner, GIT_PATTERNS

# In victor/devops/safety.py
from victor.core.security.patterns.infrastructure import InfrastructureScanner, KUBERNETES_PATTERNS

# In victor/dataanalysis/safety.py
from victor.core.security.patterns.pii import PIIScanner

# In victor/research/safety.py
from victor.core.security.patterns.source_credibility import SourceCredibilityScanner

# In victor/rag/safety.py
from victor.core.security.patterns.pii import PIIScanner
```

---

## Optimization Module Migration

### Architecture Overview

The optimization module has been unified under a single `victor/optimization/` namespace with three submodules:

- **victor.optimization.workflow**: Workflow profiling and optimization
- **victor.optimization.runtime**: Lazy loading, parallel execution, memory optimization
- **victor.optimization.core**: Hot path utilities (JSON, caching, monitoring)

### Migration Map

| Old (Deprecated) | New (Canonical) | Notes |
|------------------|-----------------|-------|
| `from victor.optimizations import LazyComponentLoader` | `from victor.optimization.runtime import LazyComponentLoader` | Lazy loading |
| `from victor.optimizations import AdaptiveParallelExecutor` | `from victor.optimization.runtime import AdaptiveParallelExecutor` | Parallel execution |
| `from victor.core.optimizations import json_dumps, json_loads` | `from victor.optimization.core import json_dumps, json_loads` | Fast JSON |
| `from victor.core.optimizations import LazyImport` | `from victor.optimization.core import LazyImport` | Lazy imports |
| `from victor.optimization.workflow import WorkflowOptimizer` | `from victor.optimization import WorkflowOptimizer` | **Already canonical** |

### Code Examples

#### Before (Deprecated)

```python
# Old imports from deprecated locations
from victor.optimizations import (
    LazyComponentLoader,
    AdaptiveParallelExecutor,
    create_adaptive_executor,
)
from victor.core.optimizations import (
    json_dumps,
    json_loads,
    LazyImport,
    PerformanceMonitor,
)
```

#### After (Canonical)

```python
# Option 1: Import from unified facade
from victor.optimization import (
    LazyComponentLoader,
    AdaptiveParallelExecutor,
    json_dumps,
    json_loads,
    LazyImport,
    PerformanceMonitor,
    WorkflowOptimizer,
)

# Option 2: Import from specific submodules
from victor.optimization.runtime import (
    LazyComponentLoader,
    AdaptiveParallelExecutor,
    create_adaptive_executor,
)
from victor.optimization.core import (
    json_dumps,
    json_loads,
    LazyImport,
    PerformanceMonitor,
)
from victor.optimization.workflow import WorkflowOptimizer
```

---

## Quick Reference

### All Canonical Imports

```python
# =============================================================================
# Security (Framework Infrastructure)
# =============================================================================
from victor.core.security.auth import RBACManager, Permission, Role
from victor.core.security.audit import AuditManager, AuditEvent
from victor.core.security.authorization import EnhancedAuthorizer
from victor.core.security.protocol import CVE, CVESeverity, SecurityScanResult

# Safety patterns (cross-cutting)
from victor.core.security.patterns import SafetyPattern, ISafetyScanner, SafetyRegistry
from victor.core.security.patterns.secrets import detect_secrets, CREDENTIAL_PATTERNS
from victor.core.security.patterns.pii import PIIScanner
from victor.core.security.patterns.code_patterns import CodePatternScanner
from victor.core.security.patterns.infrastructure import InfrastructureScanner
from victor.core.security.patterns.source_credibility import SourceCredibilityScanner

# =============================================================================
# Security Analysis (Vertical)
# =============================================================================
from victor.security_analysis import SecurityAnalysisAssistant
from victor.security_analysis.tools import (
    SecurityScanner,      # Dependency vulnerability scanning
    CVEDatabase,         # CVE database integration
    SecurityManager,     # High-level security orchestration
    SecurityTestSuite,   # Penetration testing
)

# =============================================================================
# Optimization (Unified)
# =============================================================================
from victor.optimization import (
    # Workflow optimization
    WorkflowOptimizer,
    WorkflowProfiler,
    # Runtime optimization
    LazyComponentLoader,
    AdaptiveParallelExecutor,
    # Hot path optimization
    json_dumps,
    json_loads,
    LazyImport,
)
```

### Import Style Guidelines

**DO**:
- ✅ Import from `victor.core.security.*` for framework security services
- ✅ Import from `victor.security_analysis.tools.*` for security analysis tools
- ✅ Import from `victor.optimization.*` for all optimization needs
- ✅ Use specific submodule imports when you only need one area (e.g., `victor.optimization.runtime`)

**DON'T**:
- ❌ Import from `victor.security.*` (deprecated, will be removed in v1.0.0)
- ❌ Import from `victor.optimizations.*` (deprecated, will be removed in v1.0.0)
- ❌ Import from `victor.core.optimizations.*` (deprecated, will be removed in v1.0.0)

---

## Testing Your Migration

### Enable Deprecation Warnings

Run your code with deprecation warnings enabled to catch any remaining deprecated imports:

```bash
# Python: Show all deprecation warnings
python -W default::DeprecationWarning your_script.py

# pytest: Show deprecation warnings
pytest -W default::DeprecationWarning tests/

# Or make them errors to ensure all are migrated
python -W error::DeprecationWarning your_script.py
pytest -W error::DeprecationWarning tests/
```

### Verify Imports

```python
#!/usr/bin/env python3
"""Verify all imports use canonical locations."""

import warnings
import sys

# Enable deprecation warnings as errors
warnings.filterwarnings('error', category=DeprecationWarning)

try:
    # Test all canonical imports
    from victor.core.security.auth import RBACManager
    from victor.core.security.audit import AuditManager
    from victor.core.security.protocol import CVE
    from victor.core.security.patterns import SafetyPattern
    from victor.security_analysis.tools import SecurityScanner
    from victor.optimization import WorkflowOptimizer, LazyComponentLoader

    print("✅ All canonical imports successful")
    sys.exit(0)

except DeprecationWarning as e:
    print(f"❌ Deprecated import detected: {e}")
    sys.exit(1)
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
```

### Check Your Codebase

```bash
# Scan for deprecated security imports
grep -r "from victor\.security\." victor/ --include="*.py" | \
  grep -v "victor/security/__init__.py" | \
  grep -v "__pycache__"

# Scan for deprecated optimization imports
grep -r "from victor\.optimizations import\|from victor\.core\.optimizations import" \
  victor/ --include="*.py" | \
  grep -v "__init__.py" | \
  grep -v "__pycache__"
```

---

## Common Migration Scenarios

### Scenario 1: Updating a Tool Using Security Patterns

**Before:**
```python
from victor.security.safety.secrets import detect_secrets
from victor.security.safety.pii import PIIScanner

class MyTool(BaseTool):
    def execute(self, content: str):
        secrets = detect_secrets(content)
        pii = PIIScanner().scan(content)
```

**After:**
```python
from victor.core.security.patterns.secrets import detect_secrets
from victor.core.security.patterns.pii import PIIScanner

class MyTool(BaseTool):
    def execute(self, content: str):
        secrets = detect_secrets(content)
        pii = PIIScanner().scan(content)
```

### Scenario 2: Updating a Vertical's Safety Module

**Before (victor/coding/safety.py):**
```python
from victor.security.safety.code_patterns import CodePatternScanner
from victor.security.safety.types import SafetyPattern
```

**After (victor/coding/safety.py):**
```python
from victor.core.security.patterns.code_patterns import CodePatternScanner
from victor.core.security.patterns.types import SafetyPattern
```

### Scenario 3: Using Optimization in Application Code

**Before:**
```python
from victor.optimizations import LazyComponentLoader, AdaptiveParallelExecutor
from victor.core.optimizations import json_dumps, json_loads
```

**After:**
```python
# Option 1: Unified import (recommended for most cases)
from victor.optimization import (
    LazyComponentLoader,
    AdaptiveParallelExecutor,
    json_dumps,
    json_loads,
)

# Option 2: Specific submodule imports (if you only need one area)
from victor.optimization.runtime import LazyComponentLoader, AdaptiveParallelExecutor
from victor.optimization.core import json_dumps, json_loads
```

---

## Troubleshooting

### Issue: ImportError after migration

**Problem:**
```python
ImportError: cannot import name 'RBACManager' from 'victor.security.auth'
```

**Solution:**
```python
# Use the canonical import
from victor.core.security.auth import RBACManager
```

### Issue: Deprecation warnings in tests

**Problem:** Tests still import from deprecated locations

**Solution:**
1. Find all deprecated imports:
   ```bash
   grep -r "from victor\.security\." tests/ --include="*.py"
   grep -r "from victor\.optimizations" tests/ --include="*.py"
   ```
2. Update to canonical imports
3. Re-run tests

### Issue: Circular import errors

**Problem:** Moving imports causes circular dependencies

**Solution:** Use `TYPE_CHECKING` for type hints:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from victor.core.security.protocol import SecurityScanResult

def my_function(scan_result: "SecurityScanResult"):  # Use string annotation
    ...
```

---

## Need Help?

- **Documentation**: See `CLAUDE.md` for canonical import reference
- **Architecture**: See `docs/architecture/REFACTORING_OVERVIEW.md`
- **Best Practices**: See `docs/architecture/BEST_PRACTICES.md`
- **Issues**: Report migration issues on GitHub

---

## Summary

| Module | Deprecated Location | Canonical Location | Status |
|--------|---------------------|-------------------|--------|
| RBAC | `victor.security.auth` | `victor.core.security.auth` | ✅ Migrated |
| Audit | `victor.security.audit` | `victor.core.security.audit` | ✅ Migrated |
| Authorization | `victor.security.authorization_enhanced` | `victor.core.security.authorization` | ✅ Migrated |
| Protocol | `victor.security.protocol` | `victor.core.security.protocol` | ✅ Migrated |
| Safety Patterns | `victor.security.safety.*` | `victor.core.security.patterns.*` | ✅ Migrated |
| Scanner | `victor.security.scanner` | `victor.security_analysis.tools.scanner` | ✅ Migrated |
| CVE Database | `victor.security.cve_database` | `victor.security_analysis.tools.cve_database` | ✅ Migrated |
| Manager | `victor.security.manager` | `victor.security_analysis.tools.manager` | ✅ Migrated |
| Pen Testing | `victor.security.penetration_testing` | `victor.security_analysis.tools.penetration_testing` | ✅ Migrated |
| Runtime Opt | `victor.optimizations` | `victor.optimization.runtime` | ✅ Migrated |
| Core Opt | `victor.core.optimizations` | `victor.optimization.core` | ✅ Migrated |
| Workflow Opt | `victor.optimization.workflow` | `victor.optimization.workflow` | ✅ Already canonical |

**Migration Complete**: v0.6.0 ✅
**Removal of Deprecated Paths**: v1.0.0 ⏳
