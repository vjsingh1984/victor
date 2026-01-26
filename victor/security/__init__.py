# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified Security Module for Victor.

.. deprecated:: 0.6.0
    This module is being reorganized. Please migrate to:

    - **Security Infrastructure** (RBAC, audit, authorization):
      Use ``victor.core.security`` instead of ``victor.security.auth``,
      ``victor.security.audit``, and ``victor.security.authorization_enhanced``.

    - **Security Analysis Tools** (scanners, CVE databases, patterns):
      Use ``victor.security_analysis`` instead of ``victor.security.scanner``,
      ``victor.security.cve_database``, and ``victor.security.safety``.

    This module will continue to work but emits deprecation warnings.
    The deprecated paths will be removed in v1.0.0.

Migration Guide:
    Old (deprecated):
        from victor.security.auth import RBACManager, Permission
        from victor.security.audit import AuditManager
        from victor.security.safety import detect_secrets

    New (recommended):
        from victor.core.security.auth import RBACManager, Permission
        from victor.core.security.audit import AuditManager
        from victor.security_analysis.patterns import detect_secrets

This module provides comprehensive security capabilities including:

Submodules:
- victor.security.auth: Authentication and authorization (RBAC)
  -> Migrate to: victor.core.security.auth
- victor.security.safety: Safety patterns (secrets, PII, code patterns)
  -> Migrate to: victor.security_analysis.patterns
- victor.security.audit: Compliance and audit logging
  -> Migrate to: victor.core.security.audit

Also includes:
- Vulnerability scanning and CVE database integration
  -> Migrate to: victor.security_analysis.tools
- Security policy enforcement and reporting

Example usage:
    from victor.security import get_security_manager, SecurityPolicy
    from victor.security.auth import RBACManager, Permission
    from victor.security.safety import detect_secrets, CodePatternScanner
    from victor.security.audit import AuditManager
    from pathlib import Path
    import asyncio

    async def scan_project():
        # Vulnerability scanning
        manager = get_security_manager(project_root=Path("."))
        manager.policy = SecurityPolicy(
            fail_on_critical=True,
            fail_on_high=True,
            max_medium=10,
        )

        # Scan project
        result = await manager.scan()

        # Generate report
        print(manager.generate_report(format="text"))

        # Check policy
        passed, failures = manager.check_policy()
        if not passed:
            for failure in failures:
                print(f"FAIL: {failure}")

        # Secret detection
        secrets = detect_secrets(code_content)

        # RBAC
        rbac = RBACManager()
        if rbac.check_permission("user", Permission.WRITE):
            # Allow write operation
            pass

        # Audit logging
        audit = AuditManager.get_instance()
        await audit.log_file_operation("read", "/path/to/file")

    asyncio.run(scan_project())
"""

import warnings
from typing import Any

# Deprecation message templates
_DEPRECATION_MSG_INFRASTRUCTURE = (
    "victor.security.{name} is deprecated and will be removed in v1.0.0. "
    "Use victor.core.security.{name} instead for security infrastructure."
)
_DEPRECATION_MSG_ANALYSIS = (
    "victor.security.{name} is deprecated and will be removed in v1.0.0. "
    "Use victor.security_analysis.{dest} instead for security analysis tools."
)

# CVE/Vulnerability scanning (existing functionality)
from victor.security.protocol import (
    CVE,
    CVSSMetrics,
    Dependency,
    SecurityPolicy,
    SecurityScanResult,
    Severity,
    Vulnerability,
    VulnerabilityStatus,
)
from victor.security.cve_database import (
    BaseCVEDatabase,
    CachingCVEDatabase,
    CVEDatabase,
    LocalCVECache,
    OfflineCVEDatabase,
    OSVDatabase,
    get_cve_database,
)
from victor.security.scanner import (
    BaseDependencyParser,
    DependencyParser,
    GoDependencyParser,
    NodeDependencyParser,
    PythonDependencyParser,
    RustDependencyParser,
    SecurityScanner,
    get_scanner,
)
from victor.security.manager import (
    SecurityManager,
    get_security_manager,
    reset_security_manager,
)

# Penetration Testing (NEW)
from victor.security.penetration_testing import (
    SecurityTestSuite,
    SeverityLevel,
    AttackType,
    Vulnerability as PenTestVulnerability,
    SecurityReport,
    ComprehensiveSecurityReport,
    run_security_tests,
)

# Enhanced Authorization (NEW)
from victor.security.authorization_enhanced import (
    EnhancedAuthorizer,
    Permission,
    Role,
    User,
    Policy,
    PolicyEffect,
    ResourceType,
    ActionType,
    AuthorizationDecision,
    get_enhanced_authorizer,
    set_enhanced_authorizer,
)

# Legacy alias for backwards compatibility
EnhancedUser = User

# Submodules exposed via lazy imports to avoid circular import issues
# Users can still do: from victor.security import auth, safety, audit
# The submodules are imported on first access via __getattr__

__all__ = [
    # Submodules (lazy loaded)
    "auth",
    "safety",
    "audit",
    # Protocol types
    "CVE",
    "CVSSMetrics",
    "Dependency",
    "SecurityPolicy",
    "SecurityScanResult",
    "Severity",
    "Vulnerability",
    "VulnerabilityStatus",
    # CVE database
    "BaseCVEDatabase",
    "CachingCVEDatabase",
    "CVEDatabase",
    "LocalCVECache",
    "OfflineCVEDatabase",
    "OSVDatabase",
    "get_cve_database",
    # Scanners
    "BaseDependencyParser",
    "DependencyParser",
    "GoDependencyParser",
    "NodeDependencyParser",
    "PythonDependencyParser",
    "RustDependencyParser",
    "SecurityScanner",
    "get_scanner",
    # Manager
    "SecurityManager",
    "get_security_manager",
    "reset_security_manager",
    # Penetration Testing (NEW)
    "SecurityTestSuite",
    "SeverityLevel",
    "AttackType",
    "PenTestVulnerability",
    "SecurityReport",
    "ComprehensiveSecurityReport",
    "run_security_tests",
    # Enhanced Authorization (NEW)
    "EnhancedAuthorizer",
    "Permission",
    "Role",
    "User",
    "EnhancedUser",  # Legacy alias
    "Policy",
    "PolicyEffect",
    "ResourceType",
    "ActionType",
    "AuthorizationDecision",
    "get_enhanced_authorizer",
    "set_enhanced_authorizer",
]


# Track which submodules have already warned to avoid duplicate warnings
_warned_submodules: set = set()
# Cache for loaded submodules to avoid re-importing
_submodule_cache: dict = {}


def __getattr__(name: str) -> Any:
    """Lazy import for submodules with deprecation warnings."""
    # Return from cache if already loaded
    if name in _submodule_cache:
        return _submodule_cache[name]

    if name == "auth":
        if "auth" not in _warned_submodules:
            _warned_submodules.add("auth")
            warnings.warn(
                _DEPRECATION_MSG_INFRASTRUCTURE.format(name="auth"),
                DeprecationWarning,
                stacklevel=2,
            )
        import importlib
        _submodule_cache["auth"] = importlib.import_module("victor.security.auth")
        return _submodule_cache["auth"]
    elif name == "safety":
        if "safety" not in _warned_submodules:
            _warned_submodules.add("safety")
            warnings.warn(
                _DEPRECATION_MSG_ANALYSIS.format(name="safety", dest="patterns"),
                DeprecationWarning,
                stacklevel=2,
            )
        import importlib
        _submodule_cache["safety"] = importlib.import_module("victor.security.safety")
        return _submodule_cache["safety"]
    elif name == "audit":
        if "audit" not in _warned_submodules:
            _warned_submodules.add("audit")
            warnings.warn(
                _DEPRECATION_MSG_INFRASTRUCTURE.format(name="audit"),
                DeprecationWarning,
                stacklevel=2,
            )
        import importlib
        _submodule_cache["audit"] = importlib.import_module("victor.security.audit")
        return _submodule_cache["audit"]
    elif name == "penetration_testing":
        if "penetration_testing" not in _warned_submodules:
            _warned_submodules.add("penetration_testing")
            warnings.warn(
                _DEPRECATION_MSG_ANALYSIS.format(name="penetration_testing", dest="tools"),
                DeprecationWarning,
                stacklevel=2,
            )
        import importlib
        _submodule_cache["penetration_testing"] = importlib.import_module(
            "victor.security.penetration_testing"
        )
        return _submodule_cache["penetration_testing"]
    elif name == "authorization_enhanced":
        if "authorization_enhanced" not in _warned_submodules:
            _warned_submodules.add("authorization_enhanced")
            warnings.warn(
                _DEPRECATION_MSG_INFRASTRUCTURE.format(name="authorization_enhanced"),
                DeprecationWarning,
                stacklevel=2,
            )
        import importlib
        _submodule_cache["authorization_enhanced"] = importlib.import_module(
            "victor.security.authorization_enhanced"
        )
        return _submodule_cache["authorization_enhanced"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
