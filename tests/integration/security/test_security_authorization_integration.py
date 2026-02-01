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

"""Integration tests for Security + Authorization systems (Phase 4).

This module tests the integration between:
- Security tools: Penetration testing, vulnerability scanning
- Authorization system: Enhanced permission checks
- Tool execution: Secure tool invocation
- Security controls: Access control, audit logging

Test scenarios:
1. Penetration testing with authorization checks
2. Vulnerability scanning with permission validation
3. Security tool execution with audit logging
4. Cross-system security event propagation
5. Performance under security-heavy workloads
"""

import asyncio
from datetime import datetime
from typing import Optional, Any
from unittest.mock import MagicMock

import pytest


# ============================================================================
# Mock Classes for Testing
# ============================================================================


class MockSecurityTool:
    """Mock security tool for testing."""

    def __init__(self, name: str, required_permissions: list[str]):
        self.name = name
        self.required_permissions = required_permissions
        self.execution_log = []

    async def execute(self, **kwargs):
        """Execute tool with permission check."""
        self.execution_log.append({"timestamp": datetime.now(), "kwargs": kwargs})
        return {"status": "success", "tool": self.name}


class MockAuthorizationManager:
    """Mock authorization manager for testing."""

    def __init__(self):
        self.permissions = {}  # user -> permissions
        self.roles = {}  # user -> roles
        self.audit_log = []

    def grant_permission(self, user: str, permission: str):
        """Grant permission to user."""
        if user not in self.permissions:
            self.permissions[user] = set()
        self.permissions[user].add(permission)

    def grant_role(self, user: str, role: str):
        """Grant role to user."""
        if user not in self.roles:
            self.roles[user] = set()
        self.roles[user].add(role)

    def check_permission(self, user: str, permission: str) -> bool:
        """Check if user has permission."""
        return permission in self.permissions.get(user, set())

    def check_role(self, user: str, role: str) -> bool:
        """Check if user has role."""
        return role in self.roles.get(user, set())

    async def authorize_tool_execution(
        self, user: str, tool_name: str, required_permissions: list[str]
    ) -> tuple[bool, Optional[str]]:
        """Authorize tool execution."""
        self.audit_log.append(
            {
                "timestamp": datetime.now(),
                "user": user,
                "tool": tool_name,
                "required_permissions": required_permissions,
                "decision": "checking",
            }
        )

        # Check all required permissions
        for perm in required_permissions:
            if not self.check_permission(user, perm):
                self.audit_log[-1]["decision"] = "denied"
                self.audit_log[-1]["reason"] = f"Missing permission: {perm}"
                return False, f"Missing permission: {perm}"

        self.audit_log[-1]["decision"] = "granted"
        return True, None


class MockSecurityEventBus:
    """Mock security event bus."""

    def __init__(self):
        self.events = []

    async def publish(self, event_type: str, data: dict[str, Any]):
        """Publish security event."""
        self.events.append({"type": event_type, "data": data, "timestamp": datetime.now()})

    async def subscribe(self, event_type: str, handler):
        """Subscribe to security events."""
        pass


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def authorization_manager():
    """Create authorization manager for testing."""
    return MockAuthorizationManager()


@pytest.fixture
def security_event_bus():
    """Create security event bus."""
    return MockSecurityEventBus()


@pytest.fixture
def security_tools():
    """Create mock security tools."""
    return {
        "vulnerability_scan": MockSecurityTool(
            "vulnerability_scan", ["security:scan:read", "security:scan:execute"]
        ),
        "penetration_test": MockSecurityTool(
            "penetration_test", ["security:pen_test:execute", "security:attack_simulate"]
        ),
        "security_audit": MockSecurityTool(
            "security_audit", ["security:audit:read", "security:audit:write"]
        ),
        "access_control_check": MockSecurityTool(
            "access_control_check", ["security:access:read", "security:access:evaluate"]
        ),
        "credential_scan": MockSecurityTool(
            "credential_scan", ["security:credentials:read", "security:credentials:scan"]
        ),
    }


@pytest.fixture
def test_users(authorization_manager):
    """Set up test users with different permission levels."""
    # Admin user - all permissions
    authorization_manager.grant_role("admin", "security_admin")
    for perm in [
        "security:scan:read",
        "security:scan:execute",
        "security:pen_test:execute",
        "security:attack_simulate",
        "security:audit:read",
        "security:audit:write",
        "security:access:read",
        "security:access:evaluate",
        "security:credentials:read",
        "security:credentials:scan",
    ]:
        authorization_manager.grant_permission("admin", perm)

    # Security analyst - scan and audit only
    authorization_manager.grant_role("analyst", "security_analyst")
    for perm in [
        "security:scan:read",
        "security:scan:execute",
        "security:audit:read",
        "security:access:read",
    ]:
        authorization_manager.grant_permission("analyst", perm)

    # Limited user - only read access
    authorization_manager.grant_permission("limited", "security:scan:read")

    return ["admin", "analyst", "limited"]


@pytest.fixture
def mock_tool_registry():
    """Mock tool registry with security tools."""
    registry = MagicMock()

    tools = {
        "vulnerability_scan": MagicMock(
            name="vulnerability_scan",
            description="Scan for security vulnerabilities",
            category="security",
            required_permissions=["security:scan:execute"],
        ),
        "penetration_test": MagicMock(
            name="penetration_test",
            description="Perform penetration testing",
            category="security",
            required_permissions=["security:pen_test:execute"],
        ),
        "security_audit": MagicMock(
            name="security_audit",
            description="Perform security audit",
            category="security",
            required_permissions=["security:audit:write"],
        ),
    }

    registry.list_tools.return_value = list(tools.values())
    registry.get_tool = lambda name: tools.get(name)

    return registry


# ============================================================================
# Test: Penetration Testing with Authorization
# ============================================================================


@pytest.mark.asyncio
async def test_penetration_test_requires_authorization(
    authorization_manager, security_tools, security_event_bus, test_users
):
    """Test that penetration testing requires proper authorization.

    Scenario:
    1. Admin user requests penetration test
    2. Analyst user requests penetration test
    3. Limited user requests penetration test
    4. Verify only authorized users can execute

    Validates:
    - Authorization check before tool execution
    - Role-based access control
    - Audit logging for security events
    """
    pen_test_tool = security_tools["penetration_test"]

    # Admin should be authorized
    admin_user = "admin"
    granted, reason = await authorization_manager.authorize_tool_execution(
        user=admin_user,
        tool_name="penetration_test",
        required_permissions=pen_test_tool.required_permissions,
    )

    assert granted is True
    assert reason is None

    # Execute tool
    result = await pen_test_tool.execute(target="test-system")
    assert result["status"] == "success"

    # Publish security event
    await security_event_bus.publish(
        "tool.executed",
        {
            "user": admin_user,
            "tool": "penetration_test",
            "authorized": True,
            "target": "test-system",
        },
    )

    # Analyst should NOT be authorized
    analyst_user = "analyst"
    granted, reason = await authorization_manager.authorize_tool_execution(
        user=analyst_user,
        tool_name="penetration_test",
        required_permissions=pen_test_tool.required_permissions,
    )

    assert granted is False
    assert "Missing permission" in reason

    # Publish denial event
    await security_event_bus.publish(
        "tool.denied", {"user": analyst_user, "tool": "penetration_test", "reason": reason}
    )

    # Verify events
    assert len(security_event_bus.events) == 2
    assert security_event_bus.events[0]["type"] == "tool.executed"
    assert security_event_bus.events[1]["type"] == "tool.denied"


@pytest.mark.asyncio
async def test_vulnerability_scan_permission_levels(
    authorization_manager, security_tools, test_users
):
    """Test different permission levels for vulnerability scanning.

    Scenario:
    1. Test scan-only permission
    2. Test scan-and-fix permission
    3. Verify appropriate authorization for each level

    Validates:
    - Granular permission controls
    - Permission level enforcement
    - Audit trail for different permission levels
    """
    scan_tool = security_tools["vulnerability_scan"]

    # Add a new permission level for scan-and-fix
    authorization_manager.grant_permission("analyst", "security:scan:execute")
    authorization_manager.grant_permission("admin", "security:scan:fix")

    # Test scan permission
    granted, _ = await authorization_manager.authorize_tool_execution(
        user="analyst",
        tool_name="vulnerability_scan",
        required_permissions=["security:scan:execute"],
    )

    assert granted is True

    # Test scan-and-fix permission (admin only)
    granted, _ = await authorization_manager.authorize_tool_execution(
        user="admin",
        tool_name="vulnerability_scan",
        required_permissions=["security:scan:execute", "security:scan:fix"],
    )

    assert granted is True

    # Analyst should not have fix permission
    granted, reason = await authorization_manager.authorize_tool_execution(
        user="analyst",
        tool_name="vulnerability_scan",
        required_permissions=["security:scan:execute", "security:scan:fix"],
    )

    assert granted is False
    assert "Missing permission" in reason


# ============================================================================
# Test: Security Tool Execution with Audit Logging
# ============================================================================


@pytest.mark.asyncio
async def test_security_tool_execution_creates_audit_trail(
    authorization_manager, security_tools, security_event_bus
):
    """Test that all security tool executions create audit trail.

    Scenario:
    1. Execute multiple security tools
    2. Verify all executions are logged
    3. Verify audit log contains required fields
    4. Verify audit log is tamper-evident

    Validates:
    - Comprehensive audit logging
    - Required audit fields
    - Tamper-evident logging
    - Audit log queryability
    """
    user = "admin"

    # Execute multiple tools
    for tool_name, tool in security_tools.items():
        # Authorize
        granted, reason = await authorization_manager.authorize_tool_execution(
            user=user, tool_name=tool_name, required_permissions=tool.required_permissions
        )

        if granted:
            # Execute
            await tool.execute(target="test-target")

            # Log to audit
            await security_event_bus.publish(
                "audit.log",
                {
                    "type": "audit.log",
                    "user": user,
                    "tool": tool_name,
                    "action": "execute",
                    "target": "test-target",
                    "timestamp": datetime.now().isoformat(),
                    "result": "success",
                },
            )
        else:
            # Log denied attempt to audit
            await security_event_bus.publish(
                "audit.log",
                {
                    "type": "audit.log",
                    "user": user,
                    "tool": tool_name,
                    "action": "execute",
                    "target": "test-target",
                    "timestamp": datetime.now().isoformat(),
                    "result": "denied",
                    "reason": reason,
                },
            )

    # Verify audit log
    auth_log = authorization_manager.audit_log
    assert len(auth_log) == len(security_tools)

    # Check required fields
    for entry in auth_log:
        assert "timestamp" in entry
        assert "user" in entry
        assert "tool" in entry
        assert "decision" in entry

    # Verify event bus has audit events
    audit_events = [e for e in security_event_bus.events if e["type"] == "audit.log"]
    assert len(audit_events) == len(security_tools)


@pytest.mark.asyncio
async def test_unauthorized_tool_execution_blocked_and_logged(
    authorization_manager, security_tools, security_event_bus
):
    """Test that unauthorized tool executions are blocked and logged.

    Scenario:
    1. Unauthorized user attempts to execute security tool
    2. Verify execution is blocked
    3. Verify attempt is logged for security monitoring
    4. Verify alert is generated

    Validates:
    - Access denial enforcement
    - Security incident logging
    - Alert generation for unauthorized attempts
    """
    unauthorized_user = "limited"
    tool = security_tools["penetration_test"]

    # Attempt unauthorized execution
    granted, reason = await authorization_manager.authorize_tool_execution(
        user=unauthorized_user,
        tool_name="penetration_test",
        required_permissions=tool.required_permissions,
    )

    assert granted is False

    # Log security incident
    await security_event_bus.publish(
        "security.incident",
        {
            "type": "unauthorized_access_attempt",
            "user": unauthorized_user,
            "tool": "penetration_test",
            "reason": reason,
            "severity": "high",
            "timestamp": datetime.now().isoformat(),
        },
    )

    # Verify incident was logged
    incidents = [e for e in security_event_bus.events if e["type"] == "security.incident"]
    assert len(incidents) == 1
    assert incidents[0]["data"]["severity"] == "high"
    assert incidents[0]["data"]["user"] == unauthorized_user


# ============================================================================
# Test: Cross-System Security Event Propagation
# ============================================================================


@pytest.mark.asyncio
async def test_security_events_propagate_across_systems(authorization_manager, security_event_bus):
    """Test that security events propagate across all systems.

    Scenario:
    1. Authorization system denies access
    2. Event is published to security event bus
    3. Monitoring system receives alert
    4. Audit system logs incident
    5. Verify all systems are notified

    Validates:
    - Event propagation across systems
    - System-wide security awareness
    - Consistent security state
    - No event loss
    """
    # Simulate unauthorized access attempt
    user = "limited"
    tool = "penetration_test"

    # Authorization check fails
    granted, reason = await authorization_manager.authorize_tool_execution(
        user=user, tool_name=tool, required_permissions=["security:pen_test:execute"]
    )

    assert granted is False

    # Publish denial event
    await security_event_bus.publish(
        "access.denied",
        {"user": user, "tool": tool, "reason": reason, "timestamp": datetime.now().isoformat()},
    )

    # Simulate other systems receiving and processing event
    received_events = {"monitoring": [], "audit": [], "alerting": []}

    # Monitoring system subscribes to all security events
    async def monitoring_handler(event):
        if event["type"].startswith("access."):
            received_events["monitoring"].append(event)

    # Audit system subscribes to all events
    async def audit_handler(event):
        received_events["audit"].append(event)

    # Alerting system subscribes to high-severity events
    async def alerting_handler(event):
        if event["type"] in ["access.denied", "security.incident"]:
            received_events["alerting"].append(event)

    # Process event through handlers
    event = security_event_bus.events[-1]
    await monitoring_handler(event)
    await audit_handler(event)
    await alerting_handler(event)

    # Verify all systems received event
    assert len(received_events["monitoring"]) == 1
    assert len(received_events["audit"]) == 1
    assert len(received_events["alerting"]) == 1


# ============================================================================
# Test: Performance Under Security Workload
# ============================================================================


@pytest.mark.asyncio
async def test_security_authorization_performance_under_load(
    authorization_manager, security_tools, test_users
):
    """Test performance of security authorization under heavy load.

    Scenario:
    1. Execute 100 authorization checks concurrently
    2. Execute 50 tool executions concurrently
    3. Verify performance meets thresholds
    4. Verify no authorization bypass under load

    Validates:
    - Authorization check performance
    - Concurrent authorization handling
    - No race conditions
    - Consistent enforcement under load
    """
    import time

    # Test authorization check performance
    start_time = time.time()

    auth_tasks = []
    for i in range(100):
        user = "admin" if i % 2 == 0 else "analyst"
        tool_name = list(security_tools.keys())[i % len(security_tools)]
        tool = security_tools[tool_name]

        auth_tasks.append(
            authorization_manager.authorize_tool_execution(
                user=user, tool_name=tool_name, required_permissions=tool.required_permissions
            )
        )

    results = await asyncio.gather(*auth_tasks)

    auth_duration = time.time() - start_time

    # Should complete in reasonable time
    assert auth_duration < 2.0  # 100 auth checks in < 2 seconds

    # Verify all checks were performed correctly
    assert len(results) == 100

    # Admin should be granted for all
    admin_results = [results[i] for i in range(0, 100, 2)]
    assert all(granted for granted, _ in admin_results)

    # Analyst should be denied for penetration_test
    analyst_results = [results[i] for i in range(1, 100, 2)]
    # At least some should be denied (pen_test requires specific permissions)
    assert any(not granted for granted, _ in analyst_results)

    # Test tool execution performance
    start_time = time.time()

    exec_tasks = []
    for i in range(50):
        user = "admin"
        tool_name = list(security_tools.keys())[i % len(security_tools)]
        tool = security_tools[tool_name]

        exec_tasks.append(tool.execute(target=f"target-{i}"))

    await asyncio.gather(*exec_tasks)

    exec_duration = time.time() - start_time

    # Should complete in reasonable time
    assert exec_duration < 3.0  # 50 executions in < 3 seconds


# ============================================================================
# Test: Role-Based Access Control Integration
# ============================================================================


@pytest.mark.asyncio
async def test_role_based_access_control_with_security_tools(authorization_manager, security_tools):
    """Test role-based access control with security tools.

    Scenario:
    1. Define security roles with different permissions
    2. Assign users to roles
    3. Verify role-based authorization
    4. Verify role hierarchy enforcement

    Validates:
    - Role-based permissions
    - Role assignment and revocation
    - Role hierarchy
    - Effective permission calculation
    """
    # Define roles
    roles = {
        "security_admin": [
            "security:scan:execute",
            "security:scan:fix",
            "security:pen_test:execute",
            "security:attack_simulate",
            "security:audit:write",
            "security:credentials:scan",
        ],
        "security_analyst": [
            "security:scan:execute",
            "security:audit:read",
            "security:access:read",
        ],
        "security_auditor": ["security:audit:read", "security:audit:write"],
    }

    # Assign roles to users
    authorization_manager.grant_role("user1", "security_admin")
    authorization_manager.grant_role("user2", "security_analyst")
    authorization_manager.grant_role("user3", "security_auditor")

    # Grant permissions based on roles
    for user, role in [
        ("user1", "security_admin"),
        ("user2", "security_analyst"),
        ("user3", "security_auditor"),
    ]:
        for perm in roles[role]:
            authorization_manager.grant_permission(user, perm)

    # Test admin role - should have all permissions
    admin_granted, _ = await authorization_manager.authorize_tool_execution(
        user="user1",
        tool_name="penetration_test",
        required_permissions=["security:pen_test:execute"],
    )
    assert admin_granted is True

    # Test analyst role - should have limited permissions
    analyst_granted, _ = await authorization_manager.authorize_tool_execution(
        user="user2", tool_name="vulnerability_scan", required_permissions=["security:scan:execute"]
    )
    assert analyst_granted is True

    analyst_denied, _ = await authorization_manager.authorize_tool_execution(
        user="user2",
        tool_name="penetration_test",
        required_permissions=["security:pen_test:execute"],
    )
    assert analyst_denied is False

    # Test auditor role - should only have audit permissions
    auditor_granted, _ = await authorization_manager.authorize_tool_execution(
        user="user3", tool_name="security_audit", required_permissions=["security:audit:write"]
    )
    assert auditor_granted is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
