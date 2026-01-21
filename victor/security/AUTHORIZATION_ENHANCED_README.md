# Enhanced Authorization Module

## Overview

The Enhanced Authorization Module provides comprehensive authorization capabilities for Victor AI, combining Role-Based Access Control (RBAC), Attribute-Based Access Control (ABAC), and policy-based authorization with fine-grained control.

## Features

### 1. Role-Based Access Control (RBAC)
- Define roles with sets of permissions
- Assign multiple roles to users
- Permission inheritance from roles
- Default roles: admin, developer, operator, viewer

### 2. Attribute-Based Access Control (ABAC)
- User attributes for fine-grained control
- Dynamic permission evaluation based on context
- Support for complex conditions (time-based, IP-based, etc.)
- Operator-based conditions (gt, lt, in, not_in, etc.)

### 3. Policy-Based Authorization
- Fine-grained policies with priorities
- ALLOW and DENY rules (DENY takes precedence)
- Exception handling for special cases
- Time-based and contextual conditions

### 4. Security Features
- Thread-safe operations with reentrant locks
- Default deny mode (secure by default)
- Audit logging for access decisions
- Integration with event bus for monitoring

### 5. Configuration
- YAML-based configuration
- Runtime modification of roles/users/policies
- Global authorizer instance
- Enable/disable authorization dynamically

## Architecture

### Core Components

```
EnhancedAuthorizer
├── Roles (Role objects)
│   ├── Permissions
│   └── Attributes
├── Users (User objects)
│   ├── Roles
│   ├── Direct Permissions
│   └── Attributes
└── Policies (Policy objects)
    ├── Allow Rules
    ├── Deny Rules
    ├── Conditions
    └── Priority
```

### Data Models

#### Permission
```python
@dataclass(frozen=True)
class Permission:
    resource: str          # Resource type (tools, workflows, etc.)
    action: str            # Action (read, write, execute, delete)
    constraints: Dict[str, Any]  # Optional constraints
```

#### Role
```python
@dataclass
class Role:
    name: str
    permissions: Set[Permission]
    attributes: Dict[str, Any]
    description: str
```

#### User
```python
@dataclass
class User:
    id: str
    username: str
    roles: Set[str]
    attributes: Dict[str, Any]
    permissions: Set[Permission]  # Direct permissions
    enabled: bool
```

#### Policy
```python
@dataclass
class Policy:
    name: str
    effect: PolicyEffect  # ALLOW or DENY
    resource: str
    action: str
    subjects: List[str]
    conditions: Dict[str, Any]
    priority: int
```

## Usage Examples

### Basic Authorization

```python
from victor.security.authorization_enhanced import (
    EnhancedAuthorizer,
    User,
    Permission,
)

# Initialize authorizer
authorizer = EnhancedAuthorizer()

# Create user
user = User(
    id="user1",
    username="alice",
    roles={"developer"},
    attributes={"department": "engineering", "level": "senior"}
)
authorizer.create_user(user.id, user.username, roles=["developer"])

# Check permission
decision = authorizer.check_permission(
    user=user,
    resource="tools",
    action="execute"
)

if decision.allowed:
    print("Access granted:", decision.reason)
else:
    print("Access denied:", decision.reason)
```

### Role Management

```python
# Create custom role
authorizer.create_role(
    name="data_scientist",
    permissions=[
        Permission(resource="workflows", action="execute"),
        Permission(resource="files", action="read"),
        Permission(resource="tools", action="execute", constraints={
            "tool_category": "data_analysis"
        })
    ],
    attributes={"clearance_level": 3},
    description="Data science role with data analysis tools"
)

# Assign role to user
authorizer.assign_role("user1", "data_scientist")

# Grant additional permission
authorizer.grant_permission(
    role_name="data_scientist",
    permission=Permission(resource="api", action="read")
)
```

### Policy-Based Authorization

```python
# Create time-based policy
authorizer.create_policy(
    name="business_hours_only",
    effect=PolicyEffect.ALLOW,
    resource="workflows",
    action="execute",
    subjects=["developer", "operator"],
    conditions={
        "time_window": [9, 17],  # 9 AM to 5 PM
        "allowed_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    },
    priority=50,
    description="Allow workflow execution only during business hours"
)

# Create restrictive policy
authorizer.create_policy(
    name="block_dangerous_tools",
    effect=PolicyEffect.DENY,
    resource="tools",
    action="execute",
    conditions={
        "tool_name": ["rm", "format", "shutdown"]
    },
    priority=100,  # High priority
    description="Block dangerous tools"
)
```

### ABAC with Attributes

```python
# User with attributes
user = User(
    id="user2",
    username="bob",
    attributes={
        "department": "finance",
        "clearance_level": 4,
        "location": "us-east"
    }
)

# Check permission with context
decision = authorizer.check_permission(
    user=user,
    resource="files",
    action="read",
    context={
        "file_classification": "confidential",
        "required_clearance": 3
    }
)

# Policy with attribute conditions
authorizer.create_policy(
    name="confidential_access",
    effect=PolicyEffect.ALLOW,
    resource="files",
    action="read",
    conditions={
        "clearance_level": {"gte": 3},  # Greater than or equal
        "department": ["finance", "legal"]
    },
    priority=60
)
```

### Configuration from YAML

```yaml
# config/authorization.yaml
enabled: true
default_deny: true

roles:
  data_analyst:
    permissions:
      - resource: workflows
        action: execute
      - resource: files
        action: read
        constraints:
          file_type: ["csv", "json", "parquet"]
    attributes:
      clearance_level: 2
    description: Data analyst with file access

users:
  user1:
    username: alice
    roles: [developer, data_analyst]
    attributes:
      department: engineering
    enabled: true

policies:
  - name: business_hours_policy
    effect: allow
    resource: workflows
    action: execute
    subjects: [developer, data_analyst]
    conditions:
      time_window: [9, 17]
    priority: 50
```

```python
# Load configuration
authorizer.load_from_yaml(Path("config/authorization.yaml"))
```

## Authorization Flow

1. **Check if authorization is enabled**
   - If disabled, all requests are allowed

2. **Check if user is enabled**
   - Disabled users are denied access

3. **Evaluate policies (by priority)**
   - DENY policies take precedence
   - First matching policy wins

4. **Check role-based permissions (RBAC)**
   - User roles are checked for matching permissions
   - Permission constraints are evaluated

5. **Check direct user permissions**
   - Direct permissions assigned to user (not through roles)

6. **Default deny (if enabled)**
   - If no permissions match and default_deny is True

## Best Practices

### 1. Principle of Least Privilege
- Start with minimal permissions
- Grant additional permissions as needed
- Use specific resource and action types

### 2. Use Roles for Organization
- Group related permissions into roles
- Assign roles instead of individual permissions
- Use role inheritance for hierarchical permissions

### 3. Policy Priority
- Use higher priorities for security-critical policies
- DENY policies should have higher priority than ALLOW
- Document priority rationale

### 4. Audit Logging
- Enable audit logging for sensitive operations
- Monitor denied access attempts
- Regular review of access patterns

### 5. Attribute Design
- Use consistent attribute naming
- Document attribute values and meanings
- Regularly review attribute assignments

## Security Considerations

### Thread Safety
All operations are thread-safe using reentrant locks:
```python
with self._lock:
    # Critical section
```

### Default Deny
Always use `default_deny=True` for security:
```python
authorizer = EnhancedAuthorizer(default_deny=True)
```

### Policy Evaluation
- Policies are evaluated in priority order (highest first)
- DENY policies take precedence over ALLOW
- First matching policy wins (short-circuit evaluation)

### Audit Trail
All authorization decisions are logged:
```python
logger.info(f"Access decision: {user.username} - {reason}")
```

## Integration with Event Bus

The authorizer integrates with Victor's event system for audit logging:

```python
from victor.core.events import IEventBackend

event_backend = create_event_backend(config)
authorizer = EnhancedAuthorizer()
# Events automatically published to "security.authorization" topic
```

## API Reference

### EnhancedAuthorizer Methods

#### User Management
- `create_user(user_id, username, roles, attributes)` - Create user
- `get_user(user_id)` - Get user by ID
- `get_user_by_username(username)` - Get user by username
- `list_users()` - List all users
- `delete_user(user_id)` - Delete user
- `assign_role(user_id, role_name)` - Assign role to user
- `revoke_role(user_id, role_name)` - Revoke role from user

#### Role Management
- `create_role(name, permissions, attributes, description)` - Create role
- `get_role(name)` - Get role by name
- `list_roles()` - List all roles
- `delete_role(name)` - Delete role

#### Permission Management
- `grant_permission(role_name, permission)` - Grant permission to role
- `revoke_permission(role_name, permission)` - Revoke permission from role

#### Policy Management
- `create_policy(name, effect, resource, action, ...)` - Create policy
- `get_policy(name)` - Get policy by name
- `list_policies()` - List all policies
- `delete_policy(name)` - Delete policy

#### Authorization Checks
- `check_permission(user, resource, action, context)` - Check permission
- `check_permission_by_id(user_id, resource, action, context)` - Check by user ID

#### Configuration
- `load_from_dict(config)` - Load from dictionary
- `load_from_yaml(path)` - Load from YAML file
- `save_to_yaml(path)` - Save to YAML file

#### Control
- `enable()` - Enable authorization
- `disable()` - Disable authorization
- `get_stats()` - Get statistics

## Testing

```python
import pytest
from victor.security.authorization_enhanced import (
    EnhancedAuthorizer,
    User,
    Permission,
    PolicyEffect,
)

def test_basic_authorization():
    authorizer = EnhancedAuthorizer()

    user = User(id="test", username="testuser", roles={"developer"})
    authorizer.create_user(user.id, user.username, roles=["developer"])

    decision = authorizer.check_permission(user, "tools", "read")
    assert decision.allowed is True

def test_policy_deny():
    authorizer = EnhancedAuthorizer()

    authorizer.create_policy(
        name="block_delete",
        effect=PolicyEffect.DENY,
        resource="files",
        action="delete",
        priority=100
    )

    user = User(id="test", username="admin", roles={"admin"})
    authorizer.create_user(user.id, user.username, roles=["admin"])

    decision = authorizer.check_permission(user, "files", "delete")
    assert decision.allowed is False
```

## Performance

The authorization system is optimized for performance:
- O(n) policy evaluation where n = number of policies
- Short-circuit evaluation on first match
- Thread-safe with minimal lock contention
- Efficient permission matching with hash-based lookups

## Troubleshooting

### Common Issues

**Issue: Permission denied unexpectedly**
- Check if user is enabled
- Verify role assignments
- Check policy priorities (DENY might take precedence)
- Review permission constraints

**Issue: Policy not being evaluated**
- Verify policy is enabled
- Check policy priority
- Ensure resource/action match

**Issue: Concurrent modification errors**
- All operations are thread-safe
- Use reentrant locks for complex operations

## Future Enhancements

- Permission expiration/tTL
- Delegation and impersonation
- Advanced auditing and reporting
- Integration with external identity providers
- Policy templates and inheritance
- Time-based permissions (automatic expiration)
- Geographic-based access control
- Resource ownership and delegation
