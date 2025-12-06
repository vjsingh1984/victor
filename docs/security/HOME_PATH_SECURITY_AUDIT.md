# Security Audit: Home Directory Path Manipulation Risks

## Executive Summary

This document identifies security vulnerabilities related to `~` (tilde) expansion and `Path.home()` usage in Victor, which system administrators or malicious actors could exploit to:
- Redirect secret storage to attacker-controlled locations
- Access API keys without audit trails
- Exfiltrate credentials through path manipulation

## Risk Matrix

| Risk ID | Severity | Impact | Likelihood | Attack Vector |
|---------|----------|--------|------------|---------------|
| SEC-001 | **CRITICAL** | Credential Theft | Medium | HOME env manipulation |
| SEC-002 | **HIGH** | Path Injection | Medium | VICTOR_DIR_NAME override |
| SEC-003 | **HIGH** | Credential Exposure | Medium | Symlink attacks |
| SEC-004 | **MEDIUM** | Audit Evasion | High | Missing access logging |
| SEC-005 | **MEDIUM** | Race Condition | Low | TOCTOU in file permissions |
| SEC-006 | **LOW** | Memory Exposure | Low | Cached credentials |

---

## SEC-001: HOME Environment Variable Manipulation

### Vulnerability

`Path.home()` in Python resolves to `$HOME` on Unix systems. A system administrator or compromised process can override this:

```bash
# Attacker redirects home directory
export HOME=/tmp/attacker_controlled
victor chat  # Now reads API keys from /tmp/attacker_controlled/.victor/api_keys.yaml
```

### Affected Files (40+ locations)

```
victor/config/api_keys.py:39     DEFAULT_KEYS_FILE = Path.home() / ".victor" / "api_keys.yaml"
victor/config/settings.py:40     GLOBAL_VICTOR_DIR = Path.home() / VICTOR_DIR_NAME
victor/mcp/registry.py:627       search_paths.append(Path.home() / ".config" / "mcp" / "servers.yaml")
victor/security/cve_database.py:635   cache_dir = Path.home() / ".victor" / "cve_cache"
victor/evaluation/harness.py:408      self._results_dir = Path.home() / ".victor" / "evaluations"
# ... and 35+ more
```

### Attack Scenario

1. Sysadmin creates `/shared/fake_victor/.victor/api_keys.yaml` with logging wrapper
2. Sets `HOME=/shared/fake_victor` in user's shell profile or PAM config
3. Victor reads API keys from attacker's location
4. API keys are logged/exfiltrated without user knowledge
5. No audit trail in Victor's logs

### Mitigation Options

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| **A. Hardcode real home** | Immune to HOME override | Breaks containers | Not recommended |
| **B. Session-local env** | Prevents cross-session attacks | Still vulnerable to root | **Partial mitigation** |
| **C. Secure enclave/keyring** | Strong protection | Platform-specific | **Best for API keys** |
| **D. Audit logging** | Detects attacks | Doesn't prevent | **Essential complement** |

### Recommended Fix (Priority 1)

```python
# api_keys.py - Use keyring for critical secrets
import keyring
import os
import pwd

def get_secure_home() -> Path:
    """Get home directory with validation against manipulation."""
    # Get home from passwd database (harder to spoof)
    try:
        real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    except KeyError:
        real_home = Path.home()

    env_home = Path(os.environ.get("HOME", ""))

    # Detect manipulation
    if env_home != real_home:
        logger.warning(
            f"HOME env ({env_home}) differs from passwd ({real_home}). "
            f"Using passwd entry. This may indicate a security issue."
        )
        # AUDIT: Log this event
        audit_log_security_event("HOME_MANIPULATION_DETECTED", {
            "env_home": str(env_home),
            "real_home": str(real_home),
            "uid": os.getuid(),
        })

    return real_home

def get_api_key_secure(provider: str) -> Optional[str]:
    """Get API key with secure storage preference."""
    # Priority 1: Session-specific env var (most secure for automation)
    env_var = PROVIDER_ENV_VARS.get(provider)
    if env_var:
        key = os.environ.get(env_var)
        if key:
            audit_log("API_KEY_ACCESS", provider=provider, source="environment")
            return key

    # Priority 2: System keyring (secure storage)
    try:
        key = keyring.get_password("victor", f"{provider}_api_key")
        if key:
            audit_log("API_KEY_ACCESS", provider=provider, source="keyring")
            return key
    except Exception:
        pass

    # Priority 3: File-based (least secure, with audit)
    # ... existing logic with audit
```

---

## SEC-002: VICTOR_DIR_NAME Environment Override

### Vulnerability

```python
# settings.py:36-37
VICTOR_DIR_NAME = os.getenv("VICTOR_DIR_NAME", ".victor")
GLOBAL_VICTOR_DIR = Path.home() / VICTOR_DIR_NAME
```

Attacker can redirect configuration directory:

```bash
export VICTOR_DIR_NAME="../../../etc/shadow_copy"  # Path traversal
export VICTOR_DIR_NAME="/tmp/malicious_config"    # Absolute path injection
```

### Attack Scenario

1. Set `VICTOR_DIR_NAME=.backdoor_victor`
2. Plant malicious config with attacker's MCP servers
3. Victor loads and executes attacker's tool configurations

### Recommended Fix

```python
def get_victor_dir_name() -> str:
    """Get Victor directory name with validation."""
    dir_name = os.getenv("VICTOR_DIR_NAME", ".victor")

    # Block path traversal
    if ".." in dir_name or "/" in dir_name or "\\" in dir_name:
        logger.error(f"Invalid VICTOR_DIR_NAME: {dir_name!r} contains path characters")
        audit_log_security_event("PATH_TRAVERSAL_ATTEMPT", {"dir_name": dir_name})
        return ".victor"  # Safe default

    # Block hidden files masquerading as directories
    if not dir_name.startswith("."):
        logger.warning(f"VICTOR_DIR_NAME should start with '.': {dir_name}")

    return dir_name
```

---

## SEC-003: Symlink Attacks

### Vulnerability

`Path.expanduser()` and `Path.resolve()` follow symlinks:

```bash
# Attacker creates symlink (requires write access to home directory)
ln -s /attacker/exfiltration ~/.victor

# Or race condition during first run
rm -rf ~/.victor && ln -s /attacker/trap ~/.victor
```

### Affected Code Paths

```python
# file_editor_tool.py:144
file_path = Path(path).expanduser().resolve()

# filesystem.py:120
file_path = Path(path).expanduser().resolve()

# chromadb_provider.py:100
persist_dir = Path(self.config.persist_directory).expanduser()
```

### Recommended Fix

```python
def safe_resolve_path(path: Path, expected_parent: Path) -> Optional[Path]:
    """Resolve path with symlink protection."""
    resolved = path.expanduser().resolve()

    # Verify resolved path is under expected parent
    try:
        resolved.relative_to(expected_parent.resolve())
    except ValueError:
        logger.error(f"Path {path} resolves outside expected directory")
        audit_log_security_event("SYMLINK_ESCAPE_ATTEMPT", {
            "original": str(path),
            "resolved": str(resolved),
            "expected_parent": str(expected_parent),
        })
        return None

    # Check if any component is a symlink
    for parent in resolved.parents:
        if parent.is_symlink():
            logger.warning(f"Symlink detected in path: {parent}")
            audit_log_security_event("SYMLINK_IN_PATH", {"symlink": str(parent)})

    return resolved
```

---

## SEC-004: Missing Audit Logging for Secret Access

### Current State

```python
# api_keys.py - No audit logging when keys are accessed
def get_key(self, provider: str) -> Optional[str]:
    # ... loads key ...
    logger.debug(f"API key for {provider} loaded from environment")  # DEBUG only!
    return env_key
```

**Problems:**
- DEBUG level logs are typically disabled in production
- No structured audit event
- No tracking of which process/user accessed keys
- No correlation ID for forensics

### Recommended Fix

```python
from victor.audit import AuditManager, AuditEventType

def get_key(self, provider: str) -> Optional[str]:
    """Get API key with mandatory audit logging."""
    audit = AuditManager.get_instance()

    # Always audit secret access attempts
    audit.log_event(
        event_type=AuditEventType.SECRET_ACCESS,
        action="api_key_request",
        details={
            "provider": provider,
            "source": "pending",
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "uid": os.getuid(),
            "cwd": os.getcwd(),
        }
    )

    # ... existing logic ...

    # Log result (without exposing key)
    audit.log_event(
        event_type=AuditEventType.SECRET_ACCESS,
        action="api_key_loaded",
        details={
            "provider": provider,
            "source": source,  # "environment", "file", "keyring"
            "key_present": key is not None,
            "key_length": len(key) if key else 0,  # Length only, not value
        }
    )
```

---

## SEC-005: TOCTOU Race Condition in File Permissions

### Vulnerability

```python
# api_keys.py:176-180
with open(self.keys_file, "w") as f:
    yaml.dump(existing_data, f, default_flow_style=False)

# Race window: file exists with default permissions
os.chmod(self.keys_file, 0o600)  # Too late!
```

Between file creation and chmod, the file may have world-readable permissions.

### Recommended Fix

```python
import stat

def set_key(self, provider: str, key: str) -> bool:
    """Save API key with secure file creation."""
    try:
        self.keys_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing
        existing_data = {}
        if self.keys_file.exists():
            with open(self.keys_file, "r") as f:
                existing_data = yaml.safe_load(f) or {}

        if "api_keys" not in existing_data:
            existing_data["api_keys"] = {}
        existing_data["api_keys"][provider] = key

        # Atomic write with secure permissions from the start
        temp_file = self.keys_file.with_suffix(".tmp")

        # Create file with restricted permissions atomically
        fd = os.open(temp_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w") as f:
                yaml.dump(existing_data, f, default_flow_style=False)
        except:
            os.close(fd)
            raise

        # Atomic rename (preserves permissions)
        os.rename(temp_file, self.keys_file)

        return True
    except Exception as e:
        logger.error(f"Failed to save API key: {e}")
        return False
```

---

## SEC-006: Cached Credentials in Memory

### Vulnerability

```python
class APIKeyManager:
    def __init__(self):
        self._cache: Dict[str, Optional[str]] = {}  # Keys stored in plain memory
```

**Risks:**
- Memory dumps expose all cached keys
- Process debugging reveals keys
- Core dumps contain secrets
- No secure memory clearing on program exit

### Recommended Fix (Defense in Depth)

```python
import secrets
import ctypes

class SecureString:
    """Memory-safe string storage with zeroing."""

    def __init__(self, value: str):
        self._value = bytearray(value.encode())
        self._length = len(self._value)

    def get(self) -> str:
        return self._value.decode()

    def clear(self):
        """Securely zero the memory."""
        for i in range(self._length):
            self._value[i] = 0
        # Force memory write
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(self._value)), 0, self._length)

    def __del__(self):
        self.clear()

class APIKeyManager:
    def __init__(self):
        self._cache: Dict[str, Optional[SecureString]] = {}

    def clear_cache(self):
        """Securely clear all cached keys."""
        for key in self._cache.values():
            if key:
                key.clear()
        self._cache.clear()
```

---

## Additional Attack Vectors

### XDG Config Path Manipulation

```python
# mcp/registry.py uses ~/.config/mcp/servers.yaml
# XDG_CONFIG_HOME can redirect this
export XDG_CONFIG_HOME=/tmp/attacker
```

### Plugin Directory Injection

```python
# plugin_registry.py:27
plugin_dirs=[Path("~/.victor/plugins")]

# Attacker can plant malicious plugins
```

### Embedding Cache Poisoning

```python
# Cache directories store embeddings
cache_dir = Path.home() / ".victor" / "embeddings"
# Attacker could poison embeddings to influence tool selection
```

---

## Summary: Recommended Mitigations by Priority

### Priority 1 (Immediate)

1. **Add audit logging for ALL secret access** (SEC-004)
2. **Fix TOCTOU race condition** in file permissions (SEC-005)
3. **Validate VICTOR_DIR_NAME** against path traversal (SEC-002)

### Priority 2 (Short-term)

4. **Implement HOME directory validation** against passwd (SEC-001)
5. **Add symlink detection** for critical paths (SEC-003)
6. **Integrate with system keyring** for API keys (SEC-001)

### Priority 3 (Medium-term)

7. **Implement SecureString** for memory safety (SEC-006)
8. **Add XDG path validation**
9. **Plugin sandboxing** with signature verification

---

## Environment Variable Security Comparison

| Method | Persistence | Visibility | Root Access | Recommendation |
|--------|-------------|------------|-------------|----------------|
| `export VAR=val` | Session | `ps`, `/proc` | Full | Session-only secrets |
| `.bashrc` | Permanent | File access | Full | **Avoid for secrets** |
| `/etc/environment` | System-wide | All users | Full | **Never for secrets** |
| `systemd --user` | Service | systemctl | Partial | CI/CD only |
| **Keyring** | Permanent | Encrypted | Kernel-level | **Recommended** |

### Most Secure Configuration

```bash
# For interactive use - session-only
export ANTHROPIC_API_KEY=$(security find-generic-password -s victor-anthropic -w)

# For automation - use keyring
keyring set victor anthropic_api_key

# Victor checks in order:
# 1. Session environment (most trusted)
# 2. System keyring (persistent, encrypted)
# 3. File (least trusted, audited)
```

---

## Audit Event Types to Add

```python
class SecurityAuditEventType(Enum):
    HOME_MANIPULATION_DETECTED = "home_manipulation"
    PATH_TRAVERSAL_ATTEMPT = "path_traversal"
    SYMLINK_ESCAPE_ATTEMPT = "symlink_escape"
    SECRET_ACCESS = "secret_access"
    SECRET_ACCESS_DENIED = "secret_access_denied"
    PERMISSION_ESCALATION = "permission_escalation"
    CONFIG_OVERRIDE_DETECTED = "config_override"
```

---

## Testing Recommendations

```bash
# Test HOME manipulation detection
HOME=/tmp/fake_home python -c "from victor.config.api_keys import get_api_key; get_api_key('anthropic')"

# Test path traversal
VICTOR_DIR_NAME="../../../tmp" victor --version

# Test symlink detection
ln -s /tmp ~/.victor_test && python -c "from pathlib import Path; print(Path('~/.victor_test').expanduser().resolve())"
```

---

*Document Version: 1.0*
*Last Updated: December 2024*
*Author: Security Audit*
