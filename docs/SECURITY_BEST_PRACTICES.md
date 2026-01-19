# Security Best Practices Guide - Victor AI

**Version**: 0.5.1
**Last Updated**: 2025-01-18
**Maintainer**: Security Team

---

## Table of Contents

1. [Input Validation](#1-input-validation)
2. [Output Sanitization](#2-output-sanitization)
3. [Authentication and Authorization](#3-authentication-and-authorization)
4. [Cryptography](#4-cryptography)
5. [Data Protection](#5-data-protection)
6. [Secure Communication](#6-secure-communication)
7. [Error Handling](#7-error-handling)
8. [Logging and Monitoring](#8-logging-and-monitoring)
9. [Dependencies](#9-dependencies)
10. [Testing](#10-testing)

---

## 1. Input Validation

### 1.1 General Principles

**✅ DO:**
- Validate all user input before processing
- Use allowlists (whitelists) instead of blocklists
- Validate type, length, format, and range
- Use Pydantic models for structured input validation

**❌ DON'T:**
- Trust user input without validation
- Use blocklists only (easy to bypass)
- Concatenate user input directly into commands/queries
- Assume client-side validation is sufficient

### 1.2 Command Injection Prevention

```python
# ❌ BAD: Direct string concatenation with shell=True
import subprocess
user_input = "file.txt; rm -rf /"
subprocess.run(f"cat {user_input}", shell=True)  # DANGEROUS!

# ✅ GOOD: Use list arguments with shell=False
subprocess.run(["cat", user_input], shell=False)

# ✅ BETTER: Use Victor's safe subprocess executor
from victor.tools.subprocess_executor import run_command

result = await run_command(
    command=["cat", user_input],
    working_dir="/safe/path",
    timeout=30,
)
```

### 1.3 Path Traversal Prevention

```python
from pathlib import Path

def safe_path_join(base_dir: str, user_path: str) -> Path:
    """Safely join paths preventing directory traversal."""
    base = Path(base_dir).resolve()
    target = (base / user_path).resolve()

    # Ensure target is within base directory
    if not str(target).startswith(str(base)):
        raise ValueError("Path traversal attempt detected")

    return target

# ✅ Usage
safe_path = safe_path_join("/var/data", "../../etc/passwd")  # Raises ValueError
```

### 1.4 SQL Injection Prevention

```python
# ❌ BAD: String concatenation
query = f"SELECT * FROM users WHERE name = '{user_input}'"
cursor.execute(query)

# ✅ GOOD: Parameterized queries
query = "SELECT * FROM users WHERE name = %s"
cursor.execute(query, (user_input,))

# ✅ BETTER: Use ORM (SQLAlchemy, Peewee, etc.)
user = User.select().where(User.name == user_input).first()
```

### 1.5 XSS Prevention

```python
# ❌ BAD: Render user input without escaping
def render_html(user_input: str) -> str:
    return f"<div>{user_input}</div>"

# ✅ GOOD: Use Jinja2 with autoescape
import jinja2

env = jinja2.Environment(
    loader=jinja2.FileSystemLoader('templates'),
    autoescape=True,  # Enable autoescaping
)

# ✅ BETTER: Use HTML escaping library
import html

def render_html(user_input: str) -> str:
    escaped = html.escape(user_input)
    return f"<div>{escaped}</div>"
```

### 1.6 Using Victor's Input Sanitization

```python
from victor.security.safety import (
    detect_secrets,
    detect_pii_in_content,
    CodePatternScanner,
)

def validate_tool_input(user_input: str, tool_name: str) -> bool:
    """Comprehensive input validation."""

    # Check for secrets
    secrets = detect_secrets(user_input)
    if secrets:
        raise ValueError(f"Input contains {len(secrets)} potential secrets")

    # Check for PII
    pii = detect_pii_in_content(user_input)
    if pii:
        logger.warning(f"Input contains {len(pii)} PII items")

    # Check for dangerous command patterns
    scanner = CodePatternScanner()
    result = scanner.scan_command(user_input)

    if result.dangerous:
        raise ValueError(
            f"Dangerous command pattern detected: {result.patterns}"
        )

    return True
```

---

## 2. Output Sanitization

### 2.1 Secret Masking

```python
from victor.security.safety import mask_secrets

def process_and_log_output(output: str) -> str:
    """Process output and mask any secrets before logging."""

    # Mask secrets before logging
    safe_output = mask_secrets(output)

    # Log the sanitized output
    logger.info(f"Tool output: {safe_output}")

    return safe_output
```

### 2.2 PII Redaction

```python
from victor.security.safety import detect_pii_in_content

def redact_pii(content: str) -> str:
    """Redact PII from content."""

    pii_matches = detect_pii_in_content(content)
    redacted = content

    for match in sorted(pii_matches, key=lambda m: m.start, reverse=True):
        redacted = (
            redacted[:match.start] +
            f"[REDACTED {match.pii_type.value}]" +
            redacted[match.end:]
        )

    return redacted
```

### 2.3 Using Output Validation Middleware

```python
from victor.framework.middleware import (
    OutputValidationMiddleware,
    SecretMaskingMiddleware,
)

# Apply middleware to tool executor
middleware_chain = MiddlewareChain()
middleware_chain.add(SecretMaskingMiddleware())
middleware_chain.add(OutputValidationMiddleware())

# Process tool output through middleware
safe_output = await middleware_chain.process(raw_output)
```

---

## 3. Authentication and Authorization

### 3.1 Using RBAC

```python
from victor.security.auth import RBACManager, Permission, Role

# Initialize RBAC
rbac = RBACManager()

# Define roles and permissions
developer = Role(
    name="developer",
    permissions=[
        Permission.READ,
        Permission.WRITE,
        Permission.EXECUTE_TOOLS,
    ]
)

admin = Role(
    name="admin",
    permissions=[
        Permission.READ,
        Permission.WRITE,
        Permission.EXECUTE_TOOLS,
        Permission.DELETE,
        Permission.MANAGE_USERS,
    ]
)

# Check permissions
def perform_action(user: str, action: Permission, resource: str):
    if not rbac.check_permission(user, action, resource):
        raise PermissionError(
            f"User {user} lacks permission {action.value} on {resource}"
        )
    # Perform action
```

### 3.2 API Key Management

```python
import os
from pathlib import Path

def get_api_key(service: str) -> str:
    """Securely retrieve API key from environment."""

    api_key = os.getenv(f"{service.upper()}_API_KEY")

    if not api_key:
        raise ValueError(f"API key for {service} not found in environment")

    # Validate key format
    if not validate_api_key_format(api_key, service):
        raise ValueError(f"Invalid API key format for {service}")

    return api_key

def validate_api_key_format(key: str, service: str) -> bool:
    """Validate API key format."""
    # Service-specific validation
    formats = {
        "openai": lambda k: k.startswith("sk-"),
        "anthropic": lambda k: k.startswith("sk-ant-"),
    }

    validator = formats.get(service, lambda k: len(k) > 20)
    return validator(key)
```

---

## 4. Cryptography

### 4.1 Hash Usage

```python
import hashlib

# ❌ BAD: Using MD5 for security
hash_md5 = hashlib.md5(data.encode()).hexdigest()

# ✅ GOOD: Using SHA-256
hash_sha256 = hashlib.sha256(data.encode()).hexdigest()

# ✅ ACCEPTABLE: MD5 with usedforsecurity=False
hash_md5_safe = hashlib.md5(
    data.encode(),
    usedforsecurity=False
).hexdigest()

# ✅ BETTER: For passwords, use bcrypt/argon2
import bcrypt

hashed = bcrypt.hashpw(
    password.encode(),
    bcrypt.gensalt()
)
```

### 4.2 Random Number Generation

```python
import secrets
import random

# ❌ BAD: Using random for security
token = random.randint(0, 1000000)

# ✅ GOOD: Using secrets module
token = secrets.randbelow(1000000)

# ✅ GOOD: Generating secure tokens
api_key = secrets.token_urlsafe(32)
```

### 4.3 Encryption

```python
from cryptography.fernet import Fernet

# Generate key
key = Fernet.generate_key()
cipher = Fernet(key)

# Encrypt
encrypted = cipher.encrypt(plaintext.encode())

# Decrypt
decrypted = cipher.decrypt(encrypted).decode()
```

---

## 5. Data Protection

### 5.1 Secrets Detection

```python
from victor.security.safety import SecretScanner, SecretSeverity

scanner = SecretScanner()

# Scan code before committing
secrets = scanner.scan(file_content)

for secret in secrets:
    if secret.severity == SecretSeverity.CRITICAL:
        # Block commit
        raise ValueError(
            f"Critical secret detected at line {secret.line_number}: "
            f"{secret.secret_type}"
        )

    # Log lower severity secrets
    logger.warning(
        f"Secret found: {secret.secret_type} at line {secret.line_number}"
    )
```

### 5.2 PII Handling

```python
from victor.security.safety import (
    detect_pii_columns,
    PIIType,
)

def analyze_dataset(df: pd.DataFrame) -> dict:
    """Analyze dataset for PII columns."""

    pii_columns = detect_pii_columns(df.columns.tolist())

    report = {
        "total_columns": len(df.columns),
        "pii_columns": len(pii_columns),
        "pii_by_type": {},
    }

    for col, pii_type in pii_columns:
        report["pii_by_type"][pii_type.value] = (
            report["pii_by_type"].get(pii_type.value, 0) + 1
        )

    return report
```

### 5.3 Data Encryption at Rest

```python
from cryptography.fernet import Fernet
import json

class SecureStorage:
    """Encrypted file storage."""

    def __init__(self, key: bytes):
        self.cipher = Fernet(key)

    def save(self, filepath: Path, data: dict):
        """Save encrypted data to file."""

        # Serialize to JSON
        json_data = json.dumps(data).encode()

        # Encrypt
        encrypted = self.cipher.encrypt(json_data)

        # Write to file
        filepath.write_bytes(encrypted)

    def load(self, filepath: Path) -> dict:
        """Load and decrypt data from file."""

        # Read encrypted data
        encrypted = filepath.read_bytes()

        # Decrypt
        decrypted = self.cipher.decrypt(encrypted)

        # Deserialize
        return json.loads(decrypted.decode())
```

---

## 6. Secure Communication

### 6.1 HTTPS Only

```python
import httpx

# ✅ GOOD: Always use HTTPS
client = httpx.Client(verify=True)  # Verify SSL certificates
response = client.get("https://api.example.com/data")

# ❌ BAD: Never use HTTP for sensitive data
response = httpx.get("http://api.example.com/data")

# ⚠️ SPECIAL CASE: Local development only
# Use HTTP only for localhost development
if os.getenv("ENVIRONMENT") == "development":
    client = httpx.Client(verify=False)  # Skip SSL verification
```

### 6.2 Certificate Validation

```python
import httpx
import ssl

# ✅ GOOD: Custom certificate validation
client = httpx.Client(
    verify="/path/to/ca-bundle.crt"
)

# ✅ GOOD: Disable SSL for specific local testing only
if is_local_development():
    client = httpx.Client(verify=False)
else:
    client = httpx.Client(verify=True)
```

---

## 7. Error Handling

### 7.1 Don't Expose Sensitive Information

```python
# ❌ BAD: Exposing internal details
def handle_error(error: Exception):
    return f"Error: {error} - {traceback.format_exc()}"

# ✅ GOOD: Generic error message
def handle_error(error: Exception):
    logger.error(f"Internal error: {error}", exc_info=True)
    return "An error occurred. Please try again later."

# ✅ BETTER: Contextual but safe error messages
def handle_file_error(error: FileNotFoundError, filename: str):
    logger.error(f"File not found: {filename}", exc_info=True)
    return f"File '{filename}' could not be found. Please check the filename."
```

### 7.2 Secure Exception Handling

```python
import logging

logger = logging.getLogger(__name__)

def process_user_input(user_input: str):
    try:
        # Process input
        result = dangerous_operation(user_input)
        return result

    except ValueError as e:
        # Expected error - safe to show user
        logger.warning(f"Validation error: {e}")
        return {"error": str(e)}

    except Exception as e:
        # Unexpected error - log details but don't expose
        logger.error(f"Unexpected error processing input: {e}", exc_info=True)
        return {"error": "An unexpected error occurred"}
```

---

## 8. Logging and Monitoring

### 8.1 Secure Logging

```python
from victor.security.safety import mask_secrets

import logging

logger = logging.getLogger(__name__)

def log_user_action(user: str, action: str, data: dict):
    """Log user action with sensitive data masked."""

    # Mask sensitive fields
    safe_data = mask_secrets(json.dumps(data))

    # Log with context
    logger.info(
        "User action",
        extra={
            "user": user,
            "action": action,
            "data": safe_data,
        }
    )
```

### 8.2 Audit Logging

```python
from victor.security.audit import AuditManager

audit = AuditManager.get_instance()

async def sensitive_operation(user: str, resource: str, action: str):
    """Perform sensitive operation with audit logging."""

    await audit.log_event(
        event_type="sensitive_operation",
        user=user,
        resource=resource,
        action=action,
        timestamp=datetime.utcnow(),
        metadata={
            "ip_address": get_client_ip(),
            "user_agent": get_user_agent(),
        }
    )

    # Perform operation
```

---

## 9. Dependencies

### 9.1 Dependency Scanning

```bash
# Run security scans
pip-audit
safety check
bandit -r victor/

# Update dependencies
pip install --upgrade -r requirements.txt

# Check for known vulnerabilities
pip-audit --desc
```

### 9.2 Pinning Versions

```txt
# requirements.txt
package==1.2.3  # Pin exact version
```

```toml
# pyproject.toml
dependencies = [
    "package>=1.2.3,<2.0.0",  # Version range
]
```

### 9.3 HuggingFace Revision Pinning

```python
# ❌ BAD: Unpinned dataset
dataset = load_dataset("username/dataset")

# ✅ GOOD: Pinned to specific revision
dataset = load_dataset(
    "username/dataset",
    revision="abc123def456",  # Commit hash
)
```

---

## 10. Testing

### 10.1 Security Unit Tests

```python
import pytest
from victor.security.safety import detect_secrets

def test_secret_detection():
    """Test that secrets are detected."""

    code_with_secret = """
    API_KEY = "sk-1234567890abcdef"
    """

    secrets = detect_secrets(code_with_secret)

    assert len(secrets) == 1
    assert secrets[0].secret_type == "OpenAI API Key"
    assert secrets[0].severity == SecretSeverity.CRITICAL

def test_command_injection_prevention():
    """Test that command injection is prevented."""

    malicious_input = "file.txt; rm -rf /"

    with pytest.raises(ValueError, match="unsafe"):
        validate_command_input(malicious_input)
```

### 10.2 Integration Tests

```python
import pytest
from victor.security.auth import RBACManager, Permission

@pytest.mark.integration
def test_rbac_enforcement():
    """Test that RBAC correctly enforces permissions."""

    rbac = RBACManager()

    # Grant read permission only
    rbac.grant_role("user1", "reader")

    # Should succeed
    assert rbac.check_permission("user1", Permission.READ, "file1")

    # Should fail
    assert not rbac.check_permission("user1", Permission.WRITE, "file1")
```

### 10.3 Security Regression Tests

```python
import pytest
from victor.tools.subprocess_executor import run_command

@pytest.mark.security
async def test_no_shell_injection():
    """Test that shell injection is prevented."""

    malicious_input = "; cat /etc/passwd"

    with pytest.raises(ValueError, match="dangerous"):
        await run_command(
            command=["echo", malicious_input],
            timeout=1,
        )
```

---

## Checklist

### Development Checklist

- [ ] All user input is validated before use
- [ ] Commands use list arguments (no shell=True)
- [ ] Secrets are masked in logs and output
- [ ] PII is handled with appropriate safeguards
- [ ] Strong cryptography (SHA-256+, bcrypt) is used
- [ ] Error messages don't expose sensitive data
- [ ] Dependencies are scanned for vulnerabilities
- [ ] Security tests are added for new features
- [ ] Code is reviewed for security issues

### Deployment Checklist

- [ ] All dependencies are up-to-date
- [ ] No known HIGH severity vulnerabilities
- [ ] API keys are stored in environment variables
- [ ] TLS/SSL is enabled for all network communication
- [ ] Audit logging is enabled
- [ ] Rate limiting is configured
- [ ] Security monitoring is in place
- [ ] Incident response plan is documented

---

## Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [OWASP Cheat Sheets](https://cheatsheetseries.owasp.org/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [Cryptography Documentation](https://cryptography.io/en/latest/)

---

**Last Updated**: 2025-01-18
**Next Review**: 2025-04-18
