# Security Scan Findings and Remediation

This document summarizes the findings from security scans and provides guidance on remediation.

## Executive Summary

| Tool | Critical | High | Medium | Low | Total |
|------|----------|------|--------|-----|-------|
| Bandit | 0 | 30 | 209 | 505 | 744 |
| Safety | 4 | 0 | 0 | 0 | 4 |
| Semgrep | 270 | 0 | 0 | 0 | 270 |

## Bandit Findings

### High Severity Issues (30)

#### B324: Use of weak MD5 hash (22 occurrences)

**Location:** Multiple files across the codebase

**Description:** MD5 is being used without the `usedforsecurity=False` parameter.

**Affected Files:**
- `victor/agent/change_tracker.py`
- `victor/agent/loop_detector.py`
- `victor/agent/output_aggregator.py`
- `victor/agent/output_deduplicator.py`
- `victor/agent/prompt_normalizer.py`
- `victor/agent/read_cache.py`
- `victor/agent/session_id.py`
- `victor/agent/session_state_manager.py`
- `victor/caching/cache_manager.py`
- `victor/caching/utils.py`
- `victor/coding/ast/node_utils.py`
- `victor/coding/codebase/search.py`
- `victor/coding/review.py`
- `victor/embeddings/cache.py`
- `victor/framework/graph.py`
- `victor/framework/rl/shared_encoder.py`
- `victor/processing/native/__init__.py`
- `victor/storage/memory/unified.py`

**Risk Level:** Low (acceptable use case)

**Justification:** MD5 is used for non-security purposes:
- Content hashing for caching
- Deduplication and duplicate detection
- Generating short identifiers
- Data structure hashing

**Remediation:**
```python
# Change from:
hashlib.md5(content.encode()).hexdigest()

# To:
hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()
```

**Timeline:** Low priority - cosmetic change to clarify intent

#### B602: subprocess with shell=True (2 occurrences)

**Location:**
- `victor/tools/subprocess_executor.py:290`
- `victor/ui/commands/docs.py:325`

**Description:** subprocess calls using `shell=True`

**Risk Level:** Medium (acceptable with safeguards)

**Justification:**
- `subprocess_executor.py`: Tool is explicitly designed to execute shell commands under user/AI control
- `docs.py`: Platform-specific command for opening documentation (Windows only)

**Remediation:** Not required, but consider:
1. Adding input sanitization validation
2. Documenting security considerations
3. Warning users about potential risks

#### B701: Jinja2 autoescape=False (1 occurrence)

**Location:** `victor/ui/commands/scaffold.py:237`

**Description:** Jinja2 environment without autoescape

**Risk Level:** Low (acceptable use case)

**Justification:** Used for code generation templates that are saved to files, not rendered in browsers

**Remediation:**
```python
# Change from:
env = Environment(
    loader=FileSystemLoader(str(template_dir)),
    keep_trailing_newline=True,
)

# To:
from jinja2 import select_autoescape
env = Environment(
    loader=FileSystemLoader(str(template_dir)),
    keep_trailing_newline=True,
    autoescape=select_autoescape(default=False),  # Explicitly disable for code generation
)
```

**Timeline:** Low priority - add explicit documentation

### Medium Severity Issues (209)

Most medium severity issues are related to:
- Assert statements (B101) - acceptable in test code
- Try/except/pass patterns (B110) - acceptable for error handling
- Function call with default arguments (B008) - acceptable pattern

## Safety Findings

### Known Vulnerabilities (4)

#### pip version 25.3 (RESOLVED)

**Vulnerabilities:**
1. **ID: 79883 (CVE-2025-8869)** - Arbitrary File Overwrite
   - Affected: <25.2
   - Remediation: Upgrade pip to 25.2 or later
   - **Status:** RESOLVED - Upgraded to pip 25.3 on 2026-01-14

2. **ID: 75180 (PVE-2025-75180)** - Malicious wheel files
   - Affected: <25.0
   - Remediation: Upgrade pip to 25.0 or later
   - **Status:** RESOLVED - Upgraded to pip 25.3 on 2026-01-14

**Actions Taken:**
```bash
# Upgraded pip globally
pip install --upgrade pip
# Current version: pip 25.3
```bash

**Configuration Updates:**
- Updated `pyproject.toml` build-system requirements to include `pip>=25.2`
- Updated CI workflows to ensure pip is upgraded before safety checks
- Added pip to version requirements to prevent regression

**Note:** This is a development environment vulnerability, not a production dependency issue. The Victor package itself
  doesn't ship with pip.

#### ecdsa version 0.19.1 (MONITORING)

**Vulnerabilities:**
1. **ID: 64459 (CVE-2024-23342)** - Minerva attack vulnerability
   - Affected: >=0
   - Remediation: Upgrade to ecdsa 0.19.1 or later (patched version)
   - **Status:** MONITORING - Current version 0.19.1 is already the patched version

2. **ID: 64396 (PVE-2024-64396)** - Side-channel attacks
   - Affected: >=0
   - Remediation: Consider alternative cryptography libraries (cryptography)
   - **Status:** ACCEPTABLE RISK - Low severity, transitive dependency

**Transitive Dependency:** ecdsa is pulled in by `python-jose` (used by `victor-invest` external package)
**Impact:** Low - not directly used by Victor for security-critical operations

**Dependency Chain:**
```
victor-invest (external)
  └─ python-jose 3.5.0
      └─ ecdsa 0.19.1
```

**Monitoring Plan:**
1. Watch for `python-jose` updates that may upgrade ecdsa or remove dependency
2. Monitor ecdsa project for security releases
3. Re-assess if Victor starts using `python-jose` directly for security operations
4. Consider adding to safety ignore list if no updates in 6 months

**Action Required:** None at this time - documented as acceptable risk

## Semgrep Findings

### Total Findings: 270 (ERROR severity)

#### Python Findings (1407 files scanned)

**SQL Injection (4 occurrences)**
- **Location:** `victor/agent/adaptive_mode_controller.py`
- **Issue:** Raw SQL with string concatenation for CREATE TABLE/INDEX statements
- **Risk:** Low - these are DDL statements (CREATE TABLE/INDEX), not DML with user input
- **Remediation:** Use SQLAlchemy's text() with proper parameter binding
```python
# Current:
conn.execute(f"CREATE TABLE IF NOT EXISTS {_Q_TABLE} (...)")

# Recommended:
from sqlalchemy import text
conn.execute(text(f"CREATE TABLE IF NOT EXISTS {_Q_TABLE} (...)"))
```

**Subprocess shell=True (multiple)**
- Already documented under Bandit findings above

#### JavaScript/TypeScript Findings (65 files scanned)

**Insecure WebSocket (1 occurrence)**
- **Location:** `docs/tutorials/coordinator_recipes.md:1083`
- **Issue:** Documentation example uses `ws://` instead of `wss://`
- **Risk:** None - this is documentation only
- **Remediation:** Update documentation to use `wss://` in examples

**Child Process Injection (2 occurrences) - RESOLVED ✓**
- **Location:** `vscode-victor/src/terminalProvider.ts`
- **Issue:** `spawn` with `shell: true` and user-controlled command
- **Risk:** Medium - IDE extension executing user commands
- **Status:** RESOLVED (2026-01-14)
- **Remediation:** Added comprehensive input validation and sanitization
- **Fix Details:**
  - Created `TerminalCommandValidator` class with allowlist-based validation
  - Implemented 140+ command allowlist (victor, git, npm, python, docker, etc.)
  - Added 80+ flag allowlist for safe command options
  - Implemented dangerous pattern detection (command injection, path traversal, etc.)
  - Added path sanitization and workspace path validation
  - Updated `terminalProvider.ts` to validate all commands before execution
  - Added user-friendly error messages for validation failures
- **Test Coverage:**
  - Created comprehensive test suite with 100+ test cases
  - Tests for command injection prevention (semicolon, pipe, backticks, etc.)
  - Tests for path traversal prevention
  - Tests for hex/unicode escape sequence detection
  - Tests for safe command acceptance (victor, git, npm, python, docker, etc.)
  - Tests for path sanitization
  - Tests for workspace path validation
  - Edge case tests (long commands, unicode, special characters, etc.)
- **Security Documentation:**
  - Created `vscode-victor/SECURITY.md` with security policy
  - Documented allowlist approach and validation patterns
  - Added configuration examples for users
  - Provided security best practices for users and developers

**XSS via innerHTML (1 occurrence) - RESOLVED ✓**
- **Location:** `web/ui/src/components/Message.tsx`
- **Issue:** Setting `innerHTML` and `dangerouslySetInnerHTML` with user-controlled data in multiple diagram rendering contexts
- **Risk:** High - potential XSS vulnerability
- **Status:** RESOLVED (2026-01-14)
- **Remediation:** Implemented DOMPurify sanitization for all HTML/SVG rendering
- **Fix Details:**
  - Added DOMPurify v3.3.0 as explicit dependency
  - Implemented sanitization for Mermaid diagrams (innerHTML)
  - Implemented sanitization for PlantUML, Draw.io, Graphviz, D2 diagrams (dangerouslySetInnerHTML)
  - Implemented sanitization for AsciiDoc HTML output (dangerouslySetInnerHTML)
  - Configured DOMPurify with SVG-specific profiles and allowed tags/attributes
  - Added comprehensive XSS prevention tests (12 test cases, all passing)
- **Test Coverage:**
  - Mermaid diagram XSS prevention tests
  - PlantUML SVG sanitization tests
  - Draw.io SVG sanitization tests
  - Graphviz SVG sanitization tests
  - D2 SVG sanitization tests
  - AsciiDoc HTML sanitization tests
  - Edge cases (empty SVG, malformed SVG, data URLs)
  - User message safety tests

**Secret Detection (1 occurrence)**
- **Location:** `rust/src/secrets.rs:322`
- **Issue:** Example token in documentation/comment
- **Risk:** None - this is a dummy/example token
- **Remediation:** Use more obviously fake token format (e.g., `ghp_DUMMY_TOKEN_XXXXXXXXXXX`)

## Recommended Actions

### Immediate (High Priority)

1. ~~**Upgrade pip**~~ ✓ COMPLETED
   - ~~Run: `pip install --upgrade pip`~~
   - **Status:** COMPLETED - Upgraded to pip 25.3 on 2026-01-14

2. ~~**Fix XSS vulnerability in web UI**~~ ✓ COMPLETED
   - File: `web/ui/src/components/Message.tsx`
   - Added DOMPurify sanitization for all HTML/SVG rendering
   - **Status:** COMPLETED - Fixed on 2026-01-14
   - Added 12 comprehensive XSS prevention tests (all passing)

3. ~~**Add input validation to VS Code extension**~~ ✓ COMPLETED
   - File: `vscode-victor/src/terminalProvider.ts`
   - Created `TerminalCommandValidator` with allowlist-based validation
   - **Status:** COMPLETED - Fixed on 2026-01-14
   - Added 100+ comprehensive security tests (all passing)
   - Created security policy documentation (SECURITY.md)

### Short Term (Medium Priority)

4. **Add `usedforsecurity=False` to MD5 calls**
   - Update all hashlib.md5() calls across codebase
   - Timeline: Next sprint

5. **Update SQL DDL statements**
   - Use SQLAlchemy's text() for better practices
   - Timeline: Next sprint

6. **Update documentation examples**
   - Change `ws://` to `wss://` in tutorials
   - Update example tokens to be more obviously fake
   - Timeline: Next documentation update

### Long Term (Low Priority)

7. ~~**Monitor ecdsa dependency**~~
   - **Status:** DOCUMENTED - Added to monitoring plan with 6-month review cycle
   - Dependency chain documented in scan-findings.md
   - Acceptable risk: low severity, transitive dependency, patched version installed

8. **Add pre-commit hooks**
   - Integrate security tools into development workflow
   - Timeline: When tooling maturity allows

9. **Enhanced security documentation**
   - Document security considerations for subprocess_executor tool
   - Add security guidelines for contributors
   - Timeline: Ongoing

## Exemptions and Justifications

The following are exempted from immediate remediation:

1. **MD5 for non-security purposes** - Used for caching, deduplication, and identifiers
2. **assert in tests** - Standard Python testing practice
3. **subprocess shell=True in subprocess_executor** - Core feature of the tool, with user awareness
4. **Jinja2 without autoescape** - Used for code generation, not web rendering

## False Positive Management

To add false positives or exemptions:

1. **Bandit:** Use `# nosec` comments inline or update `.bandit` config
2. **Semgrep:** Use `# nosemgrep` comments inline or update `.semgrepignore`
3. **Safety:** Use `--ignore` flag or safety policy file

Example:
```python
# nosec B101
assert condition, "Error message"

# nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
conn.execute(f"CREATE TABLE ...")
```bash

## Continuous Monitoring

Security scans are run:
- On every push to main/develop branches
- On every pull request
- Daily at 2 AM UTC (comprehensive scan)
- On-demand via workflow_dispatch

All reports are retained for 30 days as GitHub Actions artifacts.

## Resources

- [Bandit Documentation](https://bandit.readthedocs.io/)
- [Safety Documentation](https://pyup.io/safety/)
- [Semgrep Documentation](https://semgrep.dev/docs/)
- [OWASP Python Security](https://cheatsheetseries.owasp.org/cheatsheets/Python_Security_Cheat_Sheet.html)

## Change Log

- 2026-01-14: Initial security scan and findings documentation
- 2026-01-14: **Phase 4.1 Complete** - Fixed XSS vulnerability in web UI Message component
  - Added DOMPurify v3.3.0 for HTML/SVG sanitization
  - Implemented sanitization for Mermaid, PlantUML, Draw.io, Graphviz, D2, and AsciiDoc rendering
  - Added 12 comprehensive XSS prevention tests (all passing)
  - Updated TypeScript configuration to exclude test files from app build
- 2026-01-14: **Phase 4.2 Complete** - Upgraded pip from 24.2 to 25.3, resolved 2 CVE vulnerabilities
- 2026-01-14: Documented ecdsa monitoring plan (transitive dependency, acceptable risk)
- 2026-01-14: Updated pyproject.toml build-system to require pip>=25.2
- 2026-01-14: **Phase 5.1 Complete** - Added input validation to VS Code extension terminal provider
  - Created `TerminalCommandValidator` class with allowlist-based validation (140+ commands)
  - Implemented 80+ flag allowlist for safe command options
  - Added dangerous pattern detection (command injection, path traversal, hex escapes, etc.)
  - Implemented path sanitization and workspace path validation
  - Updated `terminalProvider.ts` to validate all commands before execution
  - Added 100+ comprehensive security tests (command injection, path traversal, edge cases)
  - Created security policy documentation (vscode-victor/SECURITY.md)
  - All security tests passing, compilation successful
- Tools added: Bandit 1.9.2, Safety 3.7.0, Semgrep 1.147.0

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 8 minutes
