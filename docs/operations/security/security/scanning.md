# Security Scanning

This document describes the security scanning tools and workflows used in the Victor project.

## Overview

Victor uses a comprehensive security scanning approach with multiple tools to identify vulnerabilities:

- **Bandit**: Python code security issues (SQL injection, shell injection, etc.)
- **Safety**: Known security vulnerabilities in dependencies
- **Semgrep**: Advanced security rules and custom patterns
- **Pip Audit**: Dependency vulnerability scanning
- **Gitleaks**: Secret and credential detection

## CI/CD Integration

### Quick Security Scans (CI Workflow)

The main CI workflow (`.github/workflows/ci.yml`) runs quick security checks on every push and pull request:

- Gitleaks (secret scanning)
- Pip Audit (dependency vulnerabilities)
- Bandit (high severity only)

### Comprehensive Security Scans (Security Workflow)

The dedicated security workflow (`.github/workflows/security.yml`) runs comprehensive scans:

- On every push to main/develop
- On every pull request to main/develop
- Daily at 2 AM UTC (scheduled scan)
- On manual trigger

## Running Security Tools Locally

### Installation

Install security scanning tools as part of dev dependencies:

```bash
# Ensure pip is up-to-date first (pip>=25.2 required)
pip install --upgrade pip
pip install "pip>=25.2"

# Install dev dependencies
pip install -e ".[dev]"
```

Or install individually:

```bash
pip install bandit[toml] safety semgrep
```

### pip Version Management

**Minimum Required Version:** pip 25.2

**Why:** pip versions prior to 25.2 have known CVE vulnerabilities (CVE-2025-8869, PVE-2025-75180)

**Upgrade Procedure:**
```bash
# Upgrade pip globally
pip install --upgrade pip

# Verify version
pip --version
# Should show: pip 25.2 or higher

# Pin minimum version in environment
pip install "pip>=25.2"
```

**CI/CD Integration:**
- All CI workflows automatically upgrade pip before running dependency installation
- `pyproject.toml` build-system requires `pip>=25.2`
- Safety checks verify pip version as part of dependency vulnerability scanning

**Monitoring:**
- Safety scan reports pip vulnerabilities
- CI workflows fail if pip is not upgraded before dependency installation
- Security scan findings document pip version status

### Bandit

Bandit finds common security issues in Python code.

**Basic usage:**
```bash
bandit -r victor/
```

**High/critical severity only:**
```bash
bandit -r victor/ -lll
```

**Exclude directories:**
```bash
bandit -r victor/ --exclude victor/test/,victor/tests/,tests/
```

**JSON output:**
```bash
bandit -r victor/ -f json -o bandit-report.json
```

**Filter by severity:**
```bash
bandit -r victor/ -ll  # Low and above
bandit -r victor/ -ii  # Medium and above
```

**Common issues detected:**
- SQL injection
- Shell injection
- Hardcoded passwords
- Insecure random number generation
- Use of assert for security checks
- YAML deserialization vulnerabilities

### Safety

Safety checks for known security vulnerabilities in dependencies.

**Basic usage:**
```bash
safety check
```

**JSON output:**
```bash
safety check --json --output safety-report.json
```

**Check specific requirements file:**
```bash
safety check --file requirements.txt
```

**Ignore vulnerabilities (use caution):**
```bash
safety check --ignore 12345
```

**Update vulnerability database:**
```bash
safety check --continue-on-error
```

### Semgrep

Semgrep is a fast, static analysis tool for finding bugs and security issues.

**Basic usage:**
```bash
semgrep --config=auto
```

**Auto-configuration with exclusions:**
```bash
semgrep --config=auto --exclude=tests/ --exclude=archive/
```

**JSON output:**
```bash
semgrep --config=auto --json --output=semgrep-report.json
```

**Error severity only:**
```bash
semgrep --config=auto --severity ERROR
```

**Specific rule sets:**
```bash
# OWASP Top 10
semgrep --config="p/owasp-top-10"

# Python security
semgrep --config="p/python"

# Custom rules
semgrep --config="path/to/rules/"
```

**Common patterns detected:**
- SQL injection
- Command injection
- Cross-site scripting (XSS)
- Insecure deserialization
- Cryptographic issues
- Authentication issues

### Pip Audit

Pip Audit checks for known vulnerabilities in dependencies.

**Basic usage:**
```bash
pip-audit
```

**Specific requirements:**
```bash
pip-audit --requirement requirements.txt
```

**JSON output:**
```bash
pip-audit --format json --output pip-audit-report.json
```

**Check installed packages:**
```bash
pip-audit
```

### Gitleaks

Gitleaks scans for secrets and credentials in code and commit history.

**Basic usage:**
```bash
gitleaks detect --source .
```

**Scan git history:**
```bash
gitleaks git --source .
```

**Configuration:**
```bash
gitleaks detect --source . --config .gitleaks.toml
```

**Common findings:**
- API keys
- Passwords
- Tokens
- Certificates
- Database connection strings

## Troubleshooting

### False Positives

If you encounter false positives, you can configure exclusions:

**Bandit:** Create `.bandit` file:
```yaml
exclude_dirs:
  - victor/test/
  - tests/
skips:
  - B101  # assert_used
```

**Semgrep:** Add inline suppressions:
```python
# nosemgrep: python.lang.security.audit.dangerous-exec-call.dangerous-exec-call
os.system(command)
```

**Safety:** Use `--ignore` flag:
```bash
safety check --ignore 12345
```

### Installation Issues

**Bandit TOML support:**
```bash
pip install "bandit[toml]"
```

**Semgrep installation:**
```bash
# For Python 3.10+
pip install semgrep

# If you encounter issues, try:
python -m pip install semgrep --user
```

**Safety database issues:**
```bash
# Update to latest safety version
pip install --upgrade safety

# Check database connectivity
safety check --debug
```

### Performance Issues

**Semgrep timeout:**
```bash
# Increase timeout
semgrep --config=auto --timeout 600
```

**Bandit slow on large codebases:**
```bash
# Scan specific directories only
bandit -r victor/providers/ victor/tools/

# Use parallel processing
bandit -r victor/ -n 4
```

### CI/CD Issues

**Security workflow failures:**
1. Check individual job logs
2. Download artifacts for detailed reports
3. Verify tool versions match local environment
4. Check for rate limiting (especially Safety API)

**Gitleaks failures:**
- Review commit history for false positives
- Update `.gitleaks.toml` configuration
- Use `gitleaks:ignore` comments for legitimate secrets

## Best Practices

1. **Run security scans locally before pushing**
   ```bash
   make security  # or run individual tools
   ```

2. **Review findings regularly**
   - Security reports are retained for 30 days
   - Download artifacts for detailed analysis
   - Prioritize high/critical severity issues

3. **Keep tools updated**
   ```bash
   pip install --upgrade bandit safety semgrep
   ```

4. **Document legitimate exceptions**
   - Use inline suppressions sparingly
   - Document why exceptions are necessary
   - Review exceptions periodically

5. **Configure tool-specific settings**
   - Adjust severity thresholds
   - Set up custom rules
   - Integrate with pre-commit hooks

## Configuration Files

### Bandit Configuration (`.bandit`)

```yaml
exclude_dirs:
  - victor/test/
  - tests/
  - archive/
skips:
  - B101  # assert_used (common in tests)
  - B601  # paramiko_calls (if using paramiko legitimately)
```

### Semgrep Configuration (`.semgrepignore`)

```
tests/
archive/
victor_test/
*.md
docs/
```

### Gitleaks Configuration (`.gitleaks.toml`)

```toml
title = "Gitleaks Custom Configuration"

[[rules]]
description = "GitHub Token"
id = "github-token"
regex = '''ghp_[a-zA-Z0-9]{36}'''
```

## Integration with Development Workflow

### Pre-commit Hooks

Add security tools to pre-commit hooks (`.pre-commit-config.yaml`):

```yaml
repos:
  - repo: https://github.com/PyCQA/bandit
    rev: '1.7.6'
    hooks:
      - id: bandit
        args: ['-lll', '-r', 'victor/']

  - repo: https://github.com/returntocorp/semgrep
    rev: 'v1.45.0'
    hooks:
      - id: semgrep
        args: ['--config=auto', '--exclude=tests/']
```

### Makefile Integration

Add security targets to Makefile:

```makefile
.PHONY: security
security:
	@echo "Running security scans..."
	bandit -r victor/ -lll || true
	safety check || true
	semgrep --config=auto --severity ERROR || true
	pip-audit || true

.PHONY: security-full
security-full:
	@echo "Running comprehensive security scans..."
	bandit -r victor/ -f json -o bandit-report.json
	safety check --json --output safety-report.json
	semgrep --config=auto --json --output=semgrep-report.json
	pip-audit --format json --output pip-audit-report.json
```

## Additional Resources

- [Bandit Documentation](https://bandit.readthedocs.io/)
- [Safety Documentation](https://pyup.io/safety/)
- [Semgrep Documentation](https://semgrep.dev/docs/)
- [Pip Audit Documentation](https://pip-audit.readthedocs.io/)
- [Gitleaks Documentation](https://github.com/gitleaks/gitleaks)

## Reporting Security Issues

For security vulnerabilities or concerns, please:

1. Do NOT open a public issue
2. Send an email to singhvjd@gmail.com
3. Include details about the vulnerability
4. Wait for confirmation before disclosing

For non-security issues, use the GitHub issue tracker.

---

**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
