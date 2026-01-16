# Pre-commit Hooks Guide

This guide covers using pre-commit hooks in Victor for code quality and consistency.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Available Hooks](#available-hooks)
- [Configuration](#configuration)
- [Usage](#usage)
- [Custom Hooks](#custom-hooks)
- [Troubleshooting](#troubleshooting)

## Overview

Pre-commit hooks automatically run checks before each commit, catching issues early.

### Benefits

- **Consistent code style**: Automatic formatting
- **Catch bugs early**: Linting and type checking
- **Validate workflows**: YAML workflow validation
- **Security scanning**: Detect secrets and vulnerabilities
- **Better commits**: Only clean code gets committed

### Hook Categories

1. **Standard Hooks**: Whitespace, file endings, syntax checks
2. **Python Hooks**: Black, Ruff, MyPy, Bandit
3. **YAML Hooks**: YAML syntax and linting
4. **Custom Hooks**: Workflow validation, import checking, docstring validation

## Installation

### Step 1: Install Pre-commit

```bash
# Using pip
pip install pre-commit

# Using Homebrew (macOS)
brew install pre-commit
```

### Step 2: Install Hooks

```bash
# From project root
pre-commit install

# Install commit-msg hook (optional)
pre-commit install --hook-type commit-msg

# Verify installation
ls .git/hooks/pre-commit
```

### Step 3: Update Hooks

```bash
# Update to latest versions
pre-commit autoupdate

# Update specific repo
pre-commit autoupdate --repo https://github.com/psf/black
```

## Available Hooks

### Standard Hooks

#### Trailing Whitespace

```yaml
- id: trailing-whitespace
  name: Trim trailing whitespace
```

Removes trailing whitespace from lines.

**Fix**: Automatic

#### End of File

```yaml
- id: end-of-file-fixer
  name: Fix end of files
```

Ensures files end with newline.

**Fix**: Automatic

#### File Syntax Checks

```yaml
- id: check-yaml    # YAML syntax
- id: check-toml    # TOML syntax
- id: check-json    # JSON syntax
```

Validates file syntax.

**Fix**: Manual

#### Large Files

```yaml
- id: check-added-large-files
  name: Check for large files
  args: ['--maxkb=1000']
```

Prevents adding files >1MB.

**Fix**: Manual

#### Merge Conflicts

```yaml
- id: check-merge-conflict
  name: Check for merge conflicts
```

Detects merge conflict markers.

**Fix**: Manual

#### Private Keys

```yaml
- id: detect-private-key
  name: Detect private keys
```

Detects private keys in files.

**Fix**: Manual (remove key)

### Python Hooks

#### Black (Formatter)

```yaml
- id: black
  name: Format code with Black
```

Formats Python code to Black style.

**Fix**: Automatic

```bash
# Run manually
black victor tests

# Check without modifying
black --check victor tests

# Diff what would change
black --diff victor tests
```

#### Ruff (Linter)

```yaml
- id: ruff
  name: Lint with Ruff
```

Checks for linting issues.

**Fix**: Automatic for most issues

```bash
# Run manually
ruff check victor tests

# Auto-fix
ruff check --fix victor tests
```

#### MyPy (Type Checker)

```yaml
- id: mypy
  name: Type check with MyPy
  entry: mypy victor/protocols victor/framework/coordinators victor/core/registries
```

Type checks strict-typed modules.

**Fix**: Manual

```bash
# Run manually
mypy victor/protocols victor/framework/coordinators

# Check specific file
mypy victor/agent/coordinators/tool_coordinator.py

# Show error codes
mypy --show-error-codes victor/protocols
```

#### Bandit (Security Scanner)

```yaml
- id: bandit
  name: Security check with Bandit
  args: [-ll, -ii]
```

Checks for security issues.

**Fix**: Manual

```bash
# Run manually
bandit -r victor/ -ll

# Check specific file
bandit victor/agent/orchestrator.py
```

### YAML Hooks

#### yamllint

```yaml
- id: yamllint
  name: Lint YAML files
  args: [-c=.yamllint.yml]
```

Checks YAML formatting.

**Fix**: Manual

```bash
# Run manually
yamllint -c .yamllint.yml victor/**/*.yaml

# Auto-fix (limited)
yamllint -f parsable victor/**/*.yaml | xargs -I{} sh -c 'yamlfix {}'
```

### Custom Hooks

#### Workflow Validation

```yaml
- id: validate-workflows
  name: Validate YAML Workflows
  entry: scripts/hooks/validate_workflows.sh
```

Validates YAML workflows.

**Fix**: Manual

```bash
# Run manually
bash scripts/hooks/validate_workflows.sh victor/coding/workflows/bugfix.yaml

# Validate all
bash scripts/ci/validate_workflows.sh
```

#### Import Checking

```yaml
- id: check-imports
  name: Check Python imports
  entry: python scripts/check_imports.py
```

Checks for import issues.

**Fix**: Manual

```bash
# Run manually
python scripts/check_imports.py
```

#### Mode Config Validation

```yaml
- id: validate-modes
  name: Validate mode configurations
  entry: python -m victor.core.mode_config
```

Validates mode configuration files.

**Fix**: Manual

#### Capability Config Validation

```yaml
- id: validate-capabilities
  name: Validate capability configurations
```

Validates capability configuration files.

**Fix**: Manual

#### Team Config Validation

```yaml
- id: validate-teams
  name: Validate team configurations
```

Validates team configuration files.

**Fix**: Manual

#### TODO Checking

```yaml
- id: check-todos
  name: Check for TODO/FIXME comments
```

Checks for TODO/FIXME/HACK comments.

**Fix**: Manual (warning only)

#### Docstring Validation

```yaml
- id: check-docstrings
  name: Validate docstrings
```

Checks for missing docstrings.

**Fix**: Manual (warning only)

#### File Size Checking

```yaml
- id: check-file-size
  name: Check file size
```

Checks for files >500 lines.

**Fix**: Manual (warning only)

#### Complexity Checking

```yaml
- id: check-complexity
  name: Check code complexity
```

Checks cyclomatic complexity.

**Fix**: Manual (warning only)

## Configuration

### Main Config (`.pre-commit-config.yaml`)

```yaml
# Minimum pre-commit version
minimum_pre_commit_version: 2.20.0

# Default Python version
default_language_version:
  python: python3.10

# CI configuration
ci:
  autofix_commit_msg: |
    [pre-commit.ci] auto fixes from pre-commit hooks
  autofix_prs: true
  autoupdate_schedule: weekly
```

### yamllint Config (`.yamllint.yml`)

```yaml
extends: default

rules:
  line-length:
    max: 120
    level: warning

  indentation:
    spaces: 2

  comments:
    min-spaces-from-content: 1
```

### Excluding Files

```yaml
- id: trailing-whitespace
  exclude: ^(tests/|examples/)

- id: black
  exclude: ^archive/

- id: bandit
  exclude: ^tests/
```

## Usage

### On Commit

```bash
# Stage files
git add victor/coding/workflows/bugfix.yaml

# Commit (hooks run automatically)
git commit -m "feat: add bugfix workflow"

# Hooks run:
# 1. Check YAML syntax
# 2. Validate workflow
# 3. Format with Black
# 4. Lint with Ruff
# 5. Type check with MyPy
# 6. Security check with Bandit
# ...
```

### Manual Execution

```bash
# Run on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files victor/coding/workflows/bugfix.yaml

# Run specific hook
pre-commit run black --all-files

# Run on staged files only
pre-commit run
```

### Skipping Hooks

```bash
# Skip all hooks (not recommended)
git commit --no-verify -m "message"

# Skip specific hook
SKIP=BLACK git commit -m "message"

# Skip multiple hooks
SKIP=BLACK,RUFF git commit -m "message"
```

### Continuous Integration (pre-commit.ci)

Victor uses [pre-commit.ci](https://pre-commit.ci) for CI auto-fixing.

When you push to GitHub:

1. pre-commit.ci runs hooks on your PR
2. Auto-fixes issues (formatting, linting)
3. Creates commit with fixes
4. Comments on PR with results

**Configuration** (in `.pre-commit-config.yaml`):

```yaml
ci:
  autofix_commit_msg: |
    [pre-commit.ci] auto fixes from pre-commit hooks
  autofix_prs: true
  autoupdate_schedule: weekly
```

## Custom Hooks

### Adding Custom Hooks

1. Create hook script in `scripts/hooks/`
2. Add to `.pre-commit-config.yaml`
3. Make executable: `chmod +x scripts/hooks/hook.sh`

### Example: Custom Hook

```yaml
# In .pre-commit-config.yaml
- repo: local
  hooks:
    - id: my-custom-hook
      name: My Custom Hook
      entry: scripts/hooks/my_hook.sh
      language: script
      files: \.py$
```

```bash
#!/bin/bash
# scripts/hooks/my_hook.sh
FILE="$1"

# Your custom logic
if grep -q "TODO" "$FILE"; then
  echo "Warning: TODO found in $FILE"
fi
```

### Hook Types

```yaml
language: system      # System executable
language: python      # Python script
language: script      # Shell script
language: docker_image # Docker container
```

## Troubleshooting

### Hook Fails

```bash
# Run hook manually with verbose output
pre-commit run <hook-id> --verbose --all-files

# Debug specific file
pre-commit run black --files victor/coding/workflows/bugfix.yaml

# Check hook logs
cat ~/.cache/pre-commit/pre-commit.log
```

### Slow Hooks

```bash
# Skip slow hooks temporarily
SKIP=MYPY pre-commit run --all-files

# Use concurrency
pre-commit run --all-files --verbose

# Cache dependencies
pre-commit run --all-files --hook-stage manual
```

### Hook Not Found

```bash
# Reinstall hooks
pre-commit clean
pre-commit install

# Update hooks
pre-commit autoupdate
pre-commit install --overwrite
```

### Permissions Error

```bash
# Make hook executable
chmod +x scripts/hooks/validate_workflows.sh

# Check permissions
ls -la scripts/hooks/
```

### Virtual Environment Issues

```bash
# Clean and recreate
pre-commit clean
pre-commit install --hook-type pre-commit
pre-commit install --hook-type pre-push
```

## Best Practices

1. **Always use hooks**: Don't skip hooks unless necessary
2. **Fix issues immediately**: Don't commit with failing hooks
3. **Auto-fix when possible**: Let hooks fix formatting
4. **Review auto-fixes**: Check what was changed
5. **Keep hooks fast**: Skip slow hooks when needed
6. **Update regularly**: Run `pre-commit autoupdate`
7. **Write custom hooks**: Add project-specific validation

## Resources

- [Pre-commit Documentation](https://pre-commit.com/)
- [pre-commit.ci](https://pre-commit.ci/)
- [Black Documentation](https://black.readthedocs.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
