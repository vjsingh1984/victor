# Pre-Commit Hooks for Victor

This document describes the pre-commit hook system for validating YAML workflows in Victor.

## Overview

Victor uses pre-commit hooks to automatically validate YAML workflow files before they are committed to git. This helps
  catch errors early and ensures that workflows are properly formatted and can be compiled successfully.

## Installation

### Prerequisites

```bash
# Install pre-commit
pip install pre-commit

# Or using pipx (recommended for system-wide installation)
pipx install pre-commit
```text

### Setup

```bash
# Install the git hooks
pre-commit install

# Enable pre-commit on all git hooks (optional)
pre-commit install --hook-type pre-commit
pre-commit install --hook-type pre-push
pre-commit install --hook-type commit-msg
```

## Workflow Validation Hook

### What It Checks

The workflow validation hook (`validate-workflows`) performs the following checks on YAML workflow files:

1. **YAML Syntax Validation**
   - Validates YAML syntax using Python's YAML parser
   - Catches syntax errors like unquoted colons, bad indentation, etc.

2. **Workflow Structure Validation**
   - Validates workflow definition structure
   - Checks for required fields (name, nodes, start_node)
   - Validates node definitions and properties

3. **Workflow Compilation**
   - Attempts to compile the workflow to a state graph
   - Checks for unknown node types
   - Validates node references (next_nodes, branches, parallel_nodes)
   - Detects circular dependencies

4. **Extended Validation** (optional)
   - Escape hatch references (CONDITIONS/TRANSFORMS)
   - Handler references (when --check-handlers is specified)
   - HITL node configuration
   - Parallel node structure

### Files Validated

The hook only validates YAML files in workflow directories:

```text
victor/*/workflows/*.yaml
victor/*/workflows/*.yml
```

This includes:
- `victor/coding/workflows/*.yaml`
- `victor/devops/workflows/*.yaml`
- `victor/rag/workflows/*.yaml`
- `victor/dataanalysis/workflows/*.yaml`
- `victor/research/workflows/*.yaml`
- `victor/benchmark/workflows/*.yaml`

## Usage

### Automatic Validation (Recommended)

Once installed, the hook runs automatically on `git commit`:

```bash
# Make changes to a workflow file
vim victor/coding/workflows/feature.yaml

# Stage the changes
git add victor/coding/workflows/feature.yaml

# Commit - hooks run automatically
git commit -m "Update feature workflow"
```text

If validation fails, the commit is aborted with an error message.

### Manual Validation

You can run the hooks manually without committing:

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run only the workflow validation hook
pre-commit run validate-workflows --all-files

# Run hooks on specific files
pre-commit run validate-workflows --files victor/coding/workflows/feature.yaml
```

### Using the Validation Script Directly

You can also run the validation script directly:

```bash
# Validate a single file
./scripts/hooks/validate_workflows.sh victor/coding/workflows/feature.yaml

# Validate multiple files
./scripts/hooks/validate_workflows.sh victor/coding/workflows/*.yaml

# With verbose output
VICTOR_VERBOSE_VALIDATION=1 ./scripts/hooks/validate_workflows.sh victor/coding/workflows/feature.yaml
```text

## Configuration

### Environment Variables

The validation script respects the following environment variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `VICTOR_SKIP_VALIDATION` | Skip all validation (use with caution) | `0` |
| `VICTOR_VERBOSE_VALIDATION` | Enable verbose output | `0` |
| `VICTOR_CACHE_VALIDATION` | Enable validation caching | `0` |
| `PYTHON_BIN` | Python interpreter to use | `python3` |

#### Examples

```bash
# Skip validation (not recommended)
git commit --no-verify -m "WIP: skip hooks"

# Enable verbose output for debugging
VICTOR_VERBOSE_VALIDATION=1 git commit -m "Update workflow"

# Enable caching for faster validation
VICTOR_CACHE_VALIDATION=1 pre-commit run --all-files
```

### Pre-Commit Configuration

The hook configuration is stored in `.pre-commit-hooks.yaml`:

```yaml
- repo: local
  hooks:
    - id: validate-workflows
      name: Validate YAML Workflows
      entry: scripts/hooks/validate_workflows.sh
      language: script
      files: '^victor/[^/]+/workflows/.*\.ya?ml$'
      pass_filenames: true
```text

You can customize the hook behavior by modifying this file.

## Bypassing Hooks

### Temporary Bypass

To skip hooks for a single commit:

```bash
git commit --no-verify -m "WIP: skip hooks"
```

### Permanent Bypass (Not Recommended)

To disable pre-commit hooks entirely:

```bash
# Uninstall the hooks
pre-commit uninstall

# Or disable specific hook
# Edit .pre-commit-config.yaml and comment out the hook
```text

## Troubleshooting

### Hook Not Running

If the hook doesn't run automatically:

1. **Check if pre-commit is installed:**
   ```bash
   which pre-commit
   ```

2. **Verify hooks are installed:**
   ```bash
   ls -la .git/hooks/pre-commit
```text

3. **Reinstall hooks:**
   ```bash
   pre-commit install
   ```

### Validation Failures

#### YAML Syntax Error

```text
[FAIL] Invalid YAML syntax in: victor/coding/workflows/feature.yaml
```

**Solution:** Fix the YAML syntax error. Common issues:
- Unquoted colons in strings
- Inconsistent indentation
- Missing closing brackets/quotes

Use a YAML validator to identify the exact issue:
```bash
python3 -c "import yaml; yaml.safe_load(open('victor/coding/workflows/feature.yaml'))"
```text

#### Compilation Error

```
[FAIL] Workflow validation failed: victor/coding/workflows/feature.yaml
  Error: Node 'step1' references non-existent node 'step2'
```text

**Solution:** Fix the workflow definition:
- Ensure all node references exist
- Check for typos in node IDs
- Validate node type compatibility

#### Permission Denied

```
bash: ./scripts/hooks/validate_workflows.sh: Permission denied
```text

**Solution:** Make the script executable:
```bash
chmod +x scripts/hooks/validate_workflows.sh
```

### Python Not Found

```text
[FAIL] Python not found: python3
```

**Solution:** Install Python or set the `PYTHON_BIN` variable:
```bash
# macOS
brew install python3

# Linux (Ubuntu/Debian)
sudo apt-get install python3

# Or set custom Python path
export PYTHON_BIN=/usr/local/bin/python3.9
```text

### Slow Validation

If validation is slow:

1. **Enable caching:**
   ```bash
   export VICTOR_CACHE_VALIDATION=1
   ```

2. **Validate only changed files:**
   ```bash
   pre-commit run validate-workflows --files victor/coding/workflows/changed.yaml
```text

3. **Run hooks in parallel:**
   ```bash
   pre-commit run --all-files --parallel
   ```

## Performance

The validation hook is optimized for speed:

- **Average validation time:** 0.5-2 seconds per file
- **Cached validation:** < 0.1 seconds per file (with `VICTOR_CACHE_VALIDATION=1`)
- **Memory usage:** ~50-100 MB per validation process

### Caching Strategy

When caching is enabled, the hook:
1. Computes a hash of the YAML file
2. Compares with cached hash from previous validation
3. Skips validation if hash matches (file unchanged)

Cache files are stored in `.cache/workflow-validation/` and should be gitignored.

## CI/CD Integration

### GitHub Actions

```yaml
name: Validate Workflows

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install Victor
        run: pip install -e ".[dev]"
      - name: Run pre-commit
        run: pre-commit run --all-files
```text

### GitLab CI

```yaml
validate-workflows:
  script:
    - pip install -e ".[dev]"
    - pre-commit run --all-files
  only:
    - merge_requests
    - main
```

## Contributing

When adding new validation rules:

1. Update `scripts/hooks/validate_workflows.sh`
2. Test the hook manually:
   ```bash
   ./scripts/hooks/validate_workflows.sh victor/coding/workflows/feature.yaml
```text
3. Update this documentation with new rules
4. Add tests to `tests/integration/workflows/test_workflow_yaml_validation.py`

## See Also

- [Workflow Validation Guide](../testing/TESTING_GUIDE.md) - Detailed workflow validation rules
- [Workflow Authoring Guide](../guides/workflow-development/dsl.md) - How to write workflows
- [Pre-Commit Documentation](https://pre-commit.com/) - Official pre-commit documentation
- [YAML Syntax Reference](https://yaml.org/spec/1.2/spec.html) - YAML specification

---

**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
