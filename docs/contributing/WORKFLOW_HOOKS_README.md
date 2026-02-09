# Pre-Commit Hooks for Workflow Validation

Victor includes pre-commit hooks to automatically validate YAML workflow files before they are committed to git.

## Quick Start

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# That's it! Hooks will now run automatically on git commit
```text

## What Gets Validated

When you commit YAML workflow files, the hook automatically checks:

- ✓ YAML syntax
- ✓ Workflow structure and nodes
- ✓ Node type validation
- ✓ Workflow compilation
- ✓ Node references and dependencies

## Usage

```bash
# Make changes to a workflow
vim victor/coding/workflows/feature.yaml

# Stage and commit - validation runs automatically
git add victor/coding/workflows/feature.yaml
git commit -m "Update feature workflow"

# If validation fails, commit is aborted with error details
# If validation passes, commit proceeds normally
```

## Bypassing Hooks (If Needed)

```bash
# Skip hooks for a single commit (not recommended)
git commit --no-verify -m "WIP: skip hooks"
```text

## Manual Validation

```bash
# Run all hooks manually
pre-commit run --all-files

# Run only workflow validation
pre-commit run validate-workflows --all-files

# Or use the validation script directly
./scripts/hooks/validate_workflows.sh victor/coding/workflows/feature.yaml
```

## Configuration

Environment variables:

```bash
# Enable verbose output
VICTOR_VERBOSE_VALIDATION=1 git commit

# Enable caching for faster validation
VICTOR_CACHE_VALIDATION=1 git commit

# Skip validation temporarily
VICTOR_SKIP_VALIDATION=1 git commit
```text

## Troubleshooting

### Hook not running?
```bash
# Reinstall hooks
pre-commit install

# Verify installation
ls -la .git/hooks/pre-commit
```

### Validation failed?
The error message will show what's wrong. Common issues:

- YAML syntax errors (indentation, unquoted colons)
- Unknown node types
- Invalid node references
- Missing required fields

Fix the errors and try again.

### See full error details
```bash
# Run with verbose output
VICTOR_VERBOSE_VALIDATION=1 ./scripts/hooks/validate_workflows.sh path/to/workflow.yaml
```text

## Documentation

See [PRE_COMMIT_HOOKS.md](PRE_COMMIT_HOOKS.md) for complete documentation.

## Performance

- Average validation: 0.5-2 seconds per file
- With caching: < 0.1 seconds per file
- Runs only on changed workflow files

Hooks are fast enough to not slow down development workflow.

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 1 min
