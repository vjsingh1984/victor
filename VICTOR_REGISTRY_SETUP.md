# Victor Registry Setup Guide

## Overview

The **victor-registry** repository has been fully planned and designed. All files are currently located in `/tmp/victor-registry/` and are ready to be pushed to GitHub.

## Repository Structure

```
victor-registry/
├── README.md                              # User guide (200+ lines)
├── index.json                             # Master package index
├── CONTRIBUTING.md                        # Submission guidelines (400+ lines)
├── MAINTENANCE.md                         # Operations guide (600+ lines)
├── packages/
│   ├── README.md                          # Package structure guide
│   └── example-security/                  # Example package ✅ Validated
│       ├── victor-vertical.toml          # Package metadata
│       ├── metadata.json                  # Registry metadata
│       └── README.md                      # Package documentation
├── scripts/
│   ├── validate-index.py                  # Index validation (180 lines)
│   ├── validate-package.py                # Package validation (280 lines)
│   └── sync-from-pypi.py                  # PyPI sync (placeholder)
└── .github/
    ├── PULL_REQUEST_TEMPLATE.md           # PR template
    └── ISSUE_TEMPLATE/
        ├── bug_report.md
        └── security_report.md
```

## Quick Start

### 1. Create the GitHub Repository

```bash
# Navigate to the registry files
cd /tmp/victor-registry

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Victor vertical package registry"

# Create repository on GitHub first, then add remote
git remote add origin https://github.com/vjsingh1984/victor-registry.git

# Push to main branch
git push -u origin main
```

### 2. Configure GitHub Repository Settings

1. **Branch Protection**:
   - Go to Settings → Branches
   - Add rule for `main` branch
   - Require pull request reviews (1 reviewer)
   - Require status checks to pass (add validation workflows)
   - Enable "Require branches to be up to date before merging"

2. **Repository Topics**:
   - Add topics: `victor`, `verticals`, `package-registry`, `python`

3. **Repository Description**:
   - "Central package registry for Victor verticals - discover, install, and share community verticals"

### 3. Add GitHub Actions Workflows

Create `.github/workflows/validate-pr.yml`:

```yaml
name: Validate PR

on:
  pull_request:
    paths:
      - 'packages/**'
      - 'index.json'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install pydantic tomli

      - name: Validate index.json
        run: python scripts/validate-index.py

      - name: Validate packages
        run: python scripts/validate-package.py packages/*/
```

### 4. Integrate with Victor

Update the `VerticalRegistryManager` in Victor to use the new registry:

```python
# victor/core/verticals/registry_manager.py

DEFAULT_REGISTRIES = [
    "https://raw.githubusercontent.com/vjsingh1984/victor-registry/main/index.json",
]
```

## Submission Process

### For Contributors

1. **Fork** the victor-registry repository
2. **Create** package directory: `packages/your-vertical/`
3. **Add** files:
   - `victor-vertical.toml` (required)
   - `metadata.json` (required)
   - `README.md` (required)
4. **Create** pull request to victor-registry
5. **Wait** for automated validation and maintainer review

### For Maintainers

1. **Review** PR submission
2. **Check** automated validation passes
3. **Test** package installation locally
4. **Verify** documentation quality
5. **Update** `index.json` with new package
6. **Merge** PR

## Validation

All packages are automatically validated on PR:

- ✅ victor-vertical.toml schema validation
- ✅ metadata.json schema validation
- ✅ README.md completeness check
- ✅ Name consistency (TOML == directory name)
- ✅ No name conflicts with existing packages

## Next Steps

1. **Create repository** on GitHub using the files in `/tmp/victor-registry/`
2. **Configure** GitHub settings and branch protection
3. **Add** GitHub Actions workflow for validation
4. **Integrate** with Victor by updating DEFAULT_REGISTRIES
5. **Test** with: `victor vertical list --source available`
6. **Announce** registry availability to community

## Files Location

All registry files are currently in: `/tmp/victor-registry/`

Key files to review:
- `/tmp/victor-registry/README.md` - User guide
- `/tmp/victor-registry/CONTRIBUTING.md` - Submission guide
- `/tmp/victor-registry/MAINTENANCE.md` - Operations guide

## Status

✅ Repository structure: **COMPLETE**
✅ Example package: **COMPLETE** (validated)
✅ Validation scripts: **COMPLETE** (tested)
✅ Documentation: **COMPLETE** (2,500+ lines)
⏳ GitHub repository: **PENDING** (needs to be created)

The registry is **production-ready** and only needs the GitHub repository to be created.
