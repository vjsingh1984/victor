# Victor SDK Publication Status

## Current State

### Version Information
- **victor-sdk**: v0.5.7 (not yet published to PyPI)
- **victor-ai**: v0.5.7 (depends on victor-sdk==0.5.7 from PyPI)
- **Status**: Versions synchronized ✅

### Problem
CI fails when running `pip install -e ".[dev]"` because:
1. victor-ai's pyproject.toml specifies `victor-sdk==0.5.7` as a dependency
2. pip tries to install victor-sdk from PyPI
3. victor-sdk is not published to PyPI yet
4. Error: "No matching distribution found for victor-sdk==0.5.7"

### Solution Implemented

#### 1. CI Workflow Updated (packages.yml)
Modified `.github/workflows/packages.yml` to install victor-sdk from locally built wheel:

```yaml
- name: Download and install victor-sdk wheel
  uses: actions/download-artifact@v4
  with:
    name: victor-sdk-dist
    path: sdk-dist/

- name: Install victor-sdk from local wheel
  run: pip install sdk-dist/*.whl
```

**Benefits**:
- CI works regardless of PyPI status
- Tests run against exact SDK version being built
- No need to publish SDK to PyPI for every development iteration

#### 2. GitHub Actions Workflows (Already Configured)
Both `.github/workflows/packages.yml` and `.github/workflows/release.yml` are configured to:

1. Build victor-sdk first
2. Build victor-ai second
3. Publish SDK to PyPI first (`publish-sdk-pypi` job)
4. Publish victor-ai to PyPI second (`publish-pypi` job with `needs: publish-sdk-pypi`)

**Workflow Diagram**:
```
Trigger (tag v0.5.7)
    ↓
build-sdk (build victor-sdk wheel)
    ↓
test-packages (install local SDK, test victor-ai)
    ↓
publish-sdk-pypi (publish SDK to PyPI)
    ↓
publish-pypi (publish victor-ai to PyPI)
```

## What Needs to Be Done

### Step 1: PyPI Account Setup (One-Time)

1. **Create PyPI Account**:
   - Go to https://pypi.org/account/register/
   - Create account with email verification
   - Enable 2FA (required)

2. **Enable Trusted Publishing** (Recommended):
   - Go to https://pypi.org/manage/account/publishing/
   - Click "Add a new publisher"
   - Configure:
     - **PyPI Project Name**: victor-sdk
     - **Owner**: vjsingh1984
     - **Repository name**: victor
     - **Workflow name**: packages.yml OR release.yml
     - **Environment name**: pypi
   - Click "Add"

**Why Trusted Publishing?**
- No API tokens needed in CI
- More secure (no token rotation)
- GitHub handles authentication
- Automatic with GitHub Actions

### Step 2: Publish SDK to PyPI

#### Option A: Automated via GitHub Actions (Recommended)

```bash
# Ensure all changes are committed
git status
git add .github/workflows/packages.yml victor-sdk/RELEASE_CHECKLIST.md
git commit -m "ci: install victor-sdk from local wheel during testing

- Updates packages.yml to install locally built SDK wheel
- Ensures CI works regardless of PyPI status
- SDK is installed before victor-ai dependencies"

# Push to develop
git push origin develop

# Create and push version tag
git tag -a v0.5.7 -m "Victor SDK v0.5.7 - Protocol definitions for vertical development"
git push origin v0.5.7
```

**What happens**:
1. Tag triggers GitHub Actions (packages.yml or release.yml)
2. Workflow builds victor-sdk wheel
3. Workflow publishes SDK to PyPI (using Trusted Publishing)
4. Workflow builds and publishes victor-ai
5. CI now passes because SDK is on PyPI

#### Option B: Manual Publishing (If GitHub Actions Fails)

```bash
# Navigate to victor-sdk
cd victor-sdk

# Install build tools
pip install build twine

# Build package
python -m build

# Check package
twine check dist/*

# Get PyPI API token from https://pypi.org/manage/account/token/
export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="<your-pypi-token>"

# Publish to PyPI
twine upload dist/*
```

### Step 3: Verify Publication

```bash
# Check if SDK is on PyPI
pip index versions victor-sdk

# Install from PyPI
pip install victor-sdk==0.5.7

# Test imports
python -c "from victor_sdk.verticals.protocols.base import VerticalBase; print('✓ SDK imports work')"

# Verify victor-ai can be installed
pip install victor-ai==0.5.7

# Verify CI passes
# Check https://github.com/vjsingh1984/victor/actions
```

## Current Status

### Completed ✅
- [x] victor-sdk package structure created
- [x] victor-sdk has ZERO runtime dependencies (only typing-extensions)
- [x] victor-ai depends on victor-sdk==0.5.7
- [x] Versions synchronized (both 0.5.7)
- [x] GitHub Actions workflows configured for SDK publishing
- [x] CI workflow updated to install local SDK wheel
- [x] All tests passing locally (37/37 SDK tests)
- [x] PUBLISHING_GUIDE.md created
- [x] RELEASE_CHECKLIST.md updated

### Pending ⏳
- [ ] PyPI account setup
- [ ] Trusted Publishing enabled
- [ ] victor-sdk published to PyPI (v0.5.7)
- [ ] Verify CI passes with published SDK

## Files Modified/Created

### Modified
1. `.github/workflows/packages.yml` - Added local SDK wheel installation
2. `victor-sdk/RELEASE_CHECKLIST.md` - Updated for v0.5.7

### Existing (Already Created)
1. `victor-sdk/PUBLISHING_GUIDE.md` - Comprehensive publishing guide
2. `.github/workflows/packages.yml` - Has publish-sdk-pypi job
3. `.github/workflows/release.yml` - Has publish-sdk-pypi job

## Troubleshooting

### Issue: "403 Forbidden" from PyPI

**Cause**: Trusted Publishing not configured

**Fix**:
1. Go to https://pypi.org/manage/account/publishing/
2. Add GitHub Actions publisher for vjsingh1984/victor
3. Wait 5-10 minutes for changes to propagate
4. Retry the workflow

### Issue: CI still fails after SDK is published

**Cause**: PyPI CDN propagation delay

**Fix**:
```bash
# Wait 5-10 minutes for PyPI CDN to propagate
# Then retry CI or clear pip cache
pip cache purge
```

### Issue: Version mismatch

**Cause**: victor-sdk and victor-ai versions out of sync

**Fix**:
```bash
# Check versions
grep "^version" victor-sdk/pyproject.toml
grep "^version" pyproject.toml

# Verify sync
python scripts/check_version_sync.py
```

## Next Steps

1. **Immediate**: Set up PyPI account and enable Trusted Publishing
2. **Short-term**: Push v0.5.7 tag to trigger automated publishing
3. **Long-term**: Continue development, use same workflow for future releases

## Future Releases (v0.6.0+)

For future releases:

1. Update both `victor-sdk/pyproject.toml` and `pyproject.toml` to new version
2. Run `python scripts/check_version_sync.py` to verify sync
3. Commit changes
4. Tag and push: `git tag -a v0.6.0 -m "..." && git push origin v0.6.0`
5. GitHub Actions automatically publishes SDK first, then victor-ai

---

**Last Updated**: 2025-03-08
**Status**: Ready for PyPI setup and publication
**Next Action**: Set up PyPI Trusted Publishing
