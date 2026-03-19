# Victor SDK v0.5.7 Release Checklist

> **Current Status**: victor-sdk v0.5.7 synchronized with victor-ai v0.5.7
> **Goal**: Publish victor-sdk to PyPI so CI can resolve the dependency

## Context

The victor-ai package depends on `victor-sdk==0.5.7` from PyPI, but victor-sdk has not been published yet. This causes CI failures when running `pip install -e ".[dev]"`.

**Solution**: Publish victor-sdk to PyPI using the existing GitHub Actions workflows.

## Pre-Publication Verification

- [x] victor-sdk version is 0.5.7 (pyproject.toml)
- [x] victor-ai version is 0.5.7 (pyproject.toml)
- [x] Versions are synchronized
- [x] victor-sdk has ZERO runtime dependencies (only typing-extensions)
- [x] All SDK tests passing locally (37/37 tests)
- [x] victor-ai tests passing locally with SDK dependency
- [x] Code formatted with black
- [x] Linting passed with ruff

## PyPI Setup (Required First-Time Setup)

### 1. Create PyPI Account
- [ ] Go to https://pypi.org/account/register/
- [ ] Create account with email verification
- [ ] Enable 2FA (required for publishing)

### 2. Enable Trusted Publishing (Recommended - No API Tokens Needed)
- [ ] Go to https://pypi.org/manage/account/publishing/
- [ ] Click "Add a new publisher"
- [ ] Configure:
  - **PyPI Project Name**: victor-sdk
  - **Owner**: vjsingh1984
  - **Repository name**: victor
  - **Workflow name**: publish-sdk-pypi (packages.yml) OR release.yml
  - **Environment name**: pypi
- [ ] Click "Add"

**Note**: Trusted Publishing allows GitHub Actions to publish without API tokens.

### 3. Alternative: API Token (Fallback)
If Trusted Publishing doesn't work:
- [ ] Go to https://pypi.org/manage/account/token/
- [ ] Create API token
- [ ] Add token to GitHub Secrets: `PYPI_API_TOKEN`

## GitHub Actions Workflows (Already Configured)

The following workflows are already set up and will publish SDK before victor-ai:

### .github/workflows/packages.yml
- [x] Has `build-sdk` job (lines 30-53)
- [x] Has `publish-sdk-pypi` job (lines 200-219)
- [x] Has `publish-pypi` job that depends on SDK (lines 221-240)
- [x] Trusted Publishing enabled (permissions: id-token: write)

### .github/workflows/release.yml
- [x] Builds both victor-sdk and victor-ai (lines 98-125)
- [x] Has `publish-sdk-pypi` job (lines 461-480)
- [x] Has `publish-pypi` job that depends on SDK (lines 482-500)
- [x] Trusted Publishing enabled (permissions: id-token: write)

## Publication Steps

### Step 1: Update CI to Use Local SDK (Optional - For Development)

This allows CI to pass even before SDK is on PyPI.

**Status**: Not yet implemented. Can be added to packages.yml if needed.

### Step 2: Enable Trusted Publishing on PyPI

1. Log in to PyPI
2. Go to https://pypi.org/manage/account/publishing/
3. Add GitHub Actions publisher for vjsingh1984/victor

### Step 3: Push Version Tag

```bash
# Ensure all changes are committed
git status
git add .
git commit -m "chore: prepare for victor-sdk v0.5.7 release"

# Create and push tag
git tag -a v0.5.7 -m "Victor SDK v0.5.7 - Protocol definitions for vertical development"
git push origin v0.5.7
```

### Step 4: Monitor GitHub Actions

1. Go to: https://github.com/vjsingh1984/victor/actions
2. Watch for "Packages CI" or "Release" workflow
3. Verify `publish-sdk-pypi` job completes successfully
4. Check that `publish-pypi` job runs after SDK publishes

### Step 5: Verify Publication

```bash
# Check if SDK is on PyPI
pip index versions victor-sdk

# Install from PyPI
pip install victor-sdk==0.5.7

# Test imports
python -c "from victor_sdk.verticals.protocols.base import VerticalBase; print('SDK imports work')"

# Verify zero dependencies
pip install --no-deps victor-sdk
# Should only install victor-sdk
```

## Post-Publication Verification

- [ ] victor-sdk is on PyPI: https://pypi.org/project/victor-sdk/
- [ ] Can install victor-sdk from PyPI
- [ ] SDK imports work correctly
- [ ] victor-ai can be installed with SDK dependency
- [ ] CI workflows pass with published SDK
- [ ] Integration tests pass

## Troubleshooting

### Issue: "No matching distribution found for victor-sdk==0.5.7"

**Cause**: SDK not published to PyPI yet

**Solution**: Complete the publication steps above

### Issue: "403 Forbidden" or "Invalid or missing authentication credentials"

**Cause**: Trusted Publishing not configured

**Solution**:
1. Go to https://pypi.org/manage/account/publishing/
2. Add GitHub Actions publisher for vjsingh1984/victor
3. Wait a few minutes for changes to propagate
4. Retry the workflow

### Issue: CI still fails after SDK is published

**Possible causes**:
1. PyPI CDN hasn't propagated yet (wait 5-10 minutes)
2. Version mismatch between SDK and victor-ai
3. Correlated pip cache

**Solutions**:
```bash
# Clear pip cache
pip cache purge

# Install with specific index
pip install -e ".[dev]" --index-url https://pypi.org/simple/
```

## Manual Publishing (If GitHub Actions Fails)

```bash
# Navigate to victor-sdk
cd victor-sdk

# Install build tools
pip install build twine

# Build package
python -m build

# Check package
twine check dist/*

# Publish to PyPI (requires PYPI_API_TOKEN)
export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="<your-pypi-token>"
twine upload dist/*
```

## CI Workflow Updates After Publication

Once victor-sdk is on PyPI, the following will happen automatically:

1. `pip install -e ".[dev]"` will resolve victor-sdk==0.5.7 from PyPI
2. All CI workflows will pass
3. Integration tests will verify package installation
4. Future releases will use the same workflow

## Version Synchronization

**Important**: victor-sdk and victor-ai versions must stay synchronized:

- victor-sdk v0.5.7 ← victor-ai v0.5.7 ✅
- victor-sdk v0.6.0 ← victor-ai v0.6.0 (next release)

When updating versions:
1. Update both pyproject.toml files
2. Run `python scripts/check_version_sync.py` to verify
3. Commit changes
4. Tag both packages with same version

## Success Criteria

- [ ] victor-sdk v0.5.7 published to PyPI
- [ ] Can install SDK from PyPI
- [ ] CI workflows pass
- [ ] victor-ai can be installed with SDK dependency
- [ ] All tests pass in CI

## Next Steps After This Release

1. Continue development
2. When ready for v0.6.0:
   - Update both packages to v0.6.0
   - Tag and push
   - GitHub Actions will publish both packages
   - SDK publishes first, then victor-ai

## Sign-Off

- [x] Code review complete
- [x] Tests passing locally
- [x] Documentation reviewed
- [ ] PyPI setup complete (Trusted Publishing enabled)
- [ ] Published to PyPI
- [ ] CI passing

---

**Current Version**: 0.5.7
**Release Manager**: Vijaykumar Singh
**Status**: Ready for PyPI setup and publication
