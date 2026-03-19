# Quick Start: Publishing victor-sdk to PyPI

## What Was Done

### 1. CI Workflow Updated ✅
- Modified `.github/workflows/packages.yml` to install victor-sdk from locally built wheel
- CI now works regardless of PyPI status
- SDK is built and installed before testing victor-ai

### 2. Documentation Created ✅
- **PUBLISHING_GUIDE.md**: Comprehensive guide with GitHub Actions workflows
- **RELEASE_CHECKLIST.md**: Step-by-step checklist for v0.5.7 release
- **PUBLICATION_STATUS.md**: Current state and what needs to be done

### 3. GitHub Actions Workflows (Already Configured) ✅
Both `packages.yml` and `release.yml` will:
1. Build victor-sdk first
2. Publish SDK to PyPI (using Trusted Publishing)
3. Build and publish victor-ai

## What You Need to Do

### Step 1: Set Up PyPI (One-Time, 5 minutes)

1. **Create PyPI Account**: https://pypi.org/account/register/
2. **Enable 2FA** (required for publishing)
3. **Enable Trusted Publishing**:
   - Go to: https://pypi.org/manage/account/publishing/
   - Click "Add a new publisher"
   - Configure:
     - **PyPI Project**: victor-sdk
     - **Owner**: vjsingh1984
     - **Repository**: victor
     - **Workflow**: packages.yml OR release.yml
     - **Environment**: pypi
   - Click "Add"

### Step 2: Publish SDK (2 minutes)

```bash
# Create and push version tag
git tag -a v0.5.7 -m "Victor SDK v0.5.7 - Protocol definitions for vertical development"
git push origin v0.5.7
```

That's it! GitHub Actions will:
1. Build victor-sdk wheel
2. Publish to PyPI (automatically via Trusted Publishing)
3. Build and publish victor-ai

### Step 3: Verify (1 minute)

```bash
# Check if SDK is on PyPI
pip index versions victor-sdk

# Install from PyPI
pip install victor-sdk==0.5.7

# Test imports
python -c "from victor_sdk.verticals.protocols.base import VerticalBase; print('✓ Works!')"
```

## Important Files

| File | Purpose |
|------|---------|
| `victor-sdk/PUBLICATION_STATUS.md` | Current status and next steps |
| `victor-sdk/RELEASE_CHECKLIST.md` | Detailed release checklist |
| `victor-sdk/PUBLISHING_GUIDE.md` | Comprehensive publishing guide |
| `.github/workflows/packages.yml` | CI workflow (updated) |
| `.github/workflows/release.yml` | Release workflow (already configured) |

## Common Issues

### Issue: "403 Forbidden" from PyPI
**Fix**: Enable Trusted Publishing (Step 1 above)

### Issue: CI fails before SDK is on PyPI
**Fix**: Already fixed! CI now installs local SDK wheel

### Issue: Version mismatch
**Fix**: Both victor-sdk and victor-ai are at v0.5.7 ✅

## After Publication

Once SDK is on PyPI:
- ✅ CI will automatically pass
- ✅ `pip install victor-ai` will work
- ✅ Future releases use same workflow

## Future Releases (v0.6.0+)

```bash
# 1. Update versions
# Edit victor-sdk/pyproject.toml and pyproject.toml

# 2. Verify sync
python scripts/check_version_sync.py

# 3. Commit and tag
git commit -am "bump: version 0.6.0"
git tag -a v0.6.0 -m "v0.6.0"
git push origin main --tags
```

## Summary

- **Current State**: Ready to publish ✅
- **Blocker**: PyPI Trusted Publishing setup (5 minutes)
- **Time to Publish**: ~2 minutes after setup
- **Risk**: Low (changes are already tested locally)

---

**Need Help?** See:
- `victor-sdk/PUBLISHING_GUIDE.md` - Detailed guide with troubleshooting
- `victor-sdk/PUBLICATION_STATUS.md` - Current status and solutions
