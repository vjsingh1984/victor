# Coordinator Orchestrator Rollback Guide

**Date**: 2026-01-14
**Feature**: use_coordinator_orchestrator
**Current Status**: ENABLED

---

## Quick Rollback

If you need to disable the coordinator orchestrator, run:

```bash
python scripts/toggle_coordinator_orchestrator.py disable --backup
```

Or manually edit `/Users/vijaysingh/.victor/profiles.yaml`:

```yaml
use_coordinator_orchestrator: false
```

---

## Verification After Rollback

Check the status:

```bash
python scripts/toggle_coordinator_orchestrator.py status
```

Expected output: `Status: DISABLED ✗`

---

## Backup Locations

### Manual Backup
- **Path**: `docs/production/backups/profiles.yaml.manual_backup_20260114_114118`
- **Created**: 2026-01-14 11:41:18
- **Purpose**: Manual backup before rollout

### Script Backup
- **Path**: `/Users/vijaysingh/.victor/backups/profiles.yaml.backup_20260114_114127`
- **Created**: 2026-01-14 11:41:27
- **Purpose**: Automatic backup from toggle script

### Latest Backup
- **Path**: `/Users/vijaysingh/.victor/backups/profiles.yaml.backup_20260114_093001`
- **Created**: 2026-01-14 09:30:01
- **Status**: Previous working configuration

---

## Manual Restore

If you need to restore from a backup:

```bash
# List available backups
ls -la ~/.victor/backups/

# Restore specific backup
cp ~/.victor/backups/profiles.yaml.backup_YYYYMMDD_HHMMSS ~/.victor/profiles.yaml
```

---

## Changes Made During Rollout

### 1. Settings Configuration
**File**: `victor/config/settings.py`

Added `ProfilesYAMLSettingsSource` class to enable loading settings from `profiles.yaml`.

**Lines Added**: ~60 lines
- Custom settings source class
- `settings_customise_sources` classmethod

### 2. Feature Flag
**File**: `~/.victor/profiles.yaml`

```yaml
use_coordinator_orchestrator: true
```

### 3. Code Changes Summary
- **Files Modified**: 1 (`victor/config/settings.py`)
- **New Classes**: 1 (`ProfilesYAMLSettingsSource`)
- **Breaking Changes**: None
- **Data Migration**: None required

---

## Rollback Verification

After rollback, verify the system is working:

```bash
# Check feature flag is disabled
python -c "from victor.config.settings import get_settings; print(get_settings().use_coordinator_orchestrator)"
# Expected: False

# Test basic functionality
python -c "from victor.teams import create_coordinator; print(create_coordinator('sequential'))"
# Expected: No errors
```

---

## Contact

**Questions**: vijay@victor.ai
**Documentation**: See `ROLLOUT_REPORT.md` for full details
