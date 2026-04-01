# Victor v0.6.0 Migration Checklist

**For**: Users upgrading to Victor AI Framework v0.6.0
**Date**: March 31, 2026
**Breaking Changes**: NONE - 100% backward compatible

---

## ✅ Pre-Upgrade Checklist

### Environment Check

- [ ] Python version is 3.10 or higher
  ```bash
  python --version  # Should be 3.10+
  ```

- [ ] Current Victor version is known
  ```bash
  pip show victor-ai
  ```

- [ ] Backup your current environment
  ```bash
  pip freeze > requirements_before_upgrade.txt
  ```

---

## 🔄 Upgrade Process

### Step 1: Update Dependencies

- [ ] Update `victor-ai` to version 0.6.0 or higher
  ```bash
  pip install --upgrade 'victor-ai>=0.6.0'
  ```

- [ ] If using `victor-sdk`, update to version 0.6.0 or higher
  ```bash
  pip install --upgrade 'victor-sdk>=0.6.0'
  ```

- [ ] Update external vertical packages
  ```bash
  pip install --upgrade 'victor-coding>=0.6.0'
  pip install --upgrade 'victor-devops>=0.6.0'
  pip install --upgrade 'victor-rag>=0.6.0'
  pip install --upgrade 'victor-dataanalysis>=0.6.0'
  pip install --upgrade 'victor-research>=0.6.0'
  pip install --upgrade 'victor-invest>=0.6.0'
  ```

### Step 2: Verify Installation

- [ ] Verify Victor version
  ```bash
  python -c "import victor; print(victor.__version__)"
  # Should output: 0.6.0 or higher
  ```

- [ ] Run health check
  ```bash
  python -m victor health-check
  ```

- [ ] Check that all verticals load
  ```bash
  python -c "
  from victor.verticals.contrib.coding import CodingAssistant
  from victor.verticals.contrib.devops import DevOpsAssistant
  from victor.verticals.contrib.rag import RAGAssistant
  print('✅ All verticals load successfully')
  "
  ```

---

## ✅ Post-Upgrade Verification

### Functionality Check

- [ ] Run your existing tests
  ```bash
  pytest tests/ -v
  ```

- [ ] Test vertical loading
  ```bash
  python -c "
  from victor.framework.agent import Agent
  from victor.verticals.contrib.coding import CodingAssistant

  config = CodingAssistant.get_config()
  print(f'✅ Tools: {len(config.tools)} tools available')
  "
  ```

- [ ] Test entry point scanning
  ```bash
  python -c "
  from victor.framework.entry_point_registry import get_entry_point_registry
  registry = get_entry_point_registry()
  metrics = registry.scan_all()
  print(f'✅ Scanned {metrics.total_groups} groups in {metrics.duration_ms}ms')
  # Should be < 50ms
  "
  ```

### Performance Validation

- [ ] Verify startup time improved
  - Measure startup time before and after upgrade
  - Expected: 200-500ms faster

- [ ] Check entry point scan duration
  ```bash
  python -c "
  from victor.framework.entry_point_registry import get_entry_point_registry
  import time

  registry = get_entry_point_registry()
  start = time.time()
  metrics = registry.scan_all()
  duration = (time.time() - start) * 1000

  print(f'Scan duration: {duration:.2f}ms')
  # Should be < 50ms (typically ~16ms)
  "
  ```

---

## 🆕 Optional Enhancements

The following are **optional** but recommended for taking advantage of new v0.6.0 features:

### For Vertical Developers

- [ ] Add `@register_vertical` decorator to your verticals
  ```python
  from victor.core.verticals.registration import register_vertical

  @register_vertical(
      name="my_vertical",
      version="1.0.0",
      min_framework_version=">=0.6.0",
  )
  class MyVertical(VerticalBase):
      pass
  ```

- [ ] Add version constraints to decorator
  ```python
  @register_vertical(
      name="my_vertical",
      version="1.0.0",
      min_framework_version=">=0.6.0",
      canonicalize_tool_names=True,
  )
  ```

- [ ] Specify extension dependencies (if any)
  ```python
  from victor.core.verticals.registration import register_vertical, ExtensionDependency

  @register_vertical(
      name="my_vertical",
      extension_dependencies=[
          ExtensionDependency("coding", min_version=">=1.0.0"),
      ],
  )
  ```

### For Advanced Users

- [ ] Enable feature flags for gradual rollout
  ```yaml
  # config/feature_flags.yaml
  new_metadata_system: true
  unified_entry_points: true
  version_compatibility_gates: true
  extension_dependency_graph: true
  async_safe_caching: true
  telemetry_instrumentation: true
  namespace_isolation: true
  ```

- [ ] Setup monitoring dashboards
  - Deploy Grafana dashboards from `docs/verticals/monitoring_dashboards.md`
  - Configure Prometheus metrics scraping

- [ ] Enable OpenTelemetry tracing
  ```python
  from victor.core.verticals.telemetry import vertical_load_span

  with vertical_load_span("my_vertical", "load") as span:
      vertical = load_vertical("my_vertical")
      span.status = "success"
  ```

---

## ⚠️ Troubleshooting

### Issue: Deprecation Warnings

**Symptom**: You see deprecation warnings about legacy patterns

**Solution**: These are warnings, not errors. Your code will still work. To silence them, add the `@register_vertical` decorator (see Optional Enhancements above).

### Issue: Import Errors

**Symptom**: `ImportError: cannot import name 'register_vertical'`

**Solution**: Ensure you have `victor-ai>=0.6.0` installed:
```bash
pip install --upgrade 'victor-ai>=0.6.0'
```

### Issue: Version Conflicts

**Symptom**: `VersionConflictError: victor-ai 0.6.0 requires ...`

**Solution**: Update all Victor-related packages:
```bash
pip install --upgrade 'victor-ai>=0.6.0' 'victor-sdk>=0.6.0'
```

### Issue: Performance Regression

**Symptom**: Startup seems slower than before

**Solution**: This should not happen - v0.6.0 is 200-500ms faster. If you see slower startup:
1. Check that entry points are being scanned once (not multiple times)
2. Verify feature flags are enabled
3. Check metrics in Grafana dashboard
4. Report issue if problem persists

---

## 📋 Rollback Procedure

If you encounter issues and need to rollback:

### Quick Rollback

```bash
# 1. Restore previous version
pip install 'victor-ai==0.5.7'

# 2. Restore vertical packages
pip install 'victor-coding==0.5.7'
pip install 'victor-devops==0.5.7'
# ... etc for other verticals

# 3. Verify rollback
python -c "import victor; print(victor.__version__)"
# Should show: 0.5.7
```

### Full Rollback with Requirements

```bash
# 1. Restore from requirements file
pip install -r requirements_before_upgrade.txt

# 2. Verify installation
python -m victor health-check
```

---

## ✅ Success Criteria

You have successfully upgraded to v0.6.0 if:

- [ ] All tests pass
- [ ] No import errors
- [ ] Verticals load correctly
- [ ] Performance is improved (or equal)
- [ ] No critical errors in logs

---

## 📚 Additional Resources

- **Migration Guide**: `docs/verticals/migration_guide.md` - Detailed migration examples
- **API Reference**: `docs/verticals/api_reference.md` - Complete API documentation
- **Best Practices**: `docs/verticals/best_practices.md` - Recommended patterns
- **Release Notes**: `RELEASE_NOTES_v0.6.0.md` - Full release details
- **Known Issues**: `docs/verticals/KNOWN_ISSUES_v0.6.0.md` - Known issues and workarounds

---

## 🆘 Support

If you encounter issues not covered in this checklist:

1. **Check documentation**: See `docs/verticals/` directory
2. **Search issues**: [GitHub Issues](https://github.com/vjsingh1984/victor-ai/issues)
3. **Create issue**: Include your Python version, Victor version, and error message
4. **Contact**: victor-support@example.com

---

## ✅ Upgrade Complete!

Congratulations! You've successfully upgraded to Victor AI Framework v0.6.0.

**What's Next**:
- Explore new features in the documentation
- Add `@register_vertical` decorator to your verticals
- Setup monitoring dashboards for observability
- Enjoy the 200-500ms startup performance improvement! 🚀

---

**Last Updated**: March 31, 2026
**Victor Version**: 0.6.0
**Status**: Production Ready
