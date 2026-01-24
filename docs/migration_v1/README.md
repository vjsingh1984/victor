# Victor Migration Documentation

Complete migration guides and resources for upgrading from Victor 0.5.x to 0.5.0.

## Quick Start

```bash
# 1. Validate current state
python scripts/migration/validate_migration.py

# 2. Migrate configuration
python scripts/migration/migrate_config.py --source .env --dest ./victor/config

# 3. Migrate workflows
python scripts/migration/migrate_workflows.py --source ./old_workflows --dest ./victor/workflows

# 4. Validate migration
python scripts/migration/validate_migration.py --report migration_report.txt
```

## Documentation

### Main Guides

1. **[Migration Guide](./MIGRATION_GUIDE.md)** - Complete migration overview
   - Architectural improvements
   - Breaking changes
   - New features
   - Step-by-step upgrade instructions

2. **[API Migration Guide](./MIGRATION_API.md)** - API changes with before/after examples
   - Core API changes
   - Provider API changes
   - Tool API changes
   - Orchestrator API changes
   - Workflow API changes
   - Event API changes

3. **[Configuration Migration Guide](./MIGRATION_CONFIG.md)** - Configuration changes
   - Settings file changes
   - Environment variable changes
   - YAML configuration changes
   - Provider configuration
   - Feature flags

4. **[Workflow Migration Guide](./MIGRATION_WORKFLOWS.md)** - Workflow changes
   - StateGraph DSL changes
   - Workflow compiler changes
   - Node type changes
   - Edge and routing changes
   - Migration examples

5. **[Testing Migration Guide](./MIGRATION_TESTING.md)** - Testing changes
   - Test infrastructure changes
   - Test fixture changes
   - Provider mock changes
   - Test migration examples
   - New testing patterns

6. **[Breaking Changes Checklist](./BREAKING_CHANGES.md)** - Complete breaking changes list
   - Critical breaking changes
   - Moderate breaking changes
   - Minor breaking changes
   - Migration checklist
   - Risk assessment

7. **[Rollback Guide](./ROLLBACK_GUIDE.md)** - How to rollback if needed
   - When to rollback
   - Pre-rollback preparation
   - Rollback procedures
   - Data migration considerations
   - Re-migration

## Migration Scripts

### 1. Configuration Migration Script

```bash
python scripts/migration/migrate_config.py --source ./old_config.py --dest ./victor/config
```

**Features**:
- Migrates Python config files to YAML
- Updates .env file with new variable names
- Validates migrated configuration
- Generates migration report

### 2. Workflow Migration Script

```bash
python scripts/migration/migrate_workflows.py --source ./old_workflows --dest ./victor/workflows
```

**Features**:
- Migrates Python workflows to YAML
- Updates workflow structure to 0.5.0 format
- Validates migrated workflows
- Preserves directory structure

### 3. Validation Script

```bash
python scripts/migration/validate_migration.py
```

**Features**:
- Validates imports
- Validates configuration
- Validates code patterns
- Checks dependencies
- Generates detailed report

## Key Changes Summary

### Critical Breaking Changes

1. **Orchestrator Initialization**: Must use `bootstrap_orchestrator()` instead of direct instantiation
2. **Provider API Keys**: No longer in constructor, use environment variables
3. **Tool Registry**: Use DI container instead of singleton
4. **Event Bus**: Use `create_event_backend()` instead of `EventBus()`
5. **Async Chat**: All chat calls must use `await`

### New Features

1. **Planning**: Automatic task decomposition and execution
2. **Memory**: Persistent memory with semantic search
3. **Skills**: Composable skills with discovery and execution
4. **Multimodal**: Image and document processing
5. **Personas**: Predefined and custom agent personalities

### Performance Improvements

1. **Initialization**: 95% faster through lazy loading
2. **Throughput**: 15-25% improvement through optimizations
3. **Tool Selection**: 24-37% latency reduction through caching

## Migration Checklist

Use this checklist to ensure a successful migration:

### Pre-Migration
- [ ] Read main migration guide
- [ ] Review breaking changes
- [ ] Backup code and configuration
- [ ] Create migration branch
- [ ] Run validation script

### Migration
- [ ] Update orchestrator initialization
- [ ] Update provider initialization
- [ ] Update tool registry access
- [ ] Update event bus usage
- [ ] Make chat calls async
- [ ] Update workflow execution
- [ ] Update imports
- [ ] Migrate configuration files
- [ ] Migrate workflow files

### Post-Migration
- [ ] Run validation script
- [ ] Run linter
- [ ] Run type checker
- [ ] Run unit tests
- [ ] Run integration tests
- [ ] Test basic functionality
- [ ] Monitor logs for errors

### Rollback Plan
- [ ] Document rollback procedure
- [ ] Keep backup of working state
- [ ] Test rollback process
- [ ] Have rollback timeline ready

## Common Issues

### Issue: Import Errors

**Solution**: Update imports to use canonical paths

```python
# Old
from victor.config import Config

# New
from victor.config.settings import Settings
```

### Issue: API Key Not Found

**Solution**: Move API keys to environment variables

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Issue: Orchestrator Creation Fails

**Solution**: Use bootstrap function

```python
from victor.core.bootstrap import bootstrap_orchestrator

orchestrator = bootstrap_orchestrator(Settings())
```

## Timeline Estimate

| Task | Estimated Time |
|------|---------------|
| Read documentation | 2-4 hours |
| Update code | 8-16 hours |
| Update tests | 4-8 hours |
| Migrate configuration | 1-2 hours |
| Migrate workflows | 2-4 hours |
| Validation | 2-4 hours |
| **Total** | **19-38 hours** |

## Support Resources

- **Documentation**: See guides above
- **GitHub Issues**: [https://github.com/vijayksingh/victor/issues](https://github.com/vijayksingh/victor/issues)
- **Discord**: Join our Discord server
- **Discussion Forum**: GitHub Discussions

## Tips for Success

1. **Start Early**: Don't wait until the last minute
2. **Test Thoroughly**: Run comprehensive tests after migration
3. **Go Gradually**: Migrate non-critical systems first
4. **Document Everything**: Keep detailed notes of changes
5. **Have a Rollback Plan**: Be prepared to rollback if needed
6. **Ask for Help**: Don't hesitate to reach out for support

## Version Compatibility

| Version | Status | Support Until |
|---------|--------|--------------|
| 0.5.x | Maintenance | 2025-06-01 |
| 0.5.0 | Current | 2026-01-01 |

## What's Next

After completing the migration:

1. **Explore new features**: Try planning, memory, skills, etc.
2. **Optimize configuration**: Tune settings for your use case
3. **Update documentation**: Document any custom changes
4. **Train team**: Share knowledge with your team
5. **Monitor performance**: Track improvements and issues

---

**Last Updated**: 2025-01-21
**Version**: 0.5.0

Need help? See the [Main Migration Guide](./MIGRATION_GUIDE.md) or [Rollback Guide](./ROLLBACK_GUIDE.md).
