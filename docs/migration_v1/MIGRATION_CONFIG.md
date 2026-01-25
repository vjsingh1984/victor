# Configuration Migration Guide: Victor 0.5.x to 0.5.0

This guide explains how to migrate your configuration from Victor 0.5.x to 0.5.0.

## Table of Contents

1. [Overview](#overview)
2. [Settings File Changes](#settings-file-changes)
3. [Environment Variable Changes](#environment-variable-changes)
4. [YAML Configuration Changes](#yaml-configuration-changes)
5. [Provider Configuration](#provider-configuration)
6. [Feature Flags](#feature-flags)
7. [Migration Examples](#migration-examples)

---

## Overview

### Configuration Changes Summary

- **Settings API**: `Config` → `Settings` (Pydantic v2)
- **API Keys**: No longer in constructor, use environment variables
- **YAML Configs**: New YAML-first configuration system
- **Feature Flags**: New flag naming and defaults
- **Provider Configs**: Centralized in `model_capabilities.yaml`

---

## Settings File Changes

### 1. Python Settings File

**Before (0.5.x)**:
```python
# victor_config.py
from victor.config import Config

config = Config()
config.max_tokens = 4096
config.temperature = 0.7
config.tool_budget = 100
config.use_semantic_tool_selection = True
```

**After (0.5.0)**:
```python
# victor_settings.py
from victor.config.settings import Settings

settings = Settings(
    max_tokens=4096,
    temperature=0.7,
    tool_budget=100,
    tool_selection_strategy="hybrid",  # Changed from use_semantic_tool_selection
)
```

### 2. Environment-Based Configuration

**Before (0.5.x)**:
```python
# Manual environment variable handling
import os

class Config:
    def __init__(self):
        self.api_key = os.getenv("VICTOR_API_KEY", "")
        self.max_tokens = int(os.getenv("VICTOR_MAX_TOKENS", "2000"))
```

**After (0.5.0)**:
```python
# Automatic via pydantic-settings
from victor.config.settings import Settings

# Environment variables automatically loaded
settings = Settings()

# Or override specific values
settings = Settings(
    max_tokens=4096,  # Override VICTOR_MAX_TOKENS
)
```

---

## Environment Variable Changes

### New Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `VICTOR_EVENT_BACKEND` | Event backend type | `in_memory` | `kafka`, `sqs`, `redis` |
| `VICTOR_TOOL_SELECTION_STRATEGY` | Tool selection strategy | `hybrid` | `keyword`, `semantic`, `hybrid` |
| `VICTOR_TOOL_CACHE_ENABLED` | Enable tool selection cache | `true` | `true`, `false` |
| `VICTOR_TOOL_CACHE_SIZE` | Tool selection cache size | `500` | `1000` |
| `VICTOR_MEMORY_ENABLED` | Enable memory capability | `false` | `true`, `false` |
| `VICTOR_MEMORY_BACKEND` | Memory storage backend | `lancedb` | `sqlite`, `postgres` |
| `VICTOR_PLANNING_ENABLED` | Enable planning capability | `true` | `true`, `false` |
| `VICTOR_SKILLS_ENABLED` | Enable skills system | `true` | `true`, `false` |
| `VICTOR_MULTIMODAL_ENABLED` | Enable multimodal support | `false` | `true`, `false` |
| `VICTOR_PROFILE` | Operational profile | `production` | `development`, `airgapped` |

### Changed Environment Variables

| Old Variable | New Variable | Change |
|--------------|--------------|--------|
| `VICTOR_TOOL_SELECTION` | `VICTOR_TOOL_SELECTION_STRATEGY` | Renamed |
| `VICTOR_CACHE_ENABLED` | `VICTOR_TOOL_CACHE_ENABLED` | More specific |
| `VICTOR_WORKFLOW_CACHE` | `VICTOR_WORKFLOW_CACHE_ENABLED` | Consistency |
| `VICTOR_SEMANTIC_SEARCH` | `VICTOR_TOOL_SELECTION_STRATEGY=semantic` | Replaced |

### Deprecated Environment Variables

| Variable | Status | Replacement |
|----------|--------|-------------|
| `VICTOR_API_KEY` | Deprecated | Provider-specific keys (e.g., `OPENAI_API_KEY`) |
| `VICTOR_USE_SEMANTIC` | Deprecated | `VICTOR_TOOL_SELECTION_STRATEGY` |
| `VICTOR_ENABLE_CACHE` | Deprecated | `VICTOR_TOOL_CACHE_ENABLED` |

---

## YAML Configuration Changes

### New YAML Configuration Structure

Victor 0.5.0 introduces YAML-first configuration:

```
victor/config/
├── modes/              # Mode configurations
│   ├── coding_modes.yaml
│   ├── devops_modes.yaml
│   └── ...
├── capabilities/       # Capability definitions
│   ├── coding_capabilities.yaml
│   ├── devops_capabilities.yaml
│   └── ...
├── teams/             # Team specifications
│   ├── coding_teams.yaml
│   └── ...
└── rl/                # RL configurations
    └── rl_configs.yaml
```

### Mode Configuration

**New in 0.5.0**:
```yaml
# victor/config/modes/coding_modes.yaml
vertical_name: coding
default_mode: build

modes:
  build:
    name: build
    display_name: Build
    exploration: standard
    edit_permission: full
    tool_budget_multiplier: 1.0
    max_iterations: 10

  plan:
    name: plan
    display_name: Plan
    exploration: thorough
    edit_permission: sandbox
    tool_budget_multiplier: 2.5
    max_iterations: 25

  explore:
    name: explore
    display_name: Explore
    exploration: comprehensive
    edit_permission: none
    tool_budget_multiplier: 3.0
    max_iterations: 30
```

### Capability Configuration

**New in 0.5.0**:
```yaml
# victor/config/capabilities/coding_capabilities.yaml
vertical_name: coding

capabilities:
  code_review:
    type: workflow
    description: "Review code for quality and security"
    enabled: true
    handler: "victor.coding.workflows:CodeReviewWorkflow"
    config:
      max_files: 50
      check_security: true

  ast_analysis:
    type: tool
    description: "Parse and analyze code structure"
    enabled: true

  complexity_analysis:
    type: validator
    description: "Analyze code complexity metrics"
    enabled: true
    config:
      max_cyclomatic: 10
      max_nesting: 5

handlers:
  quality_check: victor.coding.review:QualityChecker
  security_scan: victor.coding.security:SecurityScanner
```

### Team Configuration

**New in 0.5.0**:
```yaml
# victor/config/teams/coding_teams.yaml
teams:
  - name: code_review_team
    display_name: "Code Review Team"
    description: "Multi-agent team for comprehensive code review"
    formation: parallel
    communication_style: structured
    max_iterations: 5
    roles:
      - name: security_reviewer
        display_name: "Security Reviewer"
        description: "Focuses on security vulnerabilities"
        persona: "You are a security expert..."
        tool_categories: [security, analysis]
        capabilities: [security_scan, vulnerability_detection]

      - name: quality_reviewer
        display_name: "Quality Reviewer"
        description: "Focuses on code quality"
        persona: "You are a code quality expert..."
        tool_categories: [analysis, metrics]
        capabilities: [complexity_analysis, style_check]
```

---

## Provider Configuration

### Model Capabilities Configuration

**New in 0.5.0**:
```yaml
# victor/config/model_capabilities.yaml
models:
  "claude-sonnet-4-5*":
    training:
      tool_calling: true
      parallel_tool_calls: true
      streaming: true
    providers:
      anthropic:
        native_tool_calls: true
        parallel_tool_calls: true
        max_tokens: 8192
    context_limits:
      max_tokens: 200000
      max_output_tokens: 8192

  "gpt-4*":
    training:
      tool_calling: true
      parallel_tool_calls: true
    providers:
      openai:
        native_tool_calls: true
        parallel_tool_calls: true
        max_tokens: 4096
```

### Provider Context Limits

**New in 0.5.0**:
```yaml
# victor/config/provider_context_limits.yaml
providers:
  anthropic:
    models:
      claude-sonnet-4-5:
        max_tokens: 200000
        max_output_tokens: 8192

  openai:
    models:
      gpt-4:
        max_tokens: 128000
        max_output_tokens: 4096
```

---

## Feature Flags

### New Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `memory.enabled` | `false` | Enable memory capability |
| `planning.enabled` | `true` | Enable planning capability |
| `skills.enabled` | `true` | Enable skills system |
| `multimodal.enabled` | `false` | Enable multimodal support |
| `tool_selection.cache_enabled` | `true` | Enable tool selection caching |
| `workflows.cache_enabled` | `true` | Enable workflow caching |

### Configuration

**YAML**:
```yaml
# victor_config.yaml
features:
  memory:
    enabled: true
    backend: lancedb

  planning:
    enabled: true
    max_depth: 5

  skills:
    enabled: true
    auto_discovery: true
```

**Environment Variables**:
```bash
VICTOR_MEMORY_ENABLED=true
VICTOR_PLANNING_ENABLED=true
VICTOR_SKILLS_ENABLED=true
```

---

## Migration Examples

### Example 1: Simple Configuration Migration

**Before (0.5.x)**:
```python
# config.py
from victor.config import Config

config = Config()
config.provider = "openai"
config.model = "gpt-4"
config.api_key = "sk-..."
config.max_tokens = 4096
config.temperature = 0.7
config.use_semantic_tool_selection = True
```

**After (0.5.0)**:
```bash
# .env
OPENAI_API_KEY=sk-...
VICTOR_PROVIDER=openai
VICTOR_MODEL=gpt-4
VICTOR_MAX_TOKENS=4096
VICTOR_TEMPERATURE=0.7
VICTOR_TOOL_SELECTION_STRATEGY=hybrid
```

```python
# config.py
from victor.config.settings import Settings

settings = Settings()  # Loads from environment
```

### Example 2: Advanced Configuration Migration

**Before (0.5.x)**:
```python
# config.py
from victor.config import Config

config = Config()
config.provider = "anthropic"
config.model = "claude-sonnet-4-5"
config.api_key = "sk-ant-..."
config.max_tokens = 8192
config.tool_budget = 100
config.cache_enabled = True
config.workflow_cache_size = 100
config.enable_planning = True
config.enable_memory = False
```

**After (0.5.0)**:
```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
VICTOR_PROVIDER=anthropic
VICTOR_MODEL=claude-sonnet-4-5
VICTOR_MAX_TOKENS=8192
VICTOR_TOOL_BUDGET=100
VICTOR_TOOL_CACHE_ENABLED=true
VICTOR_TOOL_CACHE_SIZE=500
VICTOR_WORKFLOW_CACHE_ENABLED=true
VICTOR_PLANNING_ENABLED=true
VICTOR_MEMORY_ENABLED=false
```

```yaml
# victor/config/settings.yaml
agent:
  provider: anthropic
  model: claude-sonnet-4-5
  max_tokens: 8192
  tool_budget: 100

tool_selection:
  strategy: hybrid
  cache_enabled: true
  cache_size: 500

workflows:
  cache_enabled: true
  cache_size: 100

features:
  planning:
    enabled: true
  memory:
    enabled: false
```

### Example 3: Custom Mode Configuration

**New in 0.5.0**:
```yaml
# victor/config/modes/custom_modes.yaml
vertical_name: custom
default_mode: standard

modes:
  fast:
    name: fast
    display_name: Fast Mode
    exploration: minimal
    edit_permission: full
    tool_budget_multiplier: 0.5
    max_iterations: 5

  thorough:
    name: thorough
    display_name: Thorough Mode
    exploration: comprehensive
    edit_permission: full
    tool_budget_multiplier: 3.0
    max_iterations: 50
```

Usage:
```python
from victor.core.mode_config import ModeConfigRegistry

registry = ModeConfigRegistry.get_instance()
config = registry.load_config('custom')
mode = config.get_mode('thorough')
```

---

## Automated Migration

Use the migration script to automatically convert your configuration:

```bash
# Migrate configuration
python scripts/migrate_config.py \
    --source ./old_config.py \
    --dest ./victor/config

# Validate migrated configuration
python scripts/validate_config.py victor/config

# Export current configuration
victor config export > current_config.json

# Import configuration
victor config import < config.json
```

---

## Validation

Validate your configuration after migration:

```bash
# Validate all YAML files
python scripts/validate_config.py victor/config

# Validate specific file
victor-validate-config victor/config/modes/coding_modes.yaml

# Check for deprecated settings
victor-check-deprecated --config victor_config.yaml
```

---

## Troubleshooting

### Issue: Configuration Not Loading

**Symptom**: Settings not loading from environment variables

**Solution**:
```python
# Ensure .env file exists and is formatted correctly
# .env
OPENAI_API_KEY=sk-...
VICTOR_MAX_TOKENS=4096

# Load explicitly if needed
from dotenv import load_dotenv
load_dotenv()

from victor.config.settings import Settings
settings = Settings()
```

### Issue: YAML Validation Errors

**Symptom**: YAML file fails validation

**Solution**:
```bash
# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('victor/config/modes/coding_modes.yaml'))"

# Use validation script
python scripts/validate_config.py victor/config/modes/coding_modes.yaml
```

### Issue: Mode Not Found

**Symptom**: `ValueError: Mode 'custom' not found`

**Solution**:
```python
# Ensure mode is registered
from victor.core.mode_config import ModeConfigRegistry

registry = ModeConfigRegistry.get_instance()
config = registry.load_config('your_vertical')
modes = config.list_modes()
print(modes)  # Should include 'custom'
```

---

## Additional Resources

- [Main Migration Guide](./MIGRATION_GUIDE.md)
- [API Migration Guide](./MIGRATION_API.md)
- [Settings Reference](../reference/configuration/index.md)
- [YAML Configuration Guide](../reference/configuration/index.md)

---

**Last Updated**: 2025-01-21
**Version**: 0.5.0
