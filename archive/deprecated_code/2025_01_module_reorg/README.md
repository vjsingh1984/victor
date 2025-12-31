# Module Reorganization Archive (January 2025)

This directory contains the original flat module structure that was reorganized into a hierarchical structure for better organization and maintainability.

## Mapping

| Original Location | New Location |
|------------------|--------------|
| victor/graph/ | victor/storage/graph/ |
| victor/checkpoints/ | victor/storage/checkpoints/ |
| victor/memory/ | victor/storage/memory/ |
| victor/cache/ | victor/storage/cache/ |
| victor/vector_stores/ | victor/storage/vector_stores/ |
| victor/embeddings/ | victor/storage/embeddings/ |
| victor/state/ | victor/storage/state/ |
| victor/mcp/ | victor/integrations/mcp/ |
| victor/protocol/ | victor/integrations/protocol/ |
| victor/api/ | victor/integrations/api/ |
| victor/native/ | victor/processing/native/ |
| victor/merge/ | victor/processing/merge/ |
| victor/serialization/ | victor/processing/serialization/ |
| victor/editing/ | victor/processing/editing/ |
| victor/file_types/ | victor/processing/file_types/ |
| victor/debug/ | victor/observability/debug/ |
| victor/profiler/ | victor/observability/profiler/ |
| victor/analytics/ | victor/observability/analytics/ |
| victor/telemetry/ | victor/observability/telemetry/ |
| victor/pipeline/ | victor/observability/pipeline/ |
| victor/auth/ | victor/security/auth/ |
| victor/audit/ | victor/security/audit/ |

## Archive Structure

```
2025_01_module_reorg/
├── storage_flat/       # Original flat storage modules
│   ├── graph/
│   ├── checkpoints/
│   ├── memory/
│   ├── cache/
│   ├── vector_stores/
│   ├── embeddings/
│   └── state/
├── integrations_flat/  # Original flat integration modules
│   ├── mcp/
│   ├── protocol/
│   └── api/
├── processing_flat/    # Original flat processing modules
│   ├── native/
│   ├── merge/
│   ├── serialization/
│   ├── editing/
│   └── file_types/
├── observability_flat/ # Original flat observability modules
│   ├── debug/
│   ├── profiler/
│   ├── analytics/
│   ├── telemetry/
│   └── pipeline/
└── security_flat/      # Original flat security modules
    ├── auth/
    └── audit/
```

## Backward Compatibility

Re-export stubs have been created at the original locations to maintain backward compatibility:

```python
# Example: victor/graph/__init__.py
from victor.storage.graph import *
```

These stubs allow existing code to continue importing from the old locations while transitioning to the new structure.

## Migration Date

January 2025

## Reason

Reorganized flat module structure into hierarchical structure for:
- Better code organization and discoverability
- Clearer module boundaries and responsibilities
- Improved maintainability and navigation
- Logical grouping of related functionality
