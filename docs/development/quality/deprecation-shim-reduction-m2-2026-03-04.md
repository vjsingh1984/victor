# Deprecation Shim Reduction (M2, 2026-03-04)

Related tracker item: [90D][E5][M2] Remove 30% deprecated shims/constants (#49)

## Summary

Removed deprecated compatibility shims from three vertical tool-dependency modules:

- `victor.verticals.contrib.research.tool_dependencies`
- `victor.verticals.contrib.devops.tool_dependencies`
- `victor.verticals.contrib.dataanalysis.tool_dependencies`

## Removed Artifacts

- Deprecated wrapper provider classes:
  - `ResearchToolDependencyProvider`
  - `DevOpsToolDependencyProvider`
  - `DataAnalysisToolDependencyProvider`
- Legacy constant exports:
  - `RESEARCH_*`
  - `DEVOPS_*`
  - `DATA_ANALYSIS_*`

## Canonical Path

All three modules now expose canonical provider factory usage:

```python
from victor.verticals.contrib.devops.tool_dependencies import get_provider
provider = get_provider()
```

## Release Notes

Migration guidance was added in `CHANGELOG.md` under `Unreleased`.
