# YAML Workflow Migration Archive

This directory contains copies of the original Python workflow implementations
that were deprecated during the YAML-first workflow migration (January 2025).

## Background

The Victor workflow system was migrated from hybrid Python+YAML to a YAML-first
architecture where:

- **YAML workflows** define the workflow structure, nodes, and routing
- **Python escape hatches** handle complex conditions and transforms
- **Compute handlers** execute domain-specific operations

## Archived Files

### research/graph_workflows.py
Original Python implementations of:
- `create_deep_research_workflow()`
- `create_fact_check_workflow()`
- `create_literature_review_workflow()`
- ResearchGraphExecutor

**Replaced by:**
- `victor/research/workflows/deep_research.yaml`
- `victor/research/workflows/fact_check.yaml`
- `victor/research/workflows/literature_review.yaml`
- `victor/research/workflows/competitive_analysis.yaml`
- `victor/research/escape_hatches.py`

### devops/graph_workflows.py
Original Python implementations of:
- `create_deployment_workflow()`
- `create_container_workflow()`
- `create_cicd_workflow()`
- `create_security_audit_workflow()`
- DevOpsGraphExecutor

**Replaced by:**
- `victor/devops/workflows/deploy.yaml`
- `victor/devops/workflows/container_setup.yaml`
- `victor/devops/escape_hatches.py`

### dataanalysis/graph_workflows.py
Original Python implementations of:
- `create_eda_workflow()`
- `create_cleaning_workflow()`
- `create_ml_pipeline_workflow()`
- DataAnalysisGraphExecutor

**Replaced by:**
- `victor/dataanalysis/workflows/eda_pipeline.yaml`
- `victor/dataanalysis/workflows/ml_pipeline.yaml`
- `victor/dataanalysis/workflows/data_cleaning.yaml`
- `victor/dataanalysis/workflows/statistical_analysis.yaml`
- `victor/dataanalysis/escape_hatches.py`

## Backwards Compatibility

The original files are still present in the codebase and imported by the
workflow providers for backwards compatibility. The StateGraph-based workflows
remain functional for any code that depends on them.

New development should use the YAML workflow definitions.
