# Victor Workflow Diagrams

This directory contains visual workflow diagrams for Victor AI coding assistant, organized by vertical/domain.

## Quick Navigation

- **[Full Index](index.md)** - Complete catalog of all 57 workflow diagrams with metadata
- **[Naming Convention](#naming-convention)** - Guidelines for creating new diagrams
- **[Maintenance Guide](#maintenance)** - How to update and maintain diagrams

## Statistics

- **Total Diagrams**: 57 SVG files
- **Total Size**: ~980 KB
- **Verticals**: 7 (Benchmark, Coding, Core, Data Analysis, DevOps, RAG, Research)
- **Variants**: 8 quick/lite/simplified variants

## Quick Reference

| Vertical | Diagrams | Largest | Quick Variants |
|----------|----------|---------|----------------|
| [Benchmark](#benchmark) | 11 | SWE-Bench (26 KB) | 1 lite |
| [Coding](#coding) | 13 | Code Review (34 KB) | 2 quick |
| [Core](#core) | 4 | Explore+Plan+Build (8 KB) | 0 |
| [Data Analysis](#data-analysis) | 10 | EDA Pipeline (44 KB) | 4 quick |
| [DevOps](#devops) | 4 | Deploy (35 KB) | 1 quick |
| [RAG](#rag) | 6 | Document Ingest (49 KB) | 0 |
| [Research](#research) | 6 | Deep Research (23 KB) | 1 quick |

## Verticals

### Benchmark (11 diagrams)

**Purpose**: Evaluation and benchmarking workflows

**Key Workflows**:
- `benchmark_swe_bench.svg` (26 KB) - Full SWE-bench evaluation
- `benchmark_swe_bench_lite.svg` (7 KB) - Lightweight SWE-bench
- `benchmark_code_generation.svg` (14 KB) - Code generation benchmark
- `benchmark_passk_generation.svg` (12 KB) - Pass@k metrics

**Variants**:
- Lite: `benchmark_swe_bench_lite.svg`
- Simple: `benchmark_code_generation_simple.svg`

### Coding (13 diagrams)

**Purpose**: Code development, review, and maintenance

**Key Workflows**:
- `coding_code_review.svg` (34 KB) - Comprehensive code review
- `coding_tdd.svg` (34 KB) - Test-driven development
- `coding_feature_implementation.svg` (30 KB) - Feature development
- `coding_refactor.svg` (28 KB) - Refactoring

**Quick Variants**:
- `coding_quick_fix.svg` (5 KB) - Quick bug fixes
- `coding_quick_review.svg` (6 KB) - Quick code review
- `coding_tdd_quick.svg` (9 KB) - Quick TDD cycle

**⚠️ Known Issues**:
- `coding_bug_fix.svg` vs `coding_bugfix.svg` - Naming conflict, needs resolution

### Core (4 diagrams)

**Purpose**: Core agent modes and coordination

**Workflows**:
- `core_build.svg` (7.5 KB) - Build mode
- `core_plan.svg` (7.3 KB) - Plan mode
- `core_explore.svg` (3.8 KB) - Explore mode
- `core_explore_plan_build.svg` (8.3 KB) - Combined modes

### Data Analysis (10 diagrams)

**Purpose**: Data processing, ML, and analytics

**Key Workflows**:
- `dataanalysis_eda_pipeline.svg` (44 KB) - Exploratory data analysis
- `dataanalysis_automl.svg` (42 KB) - AutoML pipeline
- `dataanalysis_ml_pipeline.svg` (41 KB) - ML training pipeline

**Quick Variants** (all ~4-6 KB):
- `dataanalysis_eda_quick.svg`
- `dataanalysis_automl_quick.svg`
- `dataanalysis_ml_quick.svg`
- `dataanalysis_data_cleaning_quick.svg`

### DevOps (4 diagrams)

**Purpose**: DevOps and deployment workflows

**Workflows**:
- `devops_deploy.svg` (35 KB) - Deployment workflow
- `devops_container_setup.svg` (23 KB) - Container configuration
- `devops_cicd.svg` (13 KB) - CI/CD pipeline

**Quick Variants**:
- `devops_container_quick.svg` (4 KB)

### RAG (6 diagrams)

**Purpose**: Retrieval-Augmented Generation workflows

**Key Workflows**:
- `rag_document_ingest.svg` (49 KB) - Document ingestion pipeline
- `rag_rag_query.svg` (35 KB) - RAG query workflow
- `rag_conversation.svg` (11 KB) - RAG conversation flow

**Specialized**:
- `rag_agentic_rag.svg` (6 KB) - Agentic RAG
- `rag_incremental_update.svg` (11 KB) - Incremental updates
- `rag_maintenance.svg` (11 KB) - Index maintenance

### Research (6 diagrams)

**Purpose**: Research and knowledge gathering

**Key Workflows**:
- `research_deep_research.svg` (23 KB) - Deep research
- `research_literature_review.svg` (21 KB) - Literature review
- `research_fact_check.svg` (18 KB) - Fact checking
- `research_competitive_analysis.svg` (17 KB) - Competitive analysis

**Quick Variants**:
- `research_quick_research.svg` (2.5 KB)
- `research_competitive_scan.svg` (2.6 KB)

## Naming Convention

### Current Format
```
{vertical}_{workflow_name}[_{variant}].svg
```

**Examples**:
- `coding_bug_fix.svg` - Full workflow
- `coding_quick_fix.svg` - Quick variant
- `benchmark_swe_bench_lite.svg` - Lite variant

### Recommended Format (Future)
```
{vertical}/{workflow_name}-{variant}.svg
```

**Directory Structure**:
```
docs/workflow-diagrams/
├── coding/
│   ├── bug_fix/
│   │   ├── full.svg
│   │   └── quick.svg
│   └── code_review/
│       └── full.svg
└── benchmark/
    └── swe_bench/
        ├── full.svg
        └── lite.svg
```

### Variant Types

| Suffix | Purpose | Target Size |
|--------|---------|-------------|
| `full` (implicit) | Complete workflow | Any size |
| `quick` | Simplified for common cases | <10 nodes |
| `lite` | Resource-constrained | <10 nodes |
| `simple` | Simplified variant | <10 nodes |

**Note**: Current naming uses `quick`, `lite`, and `simple` inconsistently. Future convention will standardize on `quick` for all simplified variants.

## Metadata Format

All diagrams should include XML metadata header:

```xml
<!--
Workflow: bug_fix
Vertical: coding
Variant: full
Version: 1.0
Description: Investigate, diagnose, and fix bugs with human approval gates
Author: Your Name
Last Updated: 2026-01-18
Nodes: 8
Edges: 12
-->
```

## Maintenance

### Viewing Diagrams

SVG diagrams can be viewed in:
- Browser: Drag and drop or use `file://` URL
- VS Code: Install "SVG Preview" extension
- CLI: `svgexport` or `inkscape` tools

### Adding New Diagrams

1. **Create diagram** using your preferred tool (draw.io, Graphviz, etc.)
2. **Export as SVG** with embedded fonts
3. **Add metadata header** (see above)
4. **Follow naming convention**: `{vertical}_{workflow}[_{variant}].svg`
5. **Update index.md**: Add to appropriate vertical section
6. **Run validation**: `./scripts/generate_index.sh`

### Modifying Existing Diagrams

1. **Check version** in metadata header
2. **Increment version** if significant changes
3. **Archive old version** to an external archive repo if breaking changes
4. **Update metadata** with new date and description
5. **Update index.md** if node/edge count changes significantly

### Validation

Run the index generator script to validate:

```bash
cd docs/workflow-diagrams
./scripts/generate_index.sh
```

This will:
- Count total diagrams
- Show vertical breakdown
- List largest/smallest files
- Identify potential naming issues
- Suggest workflows needing quick variants

### Generating Thumbnails

To generate PNG thumbnails for web display:

```bash
# Requires svgexport or inkscape
for f in *.svg; do
    svgexport "$f" "${f%.svg}.png" 400:300
done
```

## Scripts

### `scripts/generate_index.sh`

Generates statistics report for all workflow diagrams.

```bash
./scripts/generate_index.sh
```

Output:
- Total diagram count
- Vertical breakdown
- Variant statistics
- Largest/smallest files
- Potential naming issues
- Suggestions for missing quick variants

## Known Issues

### High Priority 🔴

1. **Bug Fix Naming Conflict**
   - Files: `coding_bug_fix.svg` (19 KB) vs `coding_bugfix.svg` (7.4 KB)
   - Issue: Different workflows with nearly identical names
   - Resolution: Investigate and merge or rename

2. **Inconsistent Variant Naming**
   - Files: Mix of `quick`, `lite`, `simple` suffixes
   - Issue: No clear distinction between variant types
   - Resolution: Standardize on `quick` for all simplified variants

### Medium Priority 🟡

1. **Missing Quick Variants**
   - 16 workflows >15 KB without quick variants
   - Suggestion: Create quick variants for large workflows

2. **Missing Metadata**
   - Most diagrams lack metadata headers
   - Resolution: Add XML comments to all diagrams

### Low Priority 🟢

1. **Directory Organization**
   - Current: Flat structure with 57 files
   - Suggested: Organize into vertical subdirectories

2. **Thumbnails**
   - No PNG thumbnails for web display
   - Suggestion: Generate thumbnails for catalog

## Related Documentation

- **[Full Index](index.md)** - Complete catalog with metadata
- **[Workflow Architecture](../architecture/workflows.md)** - Workflow system documentation
- **[StateGraph DSL](../../victor/framework/graph.py)** - Workflow DSL implementation
- **[YAML Workflows](../../victor/workflows/unified_compiler.py)** - YAML workflow compiler

## Contributing

When adding new workflow diagrams:

1. Follow naming convention
2. Add metadata header
3. Keep file size reasonable (<50 KB for full workflows)
4. Consider creating quick variant if >20 nodes
5. Update this README and index.md
6. Run validation script

## License

Same as Victor project (MIT)

---

**Last Updated**: 2026-01-18
**Maintained By**: Victor Architecture Team
