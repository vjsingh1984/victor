# Workflow Diagrams Index

**Last Updated**: 2026-01-18
**Total Diagrams**: 57
**Total Size**: ~1.2 MB

This index catalogs all workflow diagrams in Victor, organized by vertical/domain with metadata and maintenance recommendations.

---

## Quick Statistics

| Vertical | Count | Avg Size | Total Size |
|----------|-------|----------|------------|
| **Benchmark** | 11 | 13.5 KB | 149 KB |
| **Coding** | 12 | 16.3 KB | 195 KB |
| **Core** | 4 | 6.7 KB | 27 KB |
| **Data Analysis** | 8 | 17.2 KB | 138 KB |
| **DevOps** | 4 | 18.8 KB | 75 KB |
| **RAG** | 5 | 22.4 KB | 112 KB |
| **Research** | 5 | 14.4 KB | 72 KB |

---

## Table of Contents

- [Naming Convention](#naming-convention)
- [Benchmark Workflows](#benchmark-workflows)
- [Coding Workflows](#coding-workflows)
- [Core Workflows](#core-workflows)
- [Data Analysis Workflows](#data-analysis-workflows)
- [DevOps Workflows](#devops-workflows)
- [RAG Workflows](#rag-workflows)
- [Research Workflows](#research-workflows)
- [Duplicates and Variants](#duplicates-and-variants)
- [Consolidation Recommendations](#consolidation-recommendations)
- [Maintenance Guidelines](#maintenance-guidelines)

---

## Naming Convention

### Current Convention
```
{vertical}_{workflow_name}[_{variant}].svg
```

**Examples**:
- `coding_bug_fix.svg` - Full workflow
- `coding_bugfix.svg` - Alternative version (inconsistent!)
- `coding_quick_fix.svg` - Quick variant
- `benchmark_swe_bench_lite.svg` - Lite variant

### Issues Identified
1. **Inconsistent separators**: Some use `_` between compound words (`bug_fix`), others don't (`bugfix`)
2. **Duplicate naming**: `coding_bug_fix.svg` vs `coding_bugfix.svg` refer to different workflows
3. **Variant naming**: No consistent convention for quick/lite/full variants
4. **Missing metadata**: No YAML frontmatter or embedded descriptions

### Recommended Convention
```
{vertical}/{workflow_name}/{variant}.svg
```

**Directory Structure**:
```
docs/workflow-diagrams/
├── benchmark/
│   ├── swe_bench/
│   │   ├── full.svg
│   │   ├── lite.svg
│   │   └── README.md
│   └── code_generation/
│       ├── full.svg
│       └── simple.svg
├── coding/
│   ├── bug_fix/
│   │   ├── full.svg
│   │   └── quick.svg
│   └── code_review/
│       ├── full.svg
│       └── quick.svg
...
```

**File naming**:
- `{workflow_name}-full.svg` - Complete workflow with all steps
- `{workflow_name}-quick.svg` - Simplified workflow for common cases
- `{workflow_name}-lite.svg` - Resource-constrained variant
- `{workflow_name}-v{major}.{minor}.svg` - Versioned workflows

**Metadata Header** (SVG comment):
```xml
<!--
Workflow: bug_fix
Vertical: coding
Variant: full
Version: 1.0
Description: Investigate, diagnose, and fix bugs with human approval gates
Last Updated: 2026-01-18
Nodes: 8
Edges: 12
-->
```

---

## Benchmark Workflows

**Path**: `docs/workflow-diagrams/benchmark/`

| File | Size | Description | Nodes | Status |
|------|------|-------------|-------|--------|
| `benchmark_aider_polyglot.svg` | 13 KB | Aider polyglot code generation benchmark | ~8 | ✅ Active |
| `benchmark_big_code_bench.svg` | 16 KB | BigCodeBench evaluation workflow | ~10 | ✅ Active |
| `benchmark_code_generation.svg` | 14 KB | General code generation benchmark | ~12 | ✅ Active |
| `benchmark_code_generation_simple.svg` | 3.6 KB | Simplified code generation | ~5 | 🔄 Duplicate |
| `benchmark_live_code_bench.svg` | 16 KB | LiveCodeBench evaluation | ~10 | ✅ Active |
| `benchmark_multi_function.svg` | 12 KB | Multi-function benchmark suite | ~8 | ✅ Active |
| `benchmark_passk_generation.svg` | 12 KB | Pass@k metric generation | ~8 | ✅ Active |
| `benchmark_passk_high.svg` | 10 KB | High-compute pass@k evaluation | ~6 | ⚠️ Variant |
| `benchmark_passk_refined.svg` | 10 KB | Refined pass@k methodology | ~6 | ⚠️ Variant |
| `benchmark_swe_bench.svg` | 26 KB | Full SWE-bench workflow | ~20 | ✅ Active |
| `benchmark_swe_bench_lite.svg` | 7.4 KB | Lightweight SWE-bench | ~8 | 🔄 Variant |

**Total**: 11 diagrams

**Key Workflows**:
- **SWE-Bench**: Full (26 KB) vs Lite (7.4 KB) variants
- **Pass@K**: Three variants (generation, high, refined)
- **Code Generation**: General vs Simple versions

---

## Coding Workflows

**Path**: `docs/workflow-diagrams/coding/`

| File | Size | Description | Nodes | Status |
|------|------|-------------|-------|--------|
| `coding_bug_fix.svg` | 19 KB | Comprehensive bug fix workflow (8 nodes) | 8 | ✅ Active |
| `coding_bugfix.svg` | 7.4 KB | Alternative bug fix (different structure) | 4 | ⚠️ Conflicting |
| `coding_code_review.svg` | 34 KB | Full code review workflow | ~15 | ✅ Active |
| `coding_debug_investigation.svg` | 5.6 KB | Debug investigation | ~5 | ✅ Active |
| `coding_extract.svg` | 8.2 KB | Code extraction/refactoring | ~6 | ✅ Active |
| `coding_feature_implementation.svg` | 30 KB | Feature development workflow | ~12 | ✅ Active |
| `coding_optimize.svg` | 9.4 KB | Code optimization | ~6 | ✅ Active |
| `coding_pr_review.svg` | 12 KB | Pull request review | ~8 | ✅ Active |
| `coding_quick_fix.svg` | 4.7 KB | Quick bug fix variant | ~4 | 🔄 Variant |
| `coding_quick_review.svg` | 5.8 KB | Quick code review | ~4 | 🔄 Variant |
| `coding_refactor.svg` | 28 KB | Refactoring workflow | ~12 | ✅ Active |
| `coding_rename.svg` | 8.1 KB | Symbol renaming | ~6 | ✅ Active |
| `coding_tdd.svg` | 34 KB | Test-driven development | ~15 | ✅ Active |
| `coding_tdd_quick.svg` | 8.9 KB | Quick TDD cycle | ~6 | 🔄 Variant |

**Total**: 14 diagrams (includes 2 conflicts)

**Key Workflows**:
- **Bug Fix**: Full (19 KB) vs Quick (4.7 KB) vs Conflicting `bugfix` (7.4 KB)
- **Code Review**: Full (34 KB) vs Quick (5.8 KB)
- **TDD**: Full (34 KB) vs Quick (8.9 KB)
- **Major Workflows**: Feature Implementation (30 KB), Refactor (28 KB)

---

## Core Workflows

**Path**: `docs/workflow-diagrams/core/`

| File | Size | Description | Nodes | Status |
|------|------|-------------|-------|--------|
| `core_build.svg` | 7.5 KB | Build mode workflow | ~6 | ✅ Active |
| `core_explore.svg` | 3.8 KB | Explore mode workflow | ~4 | ✅ Active |
| `core_explore_plan_build.svg` | 8.3 KB | Combined mode workflow | ~8 | ✅ Active |
| `core_plan.svg` | 7.3 KB | Plan mode workflow | ~6 | ✅ Active |

**Total**: 4 diagrams

**Description**: Core agent mode workflows (BUILD, PLAN, EXPLORE) and combined modes.

---

## Data Analysis Workflows

**Path**: `docs/workflow-diagrams/dataanalysis/`

| File | Size | Description | Nodes | Status |
|------|------|-------------|-------|--------|
| `dataanalysis_automl.svg` | 42 KB | Complete AutoML pipeline | ~18 | ✅ Active |
| `dataanalysis_automl_quick.svg` | 4.1 KB | Quick AutoML | ~4 | 🔄 Variant |
| `dataanalysis_data_cleaning.svg` | 15 KB | Data cleaning pipeline | ~8 | ✅ Active |
| `dataanalysis_data_cleaning_quick.svg` | 4.0 KB | Quick data cleaning | ~4 | 🔄 Variant |
| `dataanalysis_eda_pipeline.svg` | 44 KB | Exploratory data analysis | ~20 | ✅ Active |
| `dataanalysis_eda_quick.svg` | 6.8 KB | Quick EDA | ~6 | 🔄 Variant |
| `dataanalysis_ml_pipeline.svg` | 41 KB | ML training pipeline | ~18 | ✅ Active |
| `dataanalysis_ml_quick.svg` | 5.1 KB | Quick ML | ~4 | 🔄 Variant |
| `dataanalysis_rl_training.svg` | 3.9 KB | RL training workflow | ~4 | ✅ Active |
| `dataanalysis_statistical_analysis.svg` | 15 KB | Statistical analysis | ~8 | ✅ Active |

**Total**: 10 diagrams

**Pattern**: Every major workflow has a "quick" variant:
- AutoML (42 KB) → Quick (4.1 KB)
- Data Cleaning (15 KB) → Quick (4.0 KB)
- EDA (44 KB) → Quick (6.8 KB)
- ML Pipeline (41 KB) → Quick (5.1 KB)

---

## DevOps Workflows

**Path**: `docs/workflow-diagrams/devops/`

| File | Size | Description | Nodes | Status |
|------|------|-------------|-------|--------|
| `devops_cicd.svg` | 13 KB | CI/CD pipeline workflow | ~8 | ✅ Active |
| `devops_container_quick.svg` | 4.1 KB | Quick container setup | ~4 | 🔄 Variant |
| `devops_container_setup.svg` | 23 KB | Complete container setup | ~10 | ✅ Active |
| `devops_deploy.svg` | 35 KB | Deployment workflow | ~15 | ✅ Active |

**Total**: 4 diagrams

**Key Workflows**:
- Container Setup: Full (23 KB) vs Quick (4.1 KB)
- Deployment: Large comprehensive workflow (35 KB)
- CI/CD: Standard pipeline (13 KB)

---

## RAG Workflows

**Path**: `docs/workflow-diagrams/rag/`

| File | Size | Description | Nodes | Status |
|------|------|-------------|-------|--------|
| `rag_agentic_rag.svg` | 5.7 KB | Agentic RAG workflow | ~6 | ✅ Active |
| `rag_conversation.svg` | 11 KB | RAG conversation flow | ~8 | ✅ Active |
| `rag_document_ingest.svg` | 49 KB | Document ingestion pipeline | ~25 | ✅ Active |
| `rag_incremental_update.svg` | 11 KB | Incremental index updates | ~8 | ✅ Active |
| `rag_maintenance.svg` | 11 KB | Index maintenance | ~8 | ✅ Active |
| `rag_rag_query.svg` | 35 KB | RAG query workflow | ~15 | ✅ Active |

**Total**: 6 diagrams

**Largest Workflow**: Document Ingestion (49 KB, ~25 nodes)

**Key Areas**:
- Ingestion: Document processing and indexing
- Query: Retrieval and generation
- Maintenance: Incremental updates and index maintenance
- Agentic: Multi-step reasoning with RAG

---

## Research Workflows

**Path**: `docs/workflow-diagrams/research/`

| File | Size | Description | Nodes | Status |
|------|------|-------------|-------|--------|
| `research_competitive_analysis.svg` | 17 KB | Competitive analysis | ~10 | ✅ Active |
| `research_competitive_scan.svg` | 2.6 KB | Quick competitive scan | ~3 | 🔄 Variant |
| `research_deep_research.svg` | 23 KB | Deep research workflow | ~12 | ✅ Active |
| `research_fact_check.svg` | 18 KB | Fact checking | ~10 | ✅ Active |
| `research_literature_review.svg` | 21 KB | Literature review | ~12 | ✅ Active |
| `research_quick_research.svg` | 2.5 KB | Quick research | ~3 | 🔄 Variant |

**Total**: 6 diagrams

**Pattern**: Quick variants are ~7-10x smaller:
- Competitive Analysis (17 KB) → Scan (2.6 KB)
- Deep Research (23 KB) → Quick Research (2.5 KB)

---

## Duplicates and Variants

### Conflicting Names ⚠️

| Issue | Files | Recommendation |
|-------|-------|----------------|
| **Bug Fix Naming** | `coding_bug_fix.svg` (19 KB) vs `coding_bugfix.svg` (7.4 KB) | Rename `coding_bugfix.svg` → `coding_bug_fix_alt.svg` or merge |
| **Competitive Analysis** | `research_competitive_analysis.svg` (17 KB) vs `research_competitive_scan.svg` (2.6 KB) | Standardize to `competitive_analysis-full.svg` and `competitive_analysis-quick.svg` |

### Quick Variants 🔄

| Full | Quick | Size Ratio | Status |
|------|-------|------------|--------|
| `coding_bug_fix.svg` (19 KB) | `coding_quick_fix.svg` (4.7 KB) | 4:1 | ✅ Good |
| `coding_code_review.svg` (34 KB) | `coding_quick_review.svg` (5.8 KB) | 6:1 | ✅ Good |
| `coding_tdd.svg` (34 KB) | `coding_tdd_quick.svg` (8.9 KB) | 4:1 | ✅ Good |
| `dataanalysis_automl.svg` (42 KB) | `dataanalysis_automl_quick.svg` (4.1 KB) | 10:1 | ✅ Good |
| `dataanalysis_data_cleaning.svg` (15 KB) | `dataanalysis_data_cleaning_quick.svg` (4.0 KB) | 4:1 | ✅ Good |
| `dataanalysis_eda_pipeline.svg` (44 KB) | `dataanalysis_eda_quick.svg` (6.8 KB) | 6:1 | ✅ Good |
| `dataanalysis_ml_pipeline.svg` (41 KB) | `dataanalysis_ml_quick.svg` (5.1 KB) | 8:1 | ✅ Good |
| `devops_container_setup.svg` (23 KB) | `devops_container_quick.svg` (4.1 KB) | 6:1 | ✅ Good |
| `research_deep_research.svg` (23 KB) | `research_quick_research.svg` (2.5 KB) | 9:1 | ✅ Good |
| `benchmark_swe_bench.svg` (26 KB) | `benchmark_swe_bench_lite.svg` (7.4 KB) | 4:1 | ✅ Good |

### Benchmark Variants ⚠️

| Variants | Issue |
|----------|-------|
| `benchmark_passk_high.svg` vs `benchmark_passk_refined.svg` vs `benchmark_passk_generation.svg` | Three different pass@k workflows - unclear naming |
| `benchmark_code_generation.svg` vs `benchmark_code_generation_simple.svg` | "Simple" vs "Full" naming inconsistency |

---

## Consolidation Recommendations

### Phase 1: Immediate Actions (Week 1)

1. **Resolve Conflicting Names**
   - [ ] Investigate `coding_bug_fix.svg` vs `coding_bugfix.svg`
   - [ ] Determine which is current, merge or rename
   - [ ] Update all references in codebase

2. **Standardize Variant Naming**
   - [ ] Rename `*_quick.svg` → `*-quick.svg`
   - [ ] Rename `*_lite.svg` → `*-lite.svg`
   - [ ] Rename `*_simple.svg` → `*-simple.svg`

3. **Add Metadata Headers**
   - [ ] Add XML comment headers to all 57 diagrams
   - [ ] Include: workflow name, vertical, variant, version, description

### Phase 2: Directory Reorganization (Week 2-3)

1. **Create Vertical Subdirectories**
   ```bash
   mkdir -p benchmark coding core dataanalysis devops rag research
   ```

2. **Move Files to Vertical Directories**
   ```bash
   # Example
   mv coding_*.svg coding/
   mv benchmark_*.svg benchmark/
   ```

3. **Create README.md per Vertical**
   - [ ] `coding/README.md` - Coding workflows overview
   - [ ] `benchmark/README.md` - Benchmark workflows overview
   - [ ] etc.

### Phase 3: Workflow Consolidation (Week 3-4)

1. **Merge Similar Workflows**
   - [ ] Audit `coding_bug_fix.svg` vs `coding_bugfix.svg`
   - [ ] Consolidate `benchmark_passk_*.svg` variants
   - [ ] Review `research_competitive_*.svg` for merge

2. **Create Workflow Families**
   - [ ] Document relationship between full/quick variants
   - [ ] Create README.md for each workflow family
   - [ ] Add decision tree: when to use which variant

3. **Remove Obsolete Diagrams**
   - [ ] Identify workflows no longer used
   - [ ] Move to `archive/` instead of deleting
   - [ ] Document reason for deprecation

### Phase 4: Documentation and Tooling (Week 4+)

1. **Generate SVG Metadata Tool**
   ```python
   # scripts/workflow_metadata.py
   # Extract nodes, edges, depth from SVG
   # Generate markdown index
   # Validate naming convention
   ```

2. **Create Workflow Catalog**
   - [ ] Generate `docs/workflow-diagrams/CATALOG.md`
   - [ ] Include visual thumbnails
   - [ ] Add workflow descriptions and use cases

3. **Setup Automated Validation**
   - [ ] Pre-commit hook for naming convention
   - [ ] CI check for metadata headers
   - [ ] Automated index generation

---

## Maintenance Guidelines

### Adding New Workflows

1. **Follow Naming Convention**
   ```
   {vertical}/{workflow_name}-{variant}.svg
   ```

2. **Add Metadata Header**
   ```xml
   <!--
   Workflow: {name}
   Vertical: {vertical}
   Variant: {full|quick|lite|custom}
   Version: 1.0
   Description: {1-2 sentences}
   Author: {name}
   Last Updated: {YYYY-MM-DD}
   Nodes: {count}
   Edges: {count}
   -->
   ```

3. **Update This Index**
   - Add entry to appropriate vertical section
   - Update statistics
   - Increment total count

4. **Create Variant if Appropriate**
   - If workflow >20 nodes, consider creating "quick" variant
   - Quick variant should be <10 nodes
   - Document relationship in README

### Modifying Existing Workflows

1. **Version Control**
   - Increment version in metadata header
   - Add changelog to vertical README
   - Keep previous version in `archive/` if breaking change

2. **Validation**
   - Ensure SVG is valid XML
   - Check all nodes have labels
   - Verify edges connect properly
   - Test with SVG renderer

3. **Documentation**
   - Update description if workflow logic changes
   - Note complexity changes (node/edge count)
   - Update this index

### Removing Workflows

1. **Deprecation Process**
   - Move to `archive/{vertical}/{workflow}-{version}.svg`
   - Add to `archive/DEPRECATED.md` with reason
   - Update code references

2. **Do Not Delete**
   - Keep archive for historical reference
   - Document migration path to new workflow
   - Update all references in codebase

---

## Statistics Summary

### File Size Distribution

| Size Range | Count | Percentage |
|------------|-------|------------|
| <5 KB | 11 | 19% |
| 5-10 KB | 16 | 28% |
| 10-20 KB | 18 | 32% |
| 20-30 KB | 7 | 12% |
| >30 KB | 5 | 9% |

**Largest Files**:
1. `rag_document_ingest.svg` - 49 KB
2. `dataanalysis_eda_pipeline.svg` - 44 KB
3. `dataanalysis_automl.svg` - 42 KB
4. `dataanalysis_ml_pipeline.svg` - 41 KB
5. `devops_deploy.svg` - 35 KB
6. `coding_code_review.svg` - 34 KB
7. `coding_tdd.svg` - 34 KB

### Variant Distribution

| Variant Type | Count | Verticals |
|--------------|-------|-----------|
| Quick | 10 | coding (3), dataanalysis (4), devops (1), research (2) |
| Lite | 1 | benchmark (1) |
| Simple | 1 | benchmark (1) |
| Full/Standard | 45 | All verticals |

### Naming Convention Compliance

| Pattern | Count | Status |
|---------|-------|--------|
| `{vertical}_{workflow}.svg` | 34 | ✅ Compliant |
| `{vertical}_{workflow}_quick.svg` | 10 | ✅ Compliant |
| `{vertical}_{workflow}_lite.svg` | 1 | ⚠️ Inconsistent (should be quick) |
| `{vertical}_{workflow}_simple.svg` | 1 | ⚠️ Inconsistent (should be quick) |
| `coding_bugfix.svg` | 1 | ❌ Non-standard (no underscore) |

---

## Action Items Summary

### High Priority 🔴
- [ ] Resolve `coding_bug_fix.svg` vs `coding_bugfix.svg` conflict
- [ ] Standardize variant naming (`quick` vs `lite` vs `simple`)
- [ ] Add metadata headers to all diagrams

### Medium Priority 🟡
- [ ] Reorganize into vertical subdirectories
- [ ] Create README.md for each vertical
- [ ] Consolidate benchmark pass@k variants
- [ ] Archive deprecated workflows

### Low Priority 🟢
- [ ] Generate workflow catalog with thumbnails
- [ ] Create automated validation tooling
- [ ] Setup pre-commit hooks for naming convention
- [ ] Add SVG complexity metrics

---

## Related Documentation

- **Workflow Architecture**: `docs/architecture/workflows.md`
- **StateGraph DSL**: `victor/framework/graph.py`
- **YAML Workflows**: `victor/workflows/unified_compiler.py`
- **Vertical Base Classes**: `victor/core/verticals/base.py`

---

## Change Log

### 2026-01-18
- Initial comprehensive index created
- Cataloged 57 workflow diagrams across 7 verticals
- Identified naming conflicts and variant patterns
- Established naming convention and maintenance guidelines
- Created consolidation roadmap

---

**Maintainers**: Victor Architecture Team
**Last Review**: 2026-01-18
**Next Review**: 2026-02-18
