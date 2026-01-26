#!/usr/bin/env python3
"""Analyze coordinator overlap and consolidation opportunities.

This script identifies:
1. Coordinators we created in Phase 1
2. Coordinators that already exist and are being used
3. Overlap/duplication between them
4. Consolidation recommendations
"""

import ast
import inspect
from pathlib import Path

COORDINATORS_DIR = Path("/Users/vijaysingh/code/codingagent/victor/agent/coordinators")

# Coordinators we created in Phase 1 (not yet integrated)
PHASE_1_COORDINATORS = [
    "ConfigCoordinator",
    "PromptCoordinator",
    "ContextCoordinator",
    "AnalyticsCoordinator",
]

# Existing coordinators (being used in orchestrator)
EXISTING_COORDINATORS = [
    "EvaluationCoordinator",
    "MetricsCoordinator",
    "WorkflowCoordinator",
    "CheckpointCoordinator",
    "ChatCoordinator",
    "ProviderCoordinator",
    "SessionCoordinator",
    "ToolCoordinator",
    "ToolSelectionCoordinator",
]


def analyze_coordinator_file(filename):
    """Analyze a coordinator file to extract its purpose and methods."""
    filepath = COORDINATORS_DIR / filename

    if not filepath.exists():
        return None

    with open(filepath, "r") as f:
        content = f.read()

    # Parse docstring for purpose
    try:
        tree = ast.parse(content)
        docstring = ast.get_docstring(tree)

        # Extract class names
        classes = []
        methods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
                # Extract methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(item.name)

        return {
            "file": filename,
            "classes": classes,
            "methods": methods,
            "docstring": docstring,
        }
    except:
        return {
            "file": filename,
            "classes": [],
            "methods": [],
            "docstring": None,
        }


def main():
    """Main analysis."""
    print("=" * 80)
    print("COORDINATOR CONSOLIDATION ANALYSIS")
    print("=" * 80)

    # Analyze all coordinator files
    all_files = [
        "analytics_coordinator.py",
        "config_coordinator.py",
        "context_coordinator.py",
        "prompt_coordinator.py",
        "evaluation_coordinator.py",
        "metrics_coordinator.py",
        "workflow_coordinator.py",
        "checkpoint_coordinator.py",
        "chat_coordinator.py",
    ]

    analysis = {}
    for filename in all_files:
        result = analyze_coordinator_file(filename)
        if result:
            analysis[filename.replace(".py", "")] = result

    print("\n## PHASE 1 COORDINATORS (Created but NOT Integrated)\n")
    for coord in PHASE_1_COORDINATORS:
        file_key = coord.lower() + "_coordinator"
        if file_key in analysis:
            info = analysis[file_key]
            print(f"\n{coord}:")
            print(f"  File: {info['file']}")
            print(f"  Classes: {', '.join(info['classes'])}")
            print(f"  Methods: {len(info['methods'])}")
            if info["docstring"]:
                lines = info["docstring"].split("\n")[:5]
                print(f"  Purpose: {lines[0] if lines else 'N/A'}")

    print("\n\n## EXISTING COORDINATORS (Being Used)\n")
    existing_map = {
        "EvaluationCoordinator": "evaluation_coordinator",
        "MetricsCoordinator": "metrics_coordinator",
        "WorkflowCoordinator": "workflow_coordinator",
        "CheckpointCoordinator": "checkpoint_coordinator",
    }

    for coord in existing_map.keys():
        file_key = existing_map[coord]
        if file_key in analysis:
            info = analysis[file_key]
            print(f"\n{coord}:")
            print(f"  Classes: {', '.join(info['classes'])}")
            print(f"  Methods: {len(info['methods'])}")
            if info["docstring"]:
                lines = info["docstring"].split("\n")[:5]
                print(f"  Purpose: {lines[0] if lines else 'N/A'}")

    print("\n\n## CONSOLIDATION RECOMMENDATIONS\n")

    print(
        """
### 1. AnalyticsCoordinator vs EvaluationCoordinator/MetricsCoordinator

**Analysis:**
- AnalyticsCoordinator: General analytics collection/export with multiple exporters
- EvaluationCoordinator: RL feedback, usage analytics, intelligent outcomes
- MetricsCoordinator: Metrics collection and export

**Recommendation:** KEEP SEPARATE
- Different purposes: AnalyticsCoordinator is for external export, EvaluationCoordinator is for RL/optimization
- Both can coexist
- Integration: Use EvaluationCoordinator for internal, AnalyticsCoordinator for external

### 2. ConfigCoordinator

**Analysis:**
- ConfigCoordinator: Configuration loading from multiple sources (YAML, settings, env)
- No equivalent existing coordinator

**Recommendation:** INTEGRATE
- This fills a gap - no existing coordinator for this
- Add to orchestrator __init__
- Migrate config loading methods from orchestrator

### 3. PromptCoordinator

**Analysis:**
- PromptCoordinator: Prompt building from multiple contributors
- No equivalent existing coordinator

**Recommendation:** INTEGRATE
- This fills a gap - no existing coordinator for this
- Add to orchestrator __init__
- Migrate prompt building methods from orchestrator

### 4. ContextCoordinator

**Analysis:**
- ContextCoordinator: Context compaction with multiple strategies
- ContextCompactor exists but is different (lower-level compaction)

**Recommendation:** INTEGRATE (with care)
- ContextCoordinator is higher-level (orchestrates strategies)
- ContextCompactor is lower-level (actual compaction logic)
- Use ContextCoordinator to wrap ContextCompactor

### 5. Summary

| Coordinator | Action | Rationale |
|-------------|--------|-----------|
| AnalyticsCoordinator | ENHANCE | Add to orchestrator, keep separate from EvaluationCoordinator |
| ConfigCoordinator | INTEGRATE | Fills gap, no duplicate |
| PromptCoordinator | INTEGRATE | Fills gap, no duplicate |
| ContextCoordinator | INTEGRATE | Wraps ContextCompactor, no duplicate |
"""
    )

    print("\n## NEXT STEPS\n")
    print(
        """
1. AnalyticsCoordinator: Add to orchestrator for external analytics export
2. ConfigCoordinator: Integrate for configuration loading (Days 8-12)
3. PromptCoordinator: Integrate for prompt building (Days 20-22)
4. ContextCoordinator: Integrate as wrapper for ContextCompactor (Days 13-19)

All avoid duplication - they fill gaps or orchestrate existing components.
"""
    )


if __name__ == "__main__":
    main()
