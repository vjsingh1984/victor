# Deprecation Inventory (2026-03-03)

Owner: Verticals Lead (program owner), with role-level ownership per entry.

## Inventory

| Deprecated API/Symbol | Source Location | Replacement | Owner Role | Target Removal Version | Target Removal Date |
|---|---|---|---|---|---|
| `UnifiedWorkflowCompilerAdapter` legacy API surface | `victor/workflows/adapter.py` | Protocol-based compiler API (`victor.workflows.compiler`) | Architecture Lead | `v0.7.0` | `2026-06-30` |
| `UnifiedWorkflowCompiler` | `victor/workflows/unified_compiler.py` | `create_compiler("yaml://", ...)` plugin architecture | Architecture Lead | `v0.7.0` | `2026-06-30` |
| `WorkflowGraph` alias from `victor.workflows.graph` | `victor/workflows/graph.py` | `BasicWorkflowGraph` or `victor.workflows.graph_dsl.WorkflowGraph` | Architecture Lead | `v0.8.0` | `2026-12-31` |
| Sync `AgentOrchestrator.switch_provider(...)` | `victor/agent/orchestrator.py` | Async `await orchestrator.switch_provider(...)` | Architecture Lead | `v0.8.0` | `2026-12-31` |
| `get_tiered_tools()` extension hook | `victor/core/verticals/extension_loader.py` | `get_tiered_tool_config()` | Verticals Lead | `v0.7.0` | `2026-06-30` |
| `TASK_TYPE_HINTS` and deprecated task-hint fallback path | `victor/agent/prompt_builder.py` | Vertical prompt contributors + `get_task_type_hint(..., prompt_contributors=[...])` | Architecture Lead | `v0.8.0` | `2026-12-31` |
| `CodingToolDependencyProvider` + deprecated module constants | `victor/verticals/contrib/coding/tool_dependencies.py` | `create_vertical_tool_dependency_provider("coding")` | Verticals Lead | `v0.8.0` | `2026-12-31` |
| `DataAnalysisToolDependencyProvider` + deprecated module constants | `victor/verticals/contrib/dataanalysis/tool_dependencies.py` | `create_vertical_tool_dependency_provider("dataanalysis")` | Verticals Lead | `v0.8.0` | `2026-12-31` |
| `ResearchToolDependencyProvider` + legacy constants | `victor/verticals/contrib/research/tool_dependencies.py` | `create_vertical_tool_dependency_provider("research")` | Verticals Lead | `v0.8.0` | `2026-12-31` |
| Deprecated DevOps dependency constants | `victor/verticals/contrib/devops/tool_dependencies.py` | `create_vertical_tool_dependency_provider("devops")` | Verticals Lead | `v0.8.0` | `2026-12-31` |
| `CodingTeamSpec` | `victor/verticals/contrib/coding/teams/specs.py` | `TeamSpec` from `victor.framework.team_schema` | Verticals Lead | `v0.8.0` | `2026-12-31` |
| `ResearchTeamSpec` | `victor/verticals/contrib/research/teams/__init__.py` | `TeamSpec` from `victor.framework.team_schema` | Verticals Lead | `v0.8.0` | `2026-12-31` |
| Fragmented/legacy event type names (module-level compatibility) | `victor/core/events/taxonomy.py` | `UnifiedEventType` values from `victor.core.events.taxonomy` | Observability Lead | `v0.8.0` | `2026-12-31` |

## Notes

- Entries with explicit source-level targets were kept as-is (`v0.7.0` / `2026-06-30`).
- Entries without explicit source targets are assigned provisional `v0.8.0` / `2026-12-31` per policy and should be re-validated at each milestone cut.
