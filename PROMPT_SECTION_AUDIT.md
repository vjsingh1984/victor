# Prompt Section Audit

**Generated:** 2026-04-30
**Purpose:** Audit all prompt sections across legacy and framework builders, compare with evolvable sections, identify gaps.

---

## 1. Legacy Agent Prompt Builder (`victor/agent/prompt_builder.py`)

### Defined Sections (constants):

| Section | Evolvable? | Description |
|---------|-----------|-------------|
| `GROUNDING_RULES` | ✅ Yes | Basic safety grounding |
| `GROUNDING_RULES_EXTENDED` | — | Extended grounding (variant) |
| `PARALLEL_READ_GUIDANCE` | ❌ No | Parallel file reading guidance |
| `CONCISE_MODE_GUIDANCE` | ❌ No | Concise output mode |
| `COMPLETION_GUIDANCE` | ✅ Yes | Task completion behavior |
| `LARGE_FILE_PAGINATION_GUIDANCE` | ❌ No | Handling large files |
| `ASI_TOOL_EFFECTIVENESS_GUIDANCE` | ✅ Yes | Tool usage patterns |

**Total:** 7 sections, 3 evolvable

---

## 2. Framework Prompt Builder (`victor/framework/prompt_builder.py`)

### Registered PromptBlocks:

| Block Name | Kind | Evolvable? | Priority |
|------------|------|------------|----------|
| `task_hint` | task_hint | ❌ No | Dynamic |
| `task_execution_guidance` | task_execution_guidance | ❌ No | PRIORITY_TASK_HINTS |
| `tool_hints` | tool_hints | ❌ No | PRIORITY_TOOL_HINTS |
| `analysis_efficiency` | analysis_efficiency | ❌ No | PRIORITY_TOOL_HINTS |
| `safety_rules` | safety_rules | ❌ No | PRIORITY_GROUNDING |
| `context` | context | ❌ No | PRIORITY_CONTEXT |
| `grounding` | grounding | ✅ Yes (maps to GROUNDING_RULES) | PRIORITY_GROUNDING |

**Total:** 7 blocks, 1 evolvable (grounding → GROUNDING_RULES)

---

## 3. Evolvable Sections (`victor/framework/rl/learners/prompt_optimizer.py`)

### EVOLVABLE_SECTIONS List:

| Section | Defined In | Found In Builder? |
|---------|------------|-------------------|
| `ASI_TOOL_EFFECTIVENESS_GUIDANCE` | Legacy builder | ✅ Yes (as `tool_hints`) |
| `GROUNDING_RULES` | Legacy builder | ✅ Yes (as `grounding`) |
| `COMPLETION_GUIDANCE` | Legacy builder | ❌ **NOT in framework builder** |
| `FEW_SHOT_EXAMPLES` | Dynamic (RL-injected) | ❌ **NOT in framework builder** |
| `INIT_SYNTHESIS_RULES` | Init synthesizer | ❌ **NOT in framework builder** |

**Total:** 5 evolvable sections, 2 registered in framework builder

---

## 4. Missing Sections in Framework Builder

### Critical Gaps (should be added):

1. **`COMPLETION_GUIDANCE`** (P0 - High Impact)
   - Already evolvable in RL system
   - Missing from framework builder's PromptBlocks
   - **Impact:** Agent completion behavior not optimized in framework paths
   - **Action Required:** Add PromptBlock for completion guidance

2. **`FEW_SHOT_EXAMPLES`** (P0 - High Impact)
   - Already evolvable in RL system with MIPROv2 strategy
   - Missing from framework builder
   - **Impact:** No few-shot learning in framework-based execution
   - **Action Required:** Add PromptBlock that gets populated by RL system

3. **`INIT_SYNTHESIS_RULES`** (P1 - Medium Impact)
   - Evolvable, used in init synthesizer
   - Missing from framework builder
   - **Impact:** Init prompt synthesis not optimized in framework paths
   - **Action Required:** Add PromptBlock for init synthesis

### Nice-to-Have (consider adding):

4. **`PARALLEL_READ_GUIDANCE`** (P2)
   - Not currently evolvable
   - Could improve performance of parallel file operations
   - **Action:** Consider adding to EVOLVABLE_SECTIONS

5. **`CONCISE_MODE_GUIDANCE`** (P2)
   - Not currently evolvable
   - PrefPoStrategy targets this
   - **Action:** Already tracked in some strategies, could add to EVOLVABLE_SECTIONS

---

## 5. Section Name Mapping Issues

### Inconsistent Naming:

| Legacy Name | Framework Name | Evolvable Name | Status |
|-------------|----------------|----------------|--------|
| `ASI_TOOL_EFFECTIVENESS_GUIDANCE` | `tool_hints` | `ASI_TOOL_EFFECTIVENESS_GUIDANCE` | ⚠️ Mismatch |
| `GROUNDING_RULES` | `grounding` | `GROUNDING_RULES` | ⚠️ Mismatch |
| `COMPLETION_GUIDANCE` | *(missing)* | `COMPLETION_GUIDANCE` | ❌ Missing |
| `task_execution_guidance` | `task_execution_guidance` | *(not evolvable)* | ✅ Consistent |

**Issue:** Framework builder uses lowercase `kind` values (e.g., `tool_hints`, `grounding`) while evolvable sections use uppercase constants (e.g., `ASI_TOOL_EFFECTIVENESS_GUIDANCE`, `GROUNDING_RULES`).

**Impact:** The RL system evolves sections by name, but the framework builder may not recognize them if names don't match.

---

## 6. Recommendations

### Immediate Actions (P0):

1. **Add `COMPLETION_GUIDANCE` PromptBlock to framework builder**
   - Copy from `victor/agent/prompt_builder.py:96`
   - Register with `name="completion_guidance"`
   - Map to evolvable `COMPLETION_GUIDANCE` section

2. **Add `FEW_SHOT_EXAMPLES` PromptBlock to framework builder**
   - Create empty template that gets populated by RL system
   - Register with `name="few_shot_examples"`
   - Load from `optimization_injector._sample_evolved_payload()`

3. **Add `INIT_SYNTHESIS_RULES` PromptBlock to framework builder**
   - Extract from `victor/framework/init_synthesizer.py`
   - Register with `name="init_synthesis_rules"`

### Medium Priority (P1):

4. **Align section naming convention**
   - Decide: lowercase `kind` vs uppercase `SECTION_NAME`
   - Update EVOLVABLE_SECTIONS or PromptBlock names to match
   - Document mapping in one place

5. **Add `analysis_efficiency` to EVOLVABLE_SECTIONS?**
   - **NO** — this is static instructional guidance
   - Should NOT evolve; tool usage patterns are stable
   - Current implementation is correct

### Lower Priority (P2):

6. **Consider adding `PARALLEL_READ_GUIDANCE` to EVOLVABLE_SECTIONS**
   - Could optimize parallel file operations
   - Requires performance benchmarking first

7. **Consider adding `CONCISE_MODE_GUIDANCE` to EVOLVABLE_SECTIONS**
   - Already tracked by PrefPoStrategy
   - May need to add to EVOLVABLE_SECTIONS explicitly

---

## 7. Shared Helper Design for Legacy + StateGraph Paths

### Current State:

| Component | Legacy Path | StateGraph Path | Shared? |
|-----------|-------------|-----------------|---------|
| `analysis_checkpoint` tool | ✅ Auto-discovered | ✅ Auto-discovered | ✅ Yes |
| `WritePathPolicy` | ✅ Via IsolationMapper | ❓ Unclear | ⚠️ Partial |
| System prompt blocks | Legacy builder | Framework builder | ❌ No |

### Gaps Identified:

1. **WritePathPolicy activation**
   - Legacy: Activated in `IsolationMapper.from_constraints()` (workflow tasks)
   - StateGraph: Agent nodes use `SubAgent.spawn()` → may not call `IsolationMapper`
   - **Risk:** Policy not active in StateGraph agent nodes

2. **System prompt construction**
   - Legacy: Uses `victor/agent/prompt_builder.py` + `content_registry.py`
   - Framework: Uses `victor/framework/prompt_builder.py`
   - **Risk:** Inconsistent prompts, missing sections in one path

### Proposed Shared Helper Design:

```python
# victor/framework/shared/prompt_orchestrator.py

class PromptOrchestrator:
    """Unified prompt construction for both legacy and framework paths.
    
    Consolidates:
    - Legacy builder (victor/agent/prompt_builder.py)
    - Framework builder (victor/framework/prompt_builder.py)
    - RL evolved sections (ASI_TOOL_EFFECTIVENESS_GUIDANCE, etc.)
    - Analysis efficiency guidance (new)
    """

    def build_full_prompt(
        self,
        settings: Settings,
        provider: str,
        vertical: Optional[str] = None,
        mode: Optional[str] = None,
        include_analysis_efficiency: bool = True,
    ) -> str:
        """Build complete system prompt from all sources.
        
        Delegates to:
        - FrameworkPromptBuilder for structure
        - OptimizationInjector for RL-evolved sections
        - Vertical-specific contributors
        """
        ...

# victor/framework/shared/constraint_orchestrator.py

class ConstraintOrchestrator:
    """Unified constraint activation for both legacy and framework paths.
    
    Ensures WritePathPolicy is active in:
    - Legacy workflow execution (IsolationMapper)
    - StateGraph agent nodes (SubAgent.spawn)
    - Direct agent execution
    """

    @staticmethod
    def activate_constraints(
        constraints: TaskConstraints,
        context: ExecutionContext,
    ) -> None:
        """Activate write policy and other constraints.
        
        Called by:
        - IsolationMapper.from_constraints() (legacy)
        - AgentNodeExecutor.execute() (StateGraph)
        - SubAgent.execute() (sub-agents)
        """
        if constraints.write_path_policy:
            set_active_write_policy(constraints.write_path_policy)
        elif hasattr(constraints, 'write_allowed'):
            # Legacy translation
            ...
```

---

## 8. Implementation Plan

### Phase 1: Add Missing PromptBlocks (1-2 hours)
- [ ] Add `COMPLETION_GUIDANCE` to framework builder
- [ ] Add `FEW_SHOT_EXAMPLES` to framework builder
- [ ] Add `INIT_SYNTHESIS_RULES` to framework builder

### Phase 2: Unify Prompt Construction (2-3 hours)
- [ ] Create `PromptOrchestrator` shared helper
- [ ] Migrate legacy builder to use orchestrator
- [ ] Update framework builder to use orchestrator
- [ ] Test both paths produce identical prompts

### Phase 3: Ensure Constraint Activation (1-2 hours)
- [ ] Create `ConstraintOrchestrator` shared helper
- [ ] Integrate into `AgentNodeExecutor.execute()`
- [ ] Integrate into `SubAgent.__init__()`
- [ ] Verify write policy active in all paths

### Phase 4: Naming Convention Alignment (1 hour)
- [ ] Decide on naming convention (lowercase vs uppercase)
- [ ] Update PromptBlock names to match evolvable sections
- [ ] Update EVOLVABLE_SECTIONS if needed
- [ ] Document mapping in single source of truth

**Total Estimate:** 5-8 hours

---

## 9. Verification Checklist

After implementation:

- [ ] All 5 EVOLVABLE_SECTIONS registered in framework builder
- [ ] Section names consistent across all builders
- [ ] `analysis_checkpoint` works in both legacy and StateGraph paths
- [ ] `WritePathPolicy` active in both legacy and StateGraph paths
- [ ] RL system can evolve all sections in both paths
- [ ] Prompts identical for same config across both paths
- [ ] No duplicate prompt construction logic
