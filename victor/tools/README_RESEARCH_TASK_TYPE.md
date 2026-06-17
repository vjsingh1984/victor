# Research Task Type - Usage Guide

**Date**: 2026-04-20
**Status**: Production Ready

---

## Overview

The RESEARCH task type has been enhanced to support comprehensive multi-phase research tasks with improved resource allocation, progress tracking, and content preservation.

---

## Configuration

### Resource Allocation
```python
"research": {
    "tool_budget": 45,        # Allow 45 tool calls (was 20)
    "max_iterations": 25,     # Allow 25 iterations (was 10)
    "priority_tools": ["web_search", "web_fetch", "read", "grep"],
}
```

### Four-Phase Workflow

Research tasks automatically follow a 4-phase workflow:

1. **Discover** (INITIAL, PLANNING stages)
   - Use `web_search` to find relevant papers
   - Identify key search terms

2. **Search** (READING stage)
   - Use `web_fetch` and `read` to gather full papers
   - Extract key information

3. **Analyze** (ANALYSIS stage)
   - Use `read` and `grep` to analyze findings
   - Synthesize patterns and insights

4. **Synthesize** (EXECUTION, VERIFICATION stages)
   - Use `write` to create summary report
   - Verify findings

---

## Progress Tracking

### Automatic Progress Updates

Research tasks report progress every 10 iterations:

```
📊 RESEARCH PROGRESS: 25% | Phase: search | Papers found: 12 | Iteration: 10/25
```

**Progress Components**:
- **Percentage**: 0-100% based on 4 equal phases (25% each)
- **Phase**: Current research phase (discover/search/analyze/synthesize)
- **Papers Found**: Unique arXiv IDs detected in conversation
- **Iteration**: Current/maximum iterations

---

## Content Preservation

### Tool Output Pruning
Research tasks use lenient pruning rules:
- **max_lines**: 500 (vs. 100 default)
- **strip_comments**: False (preserve all context)
- **preserve_paper_ids**: True (don't truncate arXiv IDs)
- **preserve_search_results**: True
- **preserve_urls**: True

### Conversation Pruning
Research tasks preserve extensive context:
- **min_messages**: 100 (vs. 6 default)
- **target_messages**: 1000 (vs. 250 default)
- **tool_result_weight**: 3.0 (high priority on tool results)
- **recent_weight**: 1.5 (lower priority on recency)
- **Special handling**: Paper IDs and search results never pruned

---

## Shell Command Caching

Research task commands automatically cached:
- **arxiv**: All arxiv CLI commands
- **web_search**: Web search results
- **gh**: GitHub CLI (22 readonly subcommands)
- **az**: Azure CLI (9 readonly subcommands)
- **kubectl**: Kubernetes CLI (13 readonly subcommands)

**Cache TTL**:
- arxiv/web_search: 10 minutes
- gh/az/kubectl: 5-10 minutes

---

## Usage Example

```bash
victor chat "Research the latest arXiv papers on agent-side LLM optimization from 2025-2026 and summarize the top 5 findings with paper IDs"
```

**Expected Execution**:
1. Task runs for up to 25 iterations
2. Progress updates at iterations 10, 20
3. Full content preserved (500+ lines per file)
4. 1000+ messages preserved in conversation
5. Commands cached (faster subsequent runs)
6. Clear phase transitions logged

---

## API Usage

### Programmatic Usage

```python
from victor import Agent

agent = await Agent.create()

result = await agent.chat(
    "Research arXiv papers on tool output pruning",
    context={
        "task_type": "research",  # Explicit research task
        "tool_budget": 45,         # Use enhanced budget
        "max_iterations": 25,      # Use extended iterations
    }
)

print(result.response)
```

### With Streaming

```python
async for chunk in agent.stream(
    "Research agent optimization techniques",
    context={"task_type": "research"}
):
    if "progress" in chunk:
        print(f"Progress: {chunk['progress']}%")
```

---

## Monitoring

### Key Metrics

Track these metrics to assess research task performance:

1. **Completion Rate**: % of research tasks that complete successfully
2. **Average Iterations**: Mean iterations per research task (target: <25)
3. **Paper Count**: Mean unique papers found per task
4. **Cache Hit Rate**: % of readonly commands served from cache (target: >80%)
5. **Content Preservation**: % of tool output preserved (target: 100% for research)

### Logging

Research tasks log:
- Phase transitions: `RESEARCH PHASE TRANSITION: search → READING`
- Progress updates: `📊 RESEARCH PROGRESS: 25%`
- Stage enforcement: `discover → INITIAL`

---

## Troubleshooting

### Task Not Completing

**Symptom**: Task hits iteration limit without completing

**Solutions**:
1. Increase `max_iterations` (current: 25)
2. Narrow research scope (fewer papers to analyze)
3. Check for tool errors in logs

### Content Truncated

**Symptom**: Paper IDs or search results cut off

**Solutions**:
1. Verify task_type is "research" (not "general")
2. Check output_pruner.py for "research" rule
3. Verify `preserve_paper_ids: True`

### No Progress Updates

**Symptom**: No progress logged during task

**Solutions**:
1. Verify task_type is "research"
2. Check iteration count (updates every 10)
3. Check log level (INFO required)

---

## Best Practices

1. **Specify task_type explicitly**: Use `context={"task_type": "research"}`
2. **Narrow scope**: Focus on specific topics (e.g., "tool output pruning" not "LLM optimization")
3. **Use web_search wisely**: Start with broad searches, then narrow down
4. **Monitor progress**: Watch for progress updates every 10 iterations
5. **Check cache**: Second run should be faster (cached commands)

---

## Comparison: Before vs After

| Metric | Before | After |
|--------|--------|-------|
| Tool Budget | 20 | 45 (+125%) |
| Max Iterations | 10 | 25 (+150%) |
| Content Preserved | 100 lines | 500+ lines (+400%) |
| Messages Preserved | 250 | 1000+ (+300%) |
| Progress Updates | None | Every 10 iterations |
| Cached Commands | Limited | arxiv, web_search, gh, az, kubectl |
| Phases | Implicit | Explicit 4-phase workflow |

---

**Status**: ✅ Production Ready

All enhancements tested and verified. Research tasks are now fully supported with comprehensive resource allocation, progress tracking, and content preservation.
