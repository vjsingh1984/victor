# Workflows API Reference - Part 4

**Part 4 of 4:** Configuration and Complete Example

---

## Navigation

- [Part 1: StateGraph, Compiler, Provider](part-1-stategraph-compiler-provider.md)
- [Part 2: Node Types](part-2-node-types.md)
- [Part 3: State & Results](part-3-state-results.md)
- **[Part 4: Configuration & Examples](#)** (Current)
- [**Complete Reference**](../workflows-api.md)

---

```python
from typing import TypedDict, Optional, List
from victor.framework.graph import StateGraph, END
from victor.framework.config import GraphConfig, ExecutionConfig

# Define typed state
class ResearchState(TypedDict):
    query: str
    sources: List[str]
    findings: Optional[str]
    report: Optional[str]

# Define node functions
async def search(state: ResearchState) -> ResearchState:
    state["sources"] = [f"Source for: {state['query']}"]
    return state

async def analyze(state: ResearchState) -> ResearchState:
    state["findings"] = f"Analyzed: {state['sources']}"
    return state

def should_continue(state: ResearchState) -> str:
    if len(state.get("sources", [])) > 0:
        return "continue"
    return "retry"

async def report(state: ResearchState) -> ResearchState:
    state["report"] = f"# Report\n\n{state['findings']}"
    return state

# Build graph
graph = StateGraph(ResearchState)
graph.add_node("search", search)
graph.add_node("analyze", analyze)
graph.add_node("report", report)

graph.add_edge("search", "analyze")
graph.add_conditional_edge(
    "analyze",
    should_continue,
    {"continue": "report", "retry": "search"}
)
graph.set_entry_point("search")
graph.set_finish_point("report")

# Compile with configuration
config = GraphConfig(
    execution=ExecutionConfig(max_iterations=10, timeout=60.0)
)
app = graph.compile()

# Execute
async def main():
    result = await app.invoke(
        {"query": "AI trends 2024", "sources": [], "findings": None, "report": None},
        config=config
    )

    if result.success:
        print(f"Report:\n{result.state['report']}")
    else:
        print(f"Error: {result.error}")

import asyncio
asyncio.run(main())
```

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 5 minutes
