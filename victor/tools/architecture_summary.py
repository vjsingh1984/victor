from typing import Any, Dict, List, Optional
from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool
from victor.tools.graph_tool import graph


@tool(
    name="arch_summary",  # short canonical name; legacy alias via TOOL_ALIASES maps architecture_summary -> arch_summary
    category="code_intelligence",
    priority=Priority.CRITICAL,
    mandatory_keywords=[
        "architecture",
        "overview",
        "hotspot",
        "coupling",
        "executive summary",
        "impact",
        "module hubs",
    ],
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.READ_ONLY,
    stages=["initial", "planning", "analysis"],
    task_types=["design", "analysis"],
    keywords=[
        "architecture",
        "summary",
        "hotspots",
        "modules",
        "coupling",
        "pagerank",
        "centrality",
        "edge counts",
        "callsites",
    ],
    use_cases=[
        "executive architecture summary",
        "identify module hubs and coupling hotspots",
        "report edge counts and callsites for key modules",
    ],
    examples=[
        "Summarize the architecture with top modules and coupling hotspots; include edge counts and callsites.",
        "Give an executive architecture overview citing module pagerank/centrality with callsites.",
    ],
)
async def architecture_summary(
    top_k: int = 5,
    include_symbols: bool = True,
) -> Dict[str, Any]:
    """Return a structured architecture snapshot (modules + symbol hotspots).

    Uses module-level PageRank and centrality plus symbol-level PageRank to surface
    high-impact areas. Defaults to runtime-only results with CALLS-weighted edges,
    edge counts, and sample callsites included.
    """

    # Module PageRank (architectural importance)
    mod_pr = await graph(
        mode="module_pagerank",
        structured=True,
        include_modules=True,
        include_symbols=False,
        include_callsites_modules=True,
        include_calls=True,
        include_refs=False,
        only_runtime=True,
        top_k=top_k,
    )

    # Module centrality (coupling hotspots)
    mod_central = await graph(
        mode="module_centrality",
        structured=True,
        include_modules=True,
        include_symbols=False,
        include_callsites_modules=True,
        include_calls=True,
        include_refs=False,
        only_runtime=True,
        top_k=top_k,
    )

    # Symbol hotspots (PageRank) if requested
    symbol_pr: Optional[Dict[str, Any]] = None
    if include_symbols:
        symbol_pr = await graph(
            mode="pagerank",
            structured=True,
            include_modules=False,
            include_symbols=True,
            include_callsites=True,
            include_calls=True,
            include_refs=False,
            only_runtime=True,
            top_k=top_k,
        )

    return {
        "modules": {
            "pagerank": mod_pr.get("modules", []) if isinstance(mod_pr, dict) else [],
            "centrality": mod_central.get("modules", []) if isinstance(mod_central, dict) else [],
        },
        "symbols": (
            symbol_pr.get("symbols", []) if include_symbols and isinstance(symbol_pr, dict) else []
        ),
    }
