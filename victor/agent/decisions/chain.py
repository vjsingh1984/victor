# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Decision chain resolver — unified fallback chain for all decision points.

All decision sites in the agentic loop should use `get_decision_chain()`
to determine whether to use heuristic-first or LLM-first, ensuring
consistent behavior across stage detection, tool selection, task
completion, etc.

Usage:
    from victor.agent.decisions.chain import get_decision_chain, should_use_llm

    # Get the ordered chain for a decision type
    chain = get_decision_chain("stage_detection")  # ["heuristic", "llm"]

    # Quick check: should LLM be used for this decision type?
    if should_use_llm("tool_selection"):
        tools = edge_model_filter(tools)
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# Cache to avoid repeated settings loads
_chain_cache: dict = {}


def get_decision_chain(decision_type: str) -> List[str]:
    """Get the fallback chain for a decision type.

    Reads from settings.pipeline.decision_chain (per-type override)
    or settings.pipeline.decision_chain_default (global default).

    Args:
        decision_type: DecisionType enum value (e.g., "stage_detection",
                       "tool_selection", "task_completion")

    Returns:
        Ordered list of strategies: ["heuristic", "llm"] or ["llm", "heuristic"]
    """
    if decision_type in _chain_cache:
        return _chain_cache[decision_type]

    try:
        from victor.config.settings import load_settings

        settings = load_settings()
        pipeline = getattr(settings, "pipeline", None)
        if pipeline:
            # Check per-type override first
            chain_overrides = getattr(pipeline, "decision_chain", {})
            if decision_type in chain_overrides:
                chain = chain_overrides[decision_type]
                _chain_cache[decision_type] = chain
                return chain

            # Fall back to global default
            chain = getattr(
                pipeline, "decision_chain_default", ["heuristic", "llm"]
            )
            _chain_cache[decision_type] = chain
            return chain
    except Exception:
        pass

    default = ["heuristic", "llm"]
    _chain_cache[decision_type] = default
    return default


def should_use_llm(decision_type: str) -> bool:
    """Check if LLM is in the fallback chain for this decision type.

    Returns True if "llm" appears anywhere in the chain.
    Use this for simple gate checks (e.g., should we call edge model at all?).
    """
    return "llm" in get_decision_chain(decision_type)


def is_llm_primary(decision_type: str) -> bool:
    """Check if LLM is the PRIMARY strategy (first in chain).

    Returns True only when "llm" is the first strategy.
    Use this when you need to know if LLM should be tried BEFORE heuristics.
    """
    chain = get_decision_chain(decision_type)
    return len(chain) > 0 and chain[0] == "llm"


def invalidate_chain_cache() -> None:
    """Clear the chain cache (for testing or settings reload)."""
    _chain_cache.clear()


def log_decision(
    decision_type: str,
    context: dict,
    result: str,
    source: str,
    confidence: float = 0.0,
) -> None:
    """Append decision I/O to JSONL file for fine-tuning data collection.

    Logs every decision (heuristic or LLM) with input context and output,
    enabling future model fine-tuning on real decision patterns.

    Path: ~/.victor/logs/decisions.jsonl
    """
    import json
    from datetime import datetime
    from pathlib import Path

    try:
        log_dir = Path.home() / ".victor" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "decisions.jsonl"

        entry = {
            "ts": datetime.now().isoformat(),
            "type": decision_type,
            "input": context,
            "output": result,
            "source": source,
            "confidence": confidence,
        }

        with open(log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass  # Never fail on logging
