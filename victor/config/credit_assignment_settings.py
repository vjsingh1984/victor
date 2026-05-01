"""Credit assignment configuration for automatic tool-level credit tracking."""

from __future__ import annotations

from pydantic import BaseModel


class CreditAssignmentSettings(BaseModel):
    """Settings for runtime credit assignment (arXiv:2604.09459).

    When enabled, the credit tracking service automatically records
    tool execution results and assigns credit at turn boundaries.
    """

    # Master switch — opt-in (default off)
    enabled: bool = False

    # Default methodology for automatic credit assignment
    # Options: gae, shapley, monte_carlo, td, hindsight, n_step
    default_methodology: str = "gae"

    # Automatically assign credit at the end of each turn
    auto_assign_at_turn_boundary: bool = True

    # Emit credit signals to ObservabilityBus (topic: credit.*)
    emit_observability_events: bool = True

    # Persist credit data to SQLite for historical analysis
    persist_to_db: bool = False

    # GAE parameters
    gamma: float = 0.99
    lambda_gae: float = 0.95

    # Shapley sampling count (higher = more accurate but slower)
    shapley_sampling_count: int = 10

    # Enrich GEPA execution traces with credit signals
    enrich_gepa_traces: bool = True
