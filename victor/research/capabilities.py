# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dynamic capability definitions for the Research vertical.

This module provides capability declarations that can be loaded
dynamically by the CapabilityLoader, enabling runtime extension
of the Research vertical with custom functionality.

Refactored to use BaseVerticalCapabilityProvider, reducing from
810 lines to ~250 lines by eliminating duplicated patterns.

Example:
    # Use provider
    from victor.research.capabilities import ResearchCapabilityProvider

    provider = ResearchCapabilityProvider()

    # Apply capabilities
    provider.apply_source_verification(orchestrator, min_credibility=0.8)
    provider.apply_citation_management(orchestrator, style="chicago")

    # Get configurations
    config = provider.get_capability_config(orchestrator, "citation_management")
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING, cast

from victor.framework.capabilities.base_vertical_capability_provider import (
    BaseVerticalCapabilityProvider,
    CapabilityDefinition,
)
from victor.framework.protocols import CapabilityType, OrchestratorCapability
from victor.framework.capability_loader import CapabilityEntry, capability

if TYPE_CHECKING:
    from victor.core.protocols import OrchestratorProtocol as AgentOrchestrator

logger = logging.getLogger(__name__)


# =============================================================================
# Capability Handlers (configure_*, get_* functions)
# =============================================================================


def configure_source_verification(
    orchestrator: Any,
    *,
    min_credibility: float = 0.7,
    min_source_count: int = 3,
    require_diverse_sources: bool = True,
    validate_urls: bool = True,
) -> None:
    """Configure source verification rules for the orchestrator.

    This capability configures how the research assistant validates
    and assesses the credibility of sources.

    Args:
        orchestrator: Target orchestrator
        min_credibility: Minimum average credibility score (0-1)
        min_source_count: Minimum number of sources required
        require_diverse_sources: Require source diversity (domains, types)
        validate_urls: Validate URL accessibility and format
    """
    # SOLID DIP: Store config in VerticalContext instead of direct attribute write
    context = orchestrator.vertical_context
    source_verification_config = {
        "min_credibility": min_credibility,
        "min_source_count": min_source_count,
        "require_diverse_sources": require_diverse_sources,
        "validate_urls": validate_urls,
    }
    context.set_capability_config("source_verification", source_verification_config)

    logger.info(
        f"Configured source verification: credibility>={min_credibility:.0%}, "
        f"min_sources={min_source_count}"
    )


def get_source_verification(orchestrator: Any) -> Dict[str, Any]:
    """Get current source verification configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Source verification configuration dict
    """
    # SOLID DIP: Read from VerticalContext instead of direct attribute access
    context = orchestrator.vertical_context
    return cast(
        Dict[str, Any],
        context.get_capability_config(
            "source_verification",
            {
                "min_credibility": 0.7,
                "min_source_count": 3,
                "require_diverse_sources": True,
                "validate_urls": True,
            },
        ),
    )


def configure_citation_management(
    orchestrator: Any,
    *,
    default_style: str = "apa",
    require_urls: bool = True,
    include_authors: bool = True,
    include_dates: bool = True,
) -> None:
    """Configure citation and bibliography management.

    Args:
        orchestrator: Target orchestrator
        default_style: Citation style (apa, mla, chicago, harvard)
        require_urls: Require URLs for web sources
        include_authors: Include author names in citations
        include_dates: Include publication dates
    """
    # SOLID DIP: Store config in VerticalContext instead of direct attribute write
    context = orchestrator.vertical_context
    citation_config = {
        "default_style": default_style,
        "require_urls": require_urls,
        "include_authors": include_authors,
        "include_dates": include_dates,
    }
    context.set_capability_config("citation_management", citation_config)

    logger.info(f"Configured citation management: style={default_style}")


def get_citation_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current citation configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Citation configuration dict
    """
    # SOLID DIP: Read from VerticalContext instead of direct attribute access
    context = orchestrator.vertical_context
    return cast(
        Dict[str, Any],
        context.get_capability_config(
            "citation_management",
            {
                "default_style": "apa",
                "require_urls": True,
                "include_authors": True,
                "include_dates": True,
            },
        ),
    )


def configure_research_quality(
    orchestrator: Any,
    *,
    min_coverage_score: float = 0.75,
    min_source_diversity: int = 2,
    check_recency: bool = True,
    max_source_age_days: Optional[int] = 365,
) -> None:
    """Configure research quality standards.

    Args:
        orchestrator: Target orchestrator
        min_coverage_score: Minimum coverage score (0-1)
        min_source_diversity: Minimum number of source types (web, academic, code)
        check_recency: Check source recency for time-sensitive topics
        max_source_age_days: Maximum acceptable source age in days
    """
    # SOLID DIP: Store config in VerticalContext instead of direct attribute write
    context = orchestrator.vertical_context
    research_quality_config = {
        "min_coverage_score": min_coverage_score,
        "min_source_diversity": min_source_diversity,
        "check_recency": check_recency,
        "max_source_age_days": max_source_age_days,
    }
    context.set_capability_config("research_quality", research_quality_config)

    logger.info(
        f"Configured research quality: coverage>={min_coverage_score:.0%}, "
        f"diversity>={min_source_diversity}"
    )


def get_research_quality(orchestrator: Any) -> Dict[str, Any]:
    """Get current research quality configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Research quality configuration dict
    """
    # SOLID DIP: Read from VerticalContext instead of direct attribute access
    context = orchestrator.vertical_context
    return cast(
        Dict[str, Any],
        context.get_capability_config(
            "research_quality",
            {
                "min_coverage_score": 0.75,
                "min_source_diversity": 2,
                "check_recency": True,
                "max_source_age_days": 365,
            },
        ),
    )


def configure_literature_analysis(
    orchestrator: Any,
    *,
    min_relevance_score: float = 0.6,
    weight_citation_count: bool = True,
    prefer_recent_papers: bool = True,
    recent_paper_years: int = 5,
) -> None:
    """Configure literature and academic paper analysis.

    Args:
        orchestrator: Target orchestrator
        min_relevance_score: Minimum paper relevance score (0-1)
        weight_citation_count: Consider citation count in relevance
        prefer_recent_papers: Prioritize recent publications
        recent_paper_years: Years considered "recent"
    """
    # SOLID DIP: Store config in VerticalContext instead of direct attribute write
    context = orchestrator.vertical_context
    literature_config = {
        "min_relevance_score": min_relevance_score,
        "weight_citation_count": weight_citation_count,
        "prefer_recent_papers": prefer_recent_papers,
        "recent_paper_years": recent_paper_years,
    }
    context.set_capability_config("literature_analysis", literature_config)

    logger.info(
        f"Configured literature analysis: relevance>={min_relevance_score:.0%}, "
        f"recent_years={recent_paper_years}"
    )


def get_literature_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current literature analysis configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Literature configuration dict
    """
    # SOLID DIP: Read from VerticalContext instead of direct attribute access
    context = orchestrator.vertical_context
    return cast(
        Dict[str, Any],
        context.get_capability_config(
            "literature_analysis",
            {
                "min_relevance_score": 0.6,
                "weight_citation_count": True,
                "prefer_recent_papers": True,
                "recent_paper_years": 5,
            },
        ),
    )


def configure_fact_checking(
    orchestrator: Any,
    *,
    min_confidence_threshold: float = 0.5,
    require_multiple_sources: bool = True,
    min_source_count_for_claim: int = 2,
    track_supporting_refuting: bool = True,
) -> None:
    """Configure fact-checking and claim verification.

    Args:
        orchestrator: Target orchestrator
        min_confidence_threshold: Minimum confidence for verdict (0-1)
        require_multiple_sources: Require multiple sources for claims
        min_source_count_for_claim: Minimum sources to verify a claim
        track_supporting_refuting: Track supporting vs refuting evidence
    """
    # SOLID DIP: Store config in VerticalContext instead of direct attribute write
    context = orchestrator.vertical_context
    fact_checking_config = {
        "min_confidence_threshold": min_confidence_threshold,
        "require_multiple_sources": require_multiple_sources,
        "min_source_count_for_claim": min_source_count_for_claim,
        "track_supporting_refuting": track_supporting_refuting,
    }
    context.set_capability_config("fact_checking", fact_checking_config)

    logger.info(
        f"Configured fact checking: confidence>={min_confidence_threshold:.0%}, "
        f"min_sources={min_source_count_for_claim}"
    )


def get_fact_checking_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current fact-checking configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Fact-checking configuration dict
    """
    # SOLID DIP: Read from VerticalContext instead of direct attribute access
    context = orchestrator.vertical_context
    return cast(
        Dict[str, Any],
        context.get_capability_config(
            "fact_checking",
            {
                "min_confidence_threshold": 0.5,
                "require_multiple_sources": True,
                "min_source_count_for_claim": 2,
                "track_supporting_refuting": True,
            },
        ),
    )


# =============================================================================
# Capability Provider Class (Refactored to use BaseVerticalCapabilityProvider)
# =============================================================================


class ResearchCapabilityProvider(BaseVerticalCapabilityProvider):
    """Provider for Research-specific capabilities.

    Refactored to inherit from BaseVerticalCapabilityProvider, eliminating
    ~560 lines of duplicated boilerplate code.

    Example:
        provider = ResearchCapabilityProvider()

        # List available capabilities
        print(provider.list_capabilities())

        # Apply specific capabilities
        provider.apply_source_verification(orchestrator, min_credibility=0.8)
        provider.apply_citation_management(orchestrator, style="chicago")

        # Get configurations
        config = provider.get_capability_config(orchestrator, "citation_management")
    """

    def __init__(self) -> None:
        """Initialize the research capability provider."""
        super().__init__("research")

    def _get_capability_definitions(self) -> Dict[str, CapabilityDefinition]:
        """Define research capability definitions.

        Returns:
            Dictionary of research capability definitions
        """
        return {
            "source_verification": CapabilityDefinition(
                name="source_verification",
                type=CapabilityType.SAFETY,
                description="Source credibility validation and verification settings",
                version="1.0",
                configure_fn="configure_source_verification",
                get_fn="get_source_verification",
                default_config={
                    "min_credibility": 0.7,
                    "min_source_count": 3,
                    "require_diverse_sources": True,
                    "validate_urls": True,
                },
                tags=["safety", "verification", "credibility", "sources"],
            ),
            "citation_management": CapabilityDefinition(
                name="citation_management",
                type=CapabilityType.TOOL,
                description="Citation management and bibliography formatting",
                version="1.0",
                configure_fn="configure_citation_management",
                get_fn="get_citation_config",
                default_config={
                    "default_style": "apa",
                    "require_urls": True,
                    "include_authors": True,
                    "include_dates": True,
                },
                tags=["citation", "bibliography", "formatting"],
            ),
            "research_quality": CapabilityDefinition(
                name="research_quality",
                type=CapabilityType.MODE,
                description="Research quality standards and coverage requirements",
                version="1.0",
                configure_fn="configure_research_quality",
                get_fn="get_research_quality",
                default_config={
                    "min_coverage_score": 0.75,
                    "min_source_diversity": 2,
                    "check_recency": True,
                    "max_source_age_days": 365,
                },
                dependencies=["source_verification"],
                tags=["quality", "coverage", "standards"],
            ),
            "literature_analysis": CapabilityDefinition(
                name="literature_analysis",
                type=CapabilityType.TOOL,
                description="Literature analysis and academic paper evaluation",
                version="1.0",
                configure_fn="configure_literature_analysis",
                get_fn="get_literature_config",
                default_config={
                    "min_relevance_score": 0.6,
                    "weight_citation_count": True,
                    "prefer_recent_papers": True,
                    "recent_paper_years": 5,
                },
                dependencies=["source_verification"],
                tags=["literature", "academic", "papers", "research"],
            ),
            "fact_checking": CapabilityDefinition(
                name="fact_checking",
                type=CapabilityType.SAFETY,
                description="Fact-checking and claim verification configuration",
                version="1.0",
                configure_fn="configure_fact_checking",
                get_fn="get_fact_checking_config",
                default_config={
                    "min_confidence_threshold": 0.5,
                    "require_multiple_sources": True,
                    "min_source_count_for_claim": 2,
                    "track_supporting_refuting": True,
                },
                dependencies=["source_verification"],
                tags=["fact-check", "verification", "evidence", "safety"],
            ),
        }

    # Delegate to handler functions (required by BaseVerticalCapabilityProvider)
    def configure_source_verification(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure source verification capability."""
        configure_source_verification(orchestrator, **kwargs)
        self._applied.add("source_verification")

    def configure_citation_management(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure citation management capability."""
        configure_citation_management(orchestrator, **kwargs)
        self._applied.add("citation_management")

    def configure_research_quality(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure research quality capability."""
        configure_research_quality(orchestrator, **kwargs)
        self._applied.add("research_quality")

    def configure_literature_analysis(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure literature analysis capability."""
        configure_literature_analysis(orchestrator, **kwargs)
        self._applied.add("literature_analysis")

    def configure_fact_checking(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure fact-checking capability."""
        configure_fact_checking(orchestrator, **kwargs)
        self._applied.add("fact_checking")

    def get_source_verification(self, orchestrator: Any) -> Dict[str, Any]:
        """Get source verification configuration."""
        return get_source_verification(orchestrator)

    def get_citation_config(self, orchestrator: Any) -> Dict[str, Any]:
        """Get citation configuration."""
        return get_citation_config(orchestrator)

    def get_research_quality(self, orchestrator: Any) -> Dict[str, Any]:
        """Get research quality configuration."""
        return get_research_quality(orchestrator)

    def get_literature_config(self, orchestrator: Any) -> Dict[str, Any]:
        """Get literature configuration."""
        return get_literature_config(orchestrator)

    def get_fact_checking_config(self, orchestrator: Any) -> Dict[str, Any]:
        """Get fact-checking configuration."""
        return get_fact_checking_config(orchestrator)


# =============================================================================
# CAPABILITIES List for CapabilityLoader Discovery
# =============================================================================


# Create singleton instance for generating CAPABILITIES list
_provider_instance: Optional[ResearchCapabilityProvider] = None


def _get_provider() -> ResearchCapabilityProvider:  # type: ignore
    """Get or create provider instance."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = ResearchCapabilityProvider()
    return _provider_instance


# Generate CAPABILITIES list from provider
CAPABILITIES: List[CapabilityEntry] = []


def _generate_capabilities_list() -> None:
    """Generate CAPABILITIES list from provider."""
    global CAPABILITIES
    if not CAPABILITIES:
        provider = _get_provider()
        CAPABILITIES.extend(provider.generate_capabilities_list())


_generate_capabilities_list()


# =============================================================================
# Convenience Functions
# =============================================================================


def get_research_capabilities() -> List[CapabilityEntry]:
    """Get all Research capability entries.

    Returns:
        List of capability entries for loader registration
    """
    return CAPABILITIES.copy()


def create_research_capability_loader() -> Any:
    """Create a CapabilityLoader pre-configured for Research vertical.

    Returns:
        CapabilityLoader with Research capabilities registered
    """
    from victor.framework.capability_loader import CapabilityLoader

    provider = _get_provider()
    return provider.create_capability_loader()


def get_capability_configs() -> Dict[str, Any]:
    """Get Research capability configurations for centralized storage.

    Returns default Research configuration for VerticalContext storage.
    This replaces direct orchestrator.source_verification_config assignment.

    Returns:
        Dict with default Research capability configurations
    """
    provider = _get_provider()
    return provider.generate_capability_configs()


__all__ = [
    # Handlers
    "configure_source_verification",
    "configure_citation_management",
    "configure_research_quality",
    "configure_literature_analysis",
    "configure_fact_checking",
    # Getters
    "get_source_verification",
    "get_citation_config",
    "get_research_quality",
    "get_literature_config",
    "get_fact_checking_config",
    # Provider class
    "ResearchCapabilityProvider",
    # Capability list for loader
    "CAPABILITIES",
    # Convenience functions
    "get_research_capabilities",
    "create_research_capability_loader",
    # SOLID: Centralized config storage
    "get_capability_configs",
]
