"""Vertical metadata definitions.

This module provides data classes and utilities for managing
vertical metadata and capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from victor_sdk.core.types import (
    CapabilityRequirement,
    CapabilityRequirementLike,
    normalize_capability_requirement,
)


@dataclass(frozen=True)
class VerticalMetadata:
    """Metadata about a vertical.

    This class provides structured metadata that describes a vertical's
    capabilities, requirements, and characteristics.

    Attributes:
        name: Vertical identifier
        description: Human-readable description
        version: Vertical version
        author: Author name or organization
        capabilities: List of capability identifiers
        requirements: List of required capabilities or services
        capability_requirements: Structured capability requirements
        tags: List of tags for categorization
        categories: List of category names
        provider_hints: Hints about which providers to use
        evaluation_criteria: Criteria for evaluating this vertical
    """

    name: str
    description: str
    version: str = "1.0.0"
    author: str = ""
    capabilities: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    capability_requirements: List[CapabilityRequirement] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    provider_hints: Dict[str, str] = field(default_factory=dict)
    evaluation_criteria: Dict[str, Any] = field(default_factory=dict)

    def with_capability(self, capability: str) -> VerticalMetadata:
        """Return metadata with an additional capability."""
        return VerticalMetadata(
            name=self.name,
            description=self.description,
            version=self.version,
            author=self.author,
            capabilities=[*self.capabilities, capability],
            requirements=self.requirements.copy(),
            capability_requirements=self.capability_requirements.copy(),
            tags=self.tags.copy(),
            categories=self.categories.copy(),
            provider_hints=self.provider_hints.copy(),
            evaluation_criteria=self.evaluation_criteria.copy(),
        )

    def with_requirement(self, requirement: CapabilityRequirementLike) -> VerticalMetadata:
        """Return metadata with an additional capability requirement."""

        normalized = normalize_capability_requirement(requirement)
        return VerticalMetadata(
            name=self.name,
            description=self.description,
            version=self.version,
            author=self.author,
            capabilities=self.capabilities.copy(),
            requirements=[*self.requirements, normalized.capability_id],
            capability_requirements=[*self.capability_requirements, normalized],
            tags=self.tags.copy(),
            categories=self.categories.copy(),
            provider_hints=self.provider_hints.copy(),
            evaluation_criteria=self.evaluation_criteria.copy(),
        )

    def get_requirement_names(self) -> List[str]:
        """Return required capability identifiers in legacy string form."""

        if self.capability_requirements:
            return [requirement.capability_id for requirement in self.capability_requirements]
        return self.requirements.copy()

    def with_tag(self, tag: str) -> VerticalMetadata:
        """Return metadata with an additional tag."""
        return VerticalMetadata(
            name=self.name,
            description=self.description,
            version=self.version,
            author=self.author,
            capabilities=self.capabilities.copy(),
            requirements=self.requirements.copy(),
            capability_requirements=self.capability_requirements.copy(),
            tags=[*self.tags, tag],
            categories=self.categories.copy(),
            provider_hints=self.provider_hints.copy(),
            evaluation_criteria=self.evaluation_criteria.copy(),
        )

    def get_all_metadata(self) -> Dict[str, Any]:
        """Return all metadata as a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "capabilities": self.capabilities,
            "requirements": self.requirements,
            "capability_requirements": [
                requirement.to_dict() for requirement in self.capability_requirements
            ],
            "tags": self.tags,
            "categories": self.categories,
            "provider_hints": self.provider_hints,
            "evaluation_criteria": self.evaluation_criteria,
        }


@dataclass
class ProviderHints:
    """Hints for provider selection.

    This class provides hints to the framework about which providers
    to prefer for various services.

    Attributes:
        llm_provider: Preferred LLM provider
        embedding_provider: Preferred embedding provider
        tool_provider: Preferred tool provider
        storage_provider: Preferred storage provider
        cache_provider: Preferred cache provider
        custom_hints: Additional custom hints
    """

    llm_provider: Optional[str] = None
    embedding_provider: Optional[str] = None
    tool_provider: Optional[str] = None
    storage_provider: Optional[str] = None
    cache_provider: Optional[str] = None
    custom_hints: Dict[str, str] = field(default_factory=dict)

    def get_hint(self, service_type: str) -> Optional[str]:
        """Get provider hint for a service type.

        Args:
            service_type: Type of service (e.g., "llm", "embedding")

        Returns:
            Provider name hint or None
        """
        if service_type == "llm":
            return self.llm_provider
        elif service_type == "embedding":
            return self.embedding_provider
        elif service_type == "tool":
            return self.tool_provider
        elif service_type == "storage":
            return self.storage_provider
        elif service_type == "cache":
            return self.cache_provider
        else:
            return self.custom_hints.get(service_type)

    def with_hint(self, service_type: str, provider: str) -> ProviderHints:
        """Return hints with an additional hint."""
        new_hints = {**self.custom_hints, service_type: provider}
        return ProviderHints(
            llm_provider=self.llm_provider,
            embedding_provider=self.embedding_provider,
            tool_provider=self.tool_provider,
            storage_provider=self.storage_provider,
            cache_provider=self.cache_provider,
            custom_hints=new_hints,
        )
