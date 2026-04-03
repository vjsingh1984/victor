"""Compatibility shim for Research enhanced runtime safety helpers."""

from victor.verticals.contrib.research.runtime.safety_enhanced import (
    EnhancedResearchSafetyExtension,
    ResearchSafetyRules,
)

__all__ = ["ResearchSafetyRules", "EnhancedResearchSafetyExtension"]
