"""Tests for SDK enrichment and middleware host adapters."""

from victor_sdk.enrichment_runtime import (
    ContextEnrichment,
    EnrichmentContext,
    FilePatternMatcher,
    KeywordClassifier,
    extract_identifiers,
)
from victor_sdk.middleware_runtime import (
    CodeCorrectionConfig,
    CodeCorrectionMiddleware,
    CodeValidationResult,
    CorrectionResult,
    GitSafetyMiddleware,
    Language,
    MiddlewareComposer,
)


def test_enrichment_runtime_exports_host_helpers() -> None:
    assert ContextEnrichment.__name__ == "ContextEnrichment"
    assert EnrichmentContext.__name__ == "EnrichmentContext"
    assert FilePatternMatcher.__name__ == "FilePatternMatcher"
    assert KeywordClassifier.__name__ == "KeywordClassifier"
    assert callable(extract_identifiers)


def test_middleware_runtime_exports_host_helpers() -> None:
    assert CodeCorrectionConfig.__name__ == "CodeCorrectionConfig"
    assert CodeCorrectionMiddleware.__name__ == "CodeCorrectionMiddleware"
    assert CodeValidationResult.__name__ == "CodeValidationResult"
    assert CorrectionResult.__name__ == "CorrectionResult"
    assert GitSafetyMiddleware.__name__ == "GitSafetyMiddleware"
    assert Language.__name__ == "Language"
    assert MiddlewareComposer.__name__ == "MiddlewareComposer"
