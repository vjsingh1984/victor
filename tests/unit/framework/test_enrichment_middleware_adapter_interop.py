"""Interop tests for SDK enrichment and middleware host adapters."""

from victor.agent.code_correction_middleware import (
    CodeCorrectionConfig as CoreCodeCorrectionConfig,
    CodeCorrectionMiddleware as CoreCodeCorrectionMiddleware,
    CorrectionResult as CoreCorrectionResult,
)
from victor.evaluation.correction.types import (
    CodeValidationResult as CoreCodeValidationResult,
    Language as CoreLanguage,
)
from victor.framework.enrichment import (
    ContextEnrichment as CoreContextEnrichment,
    EnrichmentContext as CoreEnrichmentContext,
    FilePatternMatcher as CoreFilePatternMatcher,
    KeywordClassifier as CoreKeywordClassifier,
    extract_identifiers as core_extract_identifiers,
)
from victor.framework.middleware import (
    GitSafetyMiddleware as CoreGitSafetyMiddleware,
    MiddlewareComposer as CoreMiddlewareComposer,
)
from victor_sdk.enrichment_runtime import (
    ContextEnrichment as SdkContextEnrichment,
    EnrichmentContext as SdkEnrichmentContext,
    FilePatternMatcher as SdkFilePatternMatcher,
    KeywordClassifier as SdkKeywordClassifier,
    extract_identifiers as sdk_extract_identifiers,
)
from victor_sdk.middleware_runtime import (
    CodeCorrectionConfig as SdkCodeCorrectionConfig,
    CodeCorrectionMiddleware as SdkCodeCorrectionMiddleware,
    CodeValidationResult as SdkCodeValidationResult,
    CorrectionResult as SdkCorrectionResult,
    GitSafetyMiddleware as SdkGitSafetyMiddleware,
    Language as SdkLanguage,
    MiddlewareComposer as SdkMiddlewareComposer,
)


def test_enrichment_runtime_identity_is_shared() -> None:
    assert CoreContextEnrichment is SdkContextEnrichment
    assert CoreEnrichmentContext is SdkEnrichmentContext
    assert CoreFilePatternMatcher is SdkFilePatternMatcher
    assert CoreKeywordClassifier is SdkKeywordClassifier
    assert core_extract_identifiers is sdk_extract_identifiers


def test_middleware_runtime_identity_is_shared() -> None:
    assert CoreCodeCorrectionConfig is SdkCodeCorrectionConfig
    assert CoreCodeCorrectionMiddleware is SdkCodeCorrectionMiddleware
    assert CoreCodeValidationResult is SdkCodeValidationResult
    assert CoreCorrectionResult is SdkCorrectionResult
    assert CoreGitSafetyMiddleware is SdkGitSafetyMiddleware
    assert CoreLanguage is SdkLanguage
    assert CoreMiddlewareComposer is SdkMiddlewareComposer
