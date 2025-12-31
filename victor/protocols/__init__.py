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

"""This module has moved to victor.integrations.protocols.

This stub provides backward compatibility. Please update imports to use:
    from victor.integrations.protocols import ...
"""

# Re-export all public symbols for backward compatibility
from victor.integrations.protocols import (
    # Provider Adapter
    IProviderAdapter,
    ProviderCapabilities,
    ToolCallFormat,
    ToolCall,
    get_provider_adapter,
    register_provider_adapter,
    # Grounding
    IGroundingStrategy,
    GroundingClaimType,
    GroundingClaim,
    VerificationResult,
    AggregatedVerificationResult,
    FileExistenceStrategy,
    SymbolReferenceStrategy,
    ContentMatchStrategy,
    CompositeGroundingVerifier,
    # Quality
    IQualityAssessor,
    QualityDimension,
    DimensionScore,
    QualityScore,
    BaseQualityAssessor,
    SimpleQualityAssessor,
    ProviderAwareQualityAssessor,
    CompositeQualityAssessor,
    # Mode Awareness
    IModeController,
    ModeInfo,
    ModeAwareMixin,
    create_mode_aware_mixin,
    # Path Resolution
    IPathResolver,
    PathResolution,
    PathResolver,
    create_path_resolver,
    # LSP Types - Enumerations
    DiagnosticSeverity,
    CompletionItemKind,
    SymbolKind,
    DiagnosticTag,
    # LSP Types - Position and Range
    Position,
    Range,
    Location,
    LocationLink,
    # LSP Types - Diagnostics
    DiagnosticRelatedInformation,
    Diagnostic,
    # LSP Types - Completions
    CompletionItem,
    # LSP Types - Hover
    Hover,
    # LSP Types - Symbols
    DocumentSymbol,
    SymbolInformation,
    # LSP Types - Text Edits
    TextEdit,
    TextDocumentIdentifier,
    VersionedTextDocumentIdentifier,
    TextDocumentEdit,
)

__all__ = [
    # Provider Adapter
    "IProviderAdapter",
    "ProviderCapabilities",
    "ToolCallFormat",
    "ToolCall",
    "get_provider_adapter",
    "register_provider_adapter",
    # Grounding
    "IGroundingStrategy",
    "GroundingClaimType",
    "GroundingClaim",
    "VerificationResult",
    "AggregatedVerificationResult",
    "FileExistenceStrategy",
    "SymbolReferenceStrategy",
    "ContentMatchStrategy",
    "CompositeGroundingVerifier",
    # Quality
    "IQualityAssessor",
    "QualityDimension",
    "DimensionScore",
    "QualityScore",
    "BaseQualityAssessor",
    "SimpleQualityAssessor",
    "ProviderAwareQualityAssessor",
    "CompositeQualityAssessor",
    # Mode Awareness
    "IModeController",
    "ModeInfo",
    "ModeAwareMixin",
    "create_mode_aware_mixin",
    # Path Resolution
    "IPathResolver",
    "PathResolution",
    "PathResolver",
    "create_path_resolver",
    # LSP Types - Enumerations
    "DiagnosticSeverity",
    "CompletionItemKind",
    "SymbolKind",
    "DiagnosticTag",
    # LSP Types - Position and Range
    "Position",
    "Range",
    "Location",
    "LocationLink",
    # LSP Types - Diagnostics
    "DiagnosticRelatedInformation",
    "Diagnostic",
    # LSP Types - Completions
    "CompletionItem",
    # LSP Types - Hover
    "Hover",
    # LSP Types - Symbols
    "DocumentSymbol",
    "SymbolInformation",
    # LSP Types - Text Edits
    "TextEdit",
    "TextDocumentIdentifier",
    "VersionedTextDocumentIdentifier",
    "TextDocumentEdit",
]
