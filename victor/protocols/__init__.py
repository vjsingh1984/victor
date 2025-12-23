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

"""Victor Protocols Module.

Provides SOLID-based protocol interfaces for:
- Provider adaptation (IProviderAdapter)
- Grounding verification (IGroundingStrategy)
- Quality assessment (IQualityAssessor)

These protocols enable clean separation of concerns and
dependency inversion throughout the codebase.
"""

from victor.protocols.provider_adapter import (
    IProviderAdapter,
    ProviderCapabilities,
    ToolCallFormat,
    ToolCall,
    get_provider_adapter,
    register_provider_adapter,
)

from victor.protocols.grounding import (
    IGroundingStrategy,
    GroundingClaimType,
    GroundingClaim,
    VerificationResult,
    AggregatedVerificationResult,
    FileExistenceStrategy,
    SymbolReferenceStrategy,
    ContentMatchStrategy,
    CompositeGroundingVerifier,
)

from victor.protocols.quality import (
    IQualityAssessor,
    QualityDimension,
    DimensionScore,
    QualityScore,
    BaseQualityAssessor,
    SimpleQualityAssessor,
    ProviderAwareQualityAssessor,
    CompositeQualityAssessor,
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
]
