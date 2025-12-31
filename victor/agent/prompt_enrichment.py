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

"""Prompt Enrichment Service for auto prompt optimization.

DEPRECATED: This module now re-exports from victor.framework.enrichment.
Please import directly from victor.framework.enrichment instead:

    from victor.framework.enrichment import (
        PromptEnrichmentService,
        EnrichmentContext,
        EnrichmentType,
    )

This re-export is maintained for backward compatibility only.
"""

# Re-export from framework for backward compatibility
from victor.framework.enrichment import (
    # Enums
    EnrichmentType,
    EnrichmentPriority,
    # Data Classes
    EnrichmentContext,
    ContextEnrichment,
    EnrichedPrompt,
    EnrichmentOutcome,
    # Protocols
    EnrichmentStrategyProtocol,
    # Services
    PromptEnrichmentService,
    EnrichmentCache,
)

__all__ = [
    # Enums
    "EnrichmentType",
    "EnrichmentPriority",
    # Data Classes
    "EnrichmentContext",
    "ContextEnrichment",
    "EnrichedPrompt",
    "EnrichmentOutcome",
    # Protocols
    "EnrichmentStrategyProtocol",
    # Services
    "PromptEnrichmentService",
    "EnrichmentCache",
]
