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

"""Prompt composition pipeline (DEPRECATED).

.. deprecated::
    Superseded by ``victor.agent.prompt_pipeline.UnifiedPromptPipeline``.
    All classes are re-exported from ``victor.agent.prompt_pipeline``.
    Import from there for new code.
"""

import warnings

from victor.agent.prompt_pipeline import (  # noqa: F401
    ContentRouter,
    Placement,
    ProviderTier,
    TurnContext,
    UnifiedPromptPipeline,
    detect_provider_tier,
)

# Backward compat alias
PromptComposer = UnifiedPromptPipeline

warnings.warn(
    "victor.agent.prompt_composer is deprecated. "
    "Use victor.agent.prompt_pipeline.UnifiedPromptPipeline instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "PromptComposer",
    "UnifiedPromptPipeline",
    "ProviderTier",
    "Placement",
    "ContentRouter",
    "TurnContext",
    "detect_provider_tier",
]
