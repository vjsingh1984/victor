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

"""Service-owned system prompt compatibility runtime.

.. deprecated::
    Superseded by ``victor.agent.prompt_pipeline.UnifiedPromptPipeline``.
    This class remains a backward-compat wrapper. New code should use
    UnifiedPromptPipeline directly.

For state-passed orchestration boundaries, prefer
``victor.agent.coordinators.SystemPromptStatePassedCoordinator`` or the
matching ``OrchestrationFacade.system_prompt_state_passed`` surface.

The legacy `victor.agent.coordinators.system_prompt_coordinator` module now
re-exports this implementation for compatibility.
"""

from __future__ import annotations

from victor.agent.services.prompt_runtime_support import PromptRuntimeSupport

class SystemPromptCoordinator(PromptRuntimeSupport):
    """Backward-compatible wrapper over PromptRuntimeSupport.

    Kept for callers in component_assembler.py, coordinator_factory.py,
    and coordinators/__init__.py.
    """
