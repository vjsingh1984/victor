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

"""RAG Mode Configuration Provider."""

from typing import Dict, Optional

from victor.core.verticals.protocols import ModeConfig, ModeConfigProviderProtocol


# RAG-specific mode definitions
RAG_MODES: Dict[str, ModeConfig] = {
    "fast": ModeConfig(
        name="fast",
        tool_budget=5,
        max_iterations=10,
        description="Quick RAG queries with minimal context",
    ),
    "standard": ModeConfig(
        name="standard",
        tool_budget=10,
        max_iterations=20,
        description="Standard RAG operations",
    ),
    "thorough": ModeConfig(
        name="thorough",
        tool_budget=20,
        max_iterations=40,
        description="Thorough search and analysis",
    ),
    "bulk_ingest": ModeConfig(
        name="bulk_ingest",
        tool_budget=50,
        max_iterations=100,
        description="Bulk document ingestion",
    ),
}


class RAGModeConfigProvider(ModeConfigProviderProtocol):
    """Mode configuration provider for RAG vertical.

    Provides RAG-specific modes and task budgets.
    """

    def get_mode_configs(self) -> Dict[str, ModeConfig]:
        """Get mode configurations for RAG vertical.

        Returns:
            Dict mapping mode names to configurations
        """
        return RAG_MODES

    def get_default_mode(self) -> str:
        """Get the default mode name.

        Returns:
            Default mode name
        """
        return "standard"
