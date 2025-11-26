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

"""
CodingAgent - A universal terminal-based coding agent supporting multiple LLM providers.

Supports frontier models (Claude, GPT, Gemini) and open-source models
(Ollama, LMStudio, vLLM) with unified tool calling integration.
"""

__version__ = "0.1.0"
__author__ = "Vijaykumar Singh"
__email__ = "singhvjd@gmail.com"
__license__ = "Apache-2.0"

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings

__all__ = ["AgentOrchestrator", "Settings"]
