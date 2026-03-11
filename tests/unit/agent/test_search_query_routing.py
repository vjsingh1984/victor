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

"""Focused tests for search query routing on AgentOrchestrator."""

from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.search_router import SearchRouter


def _make_orchestrator() -> AgentOrchestrator:
    orchestrator = object.__new__(AgentOrchestrator)
    orchestrator.search_router = SearchRouter()
    return orchestrator


def test_route_search_query_includes_bug_mode_arguments() -> None:
    """Bug/regression queries should carry tool arguments for code_search."""
    orchestrator = _make_orchestrator()

    result = orchestrator.route_search_query("json parsing crash on empty payload")

    assert result["recommended_tool"] == "code_search"
    assert result["recommended_args"] == {"mode": "bugs"}
    assert result["search_type"] == "semantic"


def test_get_recommended_search_tool_uses_bug_route_tool_name() -> None:
    """Bug/regression routes should still recommend code_search as the tool name."""
    orchestrator = _make_orchestrator()

    result = orchestrator.get_recommended_search_tool("find similar regressions in auth")

    assert result == "code_search"
