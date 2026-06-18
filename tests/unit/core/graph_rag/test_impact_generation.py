# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

import pytest
from unittest.mock import MagicMock
from victor.core.graph_rag.generation import GraphAwarePromptBuilder
from victor.core.graph_rag.config import PromptConfig
from victor.storage.graph.protocol import GraphNode, GraphEdge

def test_impact_analysis_generation():
    # Setup config
    config = PromptConfig(format_style="hierarchical")
    builder = GraphAwarePromptBuilder(config)
    
    # Mock nodes: a function and its test
    node_func = GraphNode(
        node_id="func1",
        name="my_function",
        type="function",
        file="src/logic.py",
        line=10
    )
    node_test = GraphNode(
        node_id="test1",
        name="test_my_function",
        type="test",
        file="tests/test_logic.py",
        line=5
    )
    
    # Mock retrieval result
    mock_result = MagicMock()
    mock_result.nodes = [node_func, node_test]
    mock_result.edges = [
        GraphEdge(src="test1", dst="func1", type="CALLS")
    ]
    mock_result.hop_distances = {"func1": 0, "test1": 1}
    
    # Build prompt
    prompt = builder.build_prompt("How to fix my_function?", mock_result)
    
    # Assertions
    assert "### Impact & Potential Regressions" in prompt
    assert "test_my_function" in prompt
    assert "src/logic.py" in prompt
    assert "tests/test_logic.py" in prompt
