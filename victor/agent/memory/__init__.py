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

"""Advanced memory systems for agentic AI.

This package provides episodic and semantic memory systems for agents:
- EpisodicMemory: Stores and retrieves agent experiences (episodes)
- SemanticMemory: Stores and queries factual knowledge
- Memory consolidation: Converts episodic to semantic knowledge

Usage:
    from victor.agent.memory import EpisodicMemory, SemanticMemory

    # Create episodic memory
    episodic = EpisodicMemory()
    episode_id = episodic.store_episode(episode)

    # Recall relevant episodes
    relevant = episodic.recall_relevant("fix authentication bug")

    # Create semantic memory
    semantic = SemanticMemory()
    fact_id = semantic.store_knowledge("Python uses asyncio for concurrency")

    # Query knowledge
    facts = semantic.query_knowledge("concurrency in Python")
"""

from victor.agent.memory.episodic_memory import (
    Episode,
    EpisodeMemory,
    EpisodicMemory,
    MemoryIndex,
    MemoryStats,
    create_episodic_memory,
)
from victor.agent.memory.semantic_memory import (
    Knowledge,
    KnowledgeGraph,
    KnowledgeLink,
    KnowledgeTriple,
    SemanticMemory,
    create_semantic_memory,
)

__all__ = [
    "EpisodicMemory",
    "Episode",
    "EpisodeMemory",
    "MemoryIndex",
    "MemoryStats",
    "create_episodic_memory",
    "SemanticMemory",
    "Knowledge",
    "KnowledgeLink",
    "KnowledgeTriple",
    "KnowledgeGraph",
    "create_semantic_memory",
]
