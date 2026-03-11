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

"""Compatibility shim for the RAG runtime capabilities module."""

from victor.verticals.contrib.rag.runtime.capabilities import (
    CAPABILITIES,
    CapabilityMetadata,
    RAGCapabilityProvider,
    configure_indexing,
    configure_query_enhancement,
    configure_retrieval,
    configure_safety,
    configure_synthesis,
    create_rag_capability_loader,
    get_capability_configs,
    get_indexing_config,
    get_rag_capabilities,
    get_retrieval_config,
    get_synthesis_config,
)

__all__ = [
    "configure_indexing",
    "configure_retrieval",
    "configure_synthesis",
    "configure_safety",
    "configure_query_enhancement",
    "get_indexing_config",
    "get_retrieval_config",
    "get_synthesis_config",
    "RAGCapabilityProvider",
    "CapabilityMetadata",
    "CAPABILITIES",
    "get_rag_capabilities",
    "create_rag_capability_loader",
    "get_capability_configs",
]
