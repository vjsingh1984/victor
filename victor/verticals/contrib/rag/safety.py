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

"""Compatibility shim for the primary RAG runtime safety module."""

from victor.verticals.contrib.rag.runtime.safety import (
    HIGH,
    LOW,
    MEDIUM,
    INGESTION_SAFETY_PATTERNS,
    RAG_DANGER_PATTERNS,
    SECRET_PATTERNS,
    RAGSafetyExtension,
    create_all_rag_safety_rules,
    create_rag_deletion_safety_rules,
    create_rag_ingestion_safety_rules,
)

__all__ = [
    "RAGSafetyExtension",
    "RAG_DANGER_PATTERNS",
    "INGESTION_SAFETY_PATTERNS",
    "SECRET_PATTERNS",
    "HIGH",
    "MEDIUM",
    "LOW",
    "create_rag_deletion_safety_rules",
    "create_rag_ingestion_safety_rules",
    "create_all_rag_safety_rules",
]
