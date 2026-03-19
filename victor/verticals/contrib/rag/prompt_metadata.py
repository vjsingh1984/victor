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

"""Serializable prompt metadata for the RAG vertical."""

from __future__ import annotations

from typing import Any, Dict

RAG_TASK_TYPE_HINTS: Dict[str, Dict[str, Any]] = {
    "document_ingestion": {
        "hint": "[RAG INGEST] Verify the source first, then ingest and report chunk counts.",
        "tool_budget": 5,
        "priority_tools": ["rag_ingest", "read", "ls"],
        "description": "Ingesting documents into the knowledge base",
        "grounding_rules": [
            "Always confirm the file exists before ingestion.",
            "Report the number of chunks created.",
            "Suggest a relevant document type when none is specified.",
        ],
    },
    "knowledge_search": {
        "hint": "[RAG SEARCH] Search broadly, compare relevance, and cite the retrieved sources.",
        "tool_budget": 3,
        "priority_tools": ["rag_search", "rag_query"],
        "description": "Searching the knowledge base",
        "grounding_rules": [
            "Use specific search terms.",
            "Report relevance scores.",
            "Cite sources in responses.",
        ],
    },
    "question_answering": {
        "hint": "[RAG QA] Retrieve context before answering and ground every claim in sources.",
        "tool_budget": 5,
        "priority_tools": ["rag_query", "rag_search"],
        "description": "Answering questions from the knowledge base",
        "grounding_rules": [
            "Always search before answering.",
            "Cite source documents with [N] notation.",
            "If no relevant context is found, say so.",
            "Do not hallucinate facts outside retrieved sources.",
        ],
    },
    "knowledge_management": {
        "hint": "[RAG MANAGE] Inspect the knowledge base first and confirm destructive actions.",
        "tool_budget": 5,
        "priority_tools": ["rag_list", "rag_delete", "rag_stats"],
        "description": "Managing the knowledge base",
        "grounding_rules": [
            "Confirm before deleting documents.",
            "Show document details before deletion.",
        ],
    },
}

RAG_SYSTEM_PROMPT_SECTION = """
## RAG Operations

When answering questions:
1. Always use rag_query first to retrieve relevant context
2. If context is insufficient, use rag_search for broader exploration
3. Cite sources using [Source N] notation from the retrieved context
4. If the answer isn't in the knowledge base, clearly state this

When ingesting documents:
1. Verify the file exists using read or ls
2. Use appropriate doc_type (text, markdown, code, pdf)
3. Report successful ingestion with chunk count

Citation Format:
- Use [1], [2] etc. to reference sources from rag_query results
- Include the source name when summarizing multiple sources
""".strip()

RAG_PROMPT_TEMPLATES: Dict[str, str] = {
    "rag_operations": RAG_SYSTEM_PROMPT_SECTION,
}

__all__ = [
    "RAG_PROMPT_TEMPLATES",
    "RAG_SYSTEM_PROMPT_SECTION",
    "RAG_TASK_TYPE_HINTS",
]
