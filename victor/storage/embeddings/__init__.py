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

"""Shared embedding infrastructure for Victor.

This module provides a unified embedding service that supports:
- Tool selection (static, ~65 items)
- Intent classification (static, ~20 items)
- Code search (dynamic, via LanceDB - handled separately)

Design:
- Single EmbeddingService singleton to load the model once
- StaticEmbeddingCollection for pickle/numpy-backed small collections
- Cosine similarity for semantic matching
"""

from victor.storage.embeddings.service import EmbeddingService
from victor.storage.embeddings.collections import StaticEmbeddingCollection
from victor.storage.embeddings.intent_classifier import IntentClassifier
from victor.storage.embeddings.question_classifier import (
    QuestionType,
    QuestionTypeClassifier,
    QuestionClassificationResult,
    classify_question,
    should_auto_continue,
)

__all__ = [
    "EmbeddingService",
    "StaticEmbeddingCollection",
    "IntentClassifier",
    "QuestionType",
    "QuestionTypeClassifier",
    "QuestionClassificationResult",
    "classify_question",
    "should_auto_continue",
]
