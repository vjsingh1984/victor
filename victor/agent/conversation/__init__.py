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

"""Conversation management components.

This package provides focused components for conversation management,
extracted from ConversationManager to follow the Single Responsibility
Principle (SRP).

Part of SOLID-based refactoring to eliminate god class anti-pattern.
"""

from victor.agent.conversation.message_store import MessageStore
from victor.agent.conversation.context_handler import ContextOverflowHandler
from victor.agent.conversation.session_manager import SessionManager
from victor.agent.conversation.embedding_manager import EmbeddingManager

__all__ = [
    "MessageStore",
    "ContextOverflowHandler",
    "SessionManager",
    "EmbeddingManager",
]
