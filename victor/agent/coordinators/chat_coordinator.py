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

"""DEPRECATED: ChatCoordinator has moved to services.

.. deprecated::
    Import ChatCoordinator from victor.agent.services.chat_compat instead.
    This module remains as a backward compatibility redirect.
"""

from __future__ import annotations

import warnings

from victor.agent.services.chat_compat import ChatCoordinator

warnings.warn(
    "victor.agent.coordinators.ChatCoordinator is deprecated compatibility "
    "surface. Prefer ChatService from victor.agent.services.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ChatCoordinator"]
