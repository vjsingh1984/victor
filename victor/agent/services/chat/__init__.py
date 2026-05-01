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

"""Canonical chat helper services.

This package contains helper components used to decompose chat behavior, but
the canonical runtime-owned ``ChatService`` lives in
``victor.agent.services.chat_service`` and is exported from
``victor.agent.services``.
"""

from victor.agent.services.chat.chat_flow_service import ChatFlowService, ChatFlowServiceConfig
from victor.agent.services.chat.response_aggregation_service import (
    ResponseAggregationService,
    ResponseAggregationServiceConfig,
)
from victor.agent.services.chat.streaming_service import StreamingService, StreamingServiceConfig
from victor.agent.services.chat.continuation_service import (
    ContinuationService,
    ContinuationServiceConfig,
)

__all__ = [
    "ChatFlowService",
    "ChatFlowServiceConfig",
    "ResponseAggregationService",
    "ResponseAggregationServiceConfig",
    "StreamingService",
    "StreamingServiceConfig",
    "ContinuationService",
    "ContinuationServiceConfig",
]
