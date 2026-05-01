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

"""Service-owned host for active runtime support protocols.

These protocols already exist in ``victor.agent.protocols``. This module keeps
their identity stable while moving the canonical import host for the service-
first DI and factory layers under ``victor.agent.services.protocols``.
"""

from victor.agent.protocols.analysis_protocols import IntentClassifierProtocol
from victor.agent.protocols.infrastructure_protocols import (
    ReminderManagerProtocol,
    ResponseSanitizerProtocol,
)
from victor.agent.protocols.streaming_protocols import (
    StreamingConfidenceMonitorProtocol,
    StreamingHandlerProtocol,
    StreamingMetricsCollectorProtocol,
)

__all__ = [
    "IntentClassifierProtocol",
    "ReminderManagerProtocol",
    "ResponseSanitizerProtocol",
    "StreamingConfidenceMonitorProtocol",
    "StreamingHandlerProtocol",
    "StreamingMetricsCollectorProtocol",
]
