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

"""HTTP API Module for Victor.

Provides REST API endpoints for IDE integrations (VS Code, JetBrains, etc.)
and external tool access.

Components:
- VictorAPIServer: Legacy aiohttp-based server
- VictorFastAPIServer: Modern FastAPI-based server (recommended)
- APIMiddlewareStack: Authentication, rate limiting, and CORS middleware
- EventBridge: Real-time event streaming to WebSocket clients
"""

from victor.integrations.api.server import VictorAPIServer
from victor.integrations.api.fastapi_server import VictorFastAPIServer
from victor.integrations.api.middleware import (
    APIMiddlewareStack,
    RateLimitConfig,
    AuthConfig,
    TokenBucket,
)
from victor.integrations.api.event_bridge import (
    EventBroadcaster,
    BridgeEvent,
    BridgeEventType,
)

__all__ = [
    # Servers
    "VictorAPIServer",
    "VictorFastAPIServer",
    # Middleware
    "APIMiddlewareStack",
    "RateLimitConfig",
    "AuthConfig",
    "TokenBucket",
    # Event Bridge
    "EventBroadcaster",
    "BridgeEvent",
    "BridgeEventType",
]
