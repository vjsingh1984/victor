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

"""This module has moved to victor.integrations.mcp.

This stub provides backward compatibility. Please update imports to use:
    from victor.integrations.mcp import ...
"""

# Re-export all public symbols for backward compatibility
from victor.integrations.mcp import (
    MCPServer,
    MCPClient,
    MCPRegistry,
    MCPServerConfig,
    ServerStatus,
    MCPMessage,
    MCPTool,
    MCPResource,
)

__all__ = [
    "MCPServer",
    "MCPClient",
    "MCPRegistry",
    "MCPServerConfig",
    "ServerStatus",
    "MCPMessage",
    "MCPTool",
    "MCPResource",
]
