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

"""Microsoft Teams integration tool.

This tool provides:
1. Sending messages to channels
2. Searching for messages
3. Listing teams and channels
4. Creating channels

Requires Microsoft Graph API credentials:
- TEAMS_CLIENT_ID: Azure AD application client ID
- TEAMS_CLIENT_SECRET: Azure AD application client secret
- TEAMS_TENANT_ID: Azure AD tenant ID

Setup:
1. Register an app in Azure AD (https://portal.azure.com)
2. Add Microsoft Graph API permissions:
   - ChannelMessage.Send
   - Channel.ReadBasic.All
   - Team.ReadBasic.All
   - ChannelMessage.Read.All (for search)
3. Grant admin consent for the permissions
4. Create a client secret
"""

import logging
from typing import Any, Dict, List, Optional

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore
    HTTPX_AVAILABLE = False

from victor.tools.base import AccessMode, CostTier, DangerLevel, Priority
from victor.tools.decorators import tool

logger = logging.getLogger(__name__)

_config: Dict[str, Optional[str]] = {
    "client_id": None,
    "client_secret": None,
    "tenant_id": None,
    "access_token": None,
}

_GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"


def set_teams_config(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> bool:
    """Set Microsoft Teams configuration.

    Args:
        client_id: Azure AD application client ID
        client_secret: Azure AD application client secret
        tenant_id: Azure AD tenant ID

    Returns:
        True if configuration was successful, False otherwise
    """
    if not HTTPX_AVAILABLE:
        logger.error("httpx not available. Install with: pip install httpx")
        return False

    if client_id:
        _config["client_id"] = client_id
    if client_secret:
        _config["client_secret"] = client_secret
    if tenant_id:
        _config["tenant_id"] = tenant_id

    # Try to get an access token
    if all([_config["client_id"], _config["client_secret"], _config["tenant_id"]]):
        return _refresh_access_token()

    return False


def _refresh_access_token() -> bool:
    """Refresh the Microsoft Graph API access token.

    Returns:
        True if token was refreshed successfully
    """
    if not HTTPX_AVAILABLE:
        return False

    token_url = f"https://login.microsoftonline.com/{_config['tenant_id']}/oauth2/v2.0/token"

    try:
        with httpx.Client() as client:
            response = client.post(
                token_url,
                data={
                    "client_id": _config["client_id"],
                    "client_secret": _config["client_secret"],
                    "scope": "https://graph.microsoft.com/.default",
                    "grant_type": "client_credentials",
                },
            )
            response.raise_for_status()
            data = response.json()
            _config["access_token"] = data.get("access_token")
            logger.info("Successfully obtained Microsoft Graph API access token")
            return True
    except Exception as e:
        logger.error(f"Failed to get access token: {e}")
        return False


def is_teams_configured() -> bool:
    """Check if Teams client is configured and has valid token."""
    return _config["access_token"] is not None


def _get_headers() -> Dict[str, str]:
    """Get headers for Graph API requests."""
    return {
        "Authorization": f"Bearer {_config['access_token']}",
        "Content-Type": "application/json",
    }


@tool(
    cost_tier=CostTier.MEDIUM,
    category="collaboration",  # Same category as slack for grouping
    priority=Priority.MEDIUM,
    access_mode=AccessMode.NETWORK,
    danger_level=DangerLevel.LOW,
    progress_params=["query"],
    stages=["completion", "execution"],  # Same stages as slack
    task_types=["action", "search"],
    execution_category="network",
    keywords=["teams", "microsoft", "message", "chat", "channel", "notification", "collaboration"],
    mandatory_keywords=["send teams message", "post to teams", "teams channel", "microsoft teams"],
    availability_check=is_teams_configured,  # Only available when configured
)
async def teams(
    operation: str,
    team_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    text: Optional[str] = None,
    query: Optional[str] = None,
    channel_name: Optional[str] = None,
    channel_description: Optional[str] = None,
) -> Dict[str, Any]:
    """Perform operations on Microsoft Teams.

    Args:
        operation: The operation to perform: 'send_message', 'search_messages',
                   'list_teams', 'list_channels', 'create_channel'.
        team_id: The team ID for operations that require it.
        channel_id: The channel ID for operations that require it.
        text: The text of the message for 'send_message'.
        query: The query for searching messages for 'search_messages'.
        channel_name: Name for 'create_channel' operation.
        channel_description: Description for 'create_channel' operation.

    Returns:
        A dictionary with the result of the operation.
    """
    if not HTTPX_AVAILABLE:
        return {"success": False, "error": "httpx not installed. Install with: pip install httpx"}

    if not _config["access_token"]:
        return {
            "success": False,
            "error": "Teams client is not configured. Call set_teams_config() first.",
        }

    try:
        async with httpx.AsyncClient() as client:
            if operation == "send_message":
                return await _send_message(client, team_id, channel_id, text)

            elif operation == "search_messages":
                return await _search_messages(client, query)

            elif operation == "list_teams":
                return await _list_teams(client)

            elif operation == "list_channels":
                return await _list_channels(client, team_id)

            elif operation == "create_channel":
                return await _create_channel(client, team_id, channel_name, channel_description)

            else:
                return {
                    "success": False,
                    "error": f"Unsupported operation: {operation}. "
                    f"Valid operations: send_message, search_messages, "
                    f"list_teams, list_channels, create_channel",
                }

    except httpx.HTTPStatusError as e:
        logger.error(f"[teams] HTTP error during operation '{operation}': {e}")
        return {"success": False, "error": f"HTTP error: {e.response.status_code}"}
    except Exception as e:
        logger.error(f"[teams] Unexpected error during operation '{operation}': {e}")
        return {"success": False, "error": str(e)}


async def _send_message(
    client: "httpx.AsyncClient",
    team_id: Optional[str],
    channel_id: Optional[str],
    text: Optional[str],
) -> Dict[str, Any]:
    """Send a message to a Teams channel."""
    if not team_id or not channel_id or not text:
        return {
            "success": False,
            "error": "Missing required parameters: team_id, channel_id, and text",
        }

    logger.info(f"[teams] Sending message to team '{team_id}' channel '{channel_id}'")

    url = f"{_GRAPH_API_BASE}/teams/{team_id}/channels/{channel_id}/messages"
    response = await client.post(
        url,
        headers=_get_headers(),
        json={
            "body": {
                "content": text,
            }
        },
    )
    response.raise_for_status()
    data = response.json()

    return {
        "success": True,
        "message_id": data.get("id"),
        "created_datetime": data.get("createdDateTime"),
        "message": "Message sent successfully",
    }


async def _search_messages(
    client: "httpx.AsyncClient",
    query: Optional[str],
) -> Dict[str, Any]:
    """Search for messages across Teams."""
    if not query:
        return {"success": False, "error": "Missing required parameter: query"}

    logger.info(f"[teams] Searching messages with query='{query}'")

    # Microsoft Graph search API
    url = f"{_GRAPH_API_BASE}/search/query"
    response = await client.post(
        url,
        headers=_get_headers(),
        json={
            "requests": [
                {
                    "entityTypes": ["chatMessage"],
                    "query": {"queryString": query},
                    "from": 0,
                    "size": 10,
                }
            ]
        },
    )
    response.raise_for_status()
    data = response.json()

    # Parse search results
    results: List[Dict[str, Any]] = []
    hits_containers = data.get("value", [])
    for container in hits_containers:
        hits = container.get("hitsContainers", [])
        for hit_container in hits:
            for hit in hit_container.get("hits", []):
                resource = hit.get("resource", {})
                results.append(
                    {
                        "id": resource.get("id"),
                        "summary": hit.get("summary", "")[:200],
                        "created_datetime": resource.get("createdDateTime"),
                        "from": resource.get("from", {})
                        .get("user", {})
                        .get("displayName", "Unknown"),
                    }
                )

    return {"success": True, "results": results, "count": len(results)}


async def _list_teams(client: "httpx.AsyncClient") -> Dict[str, Any]:
    """List all teams the app has access to."""
    logger.info("[teams] Listing teams")

    url = f"{_GRAPH_API_BASE}/groups?$filter=resourceProvisioningOptions/Any(x:x eq 'Team')"
    response = await client.get(url, headers=_get_headers())
    response.raise_for_status()
    data = response.json()

    teams_data = data.get("value", [])
    results: List[Dict[str, Any]] = [
        {
            "id": team.get("id"),
            "name": team.get("displayName"),
            "description": team.get("description", ""),
            "mail": team.get("mail", ""),
        }
        for team in teams_data[:20]  # Limit to 20 teams
    ]

    return {"success": True, "results": results, "count": len(results)}


async def _list_channels(
    client: "httpx.AsyncClient",
    team_id: Optional[str],
) -> Dict[str, Any]:
    """List channels in a team."""
    if not team_id:
        return {"success": False, "error": "Missing required parameter: team_id"}

    logger.info(f"[teams] Listing channels for team '{team_id}'")

    url = f"{_GRAPH_API_BASE}/teams/{team_id}/channels"
    response = await client.get(url, headers=_get_headers())
    response.raise_for_status()
    data = response.json()

    channels_data = data.get("value", [])
    results: List[Dict[str, Any]] = [
        {
            "id": channel.get("id"),
            "name": channel.get("displayName"),
            "description": channel.get("description", ""),
            "membership_type": channel.get("membershipType", "standard"),
        }
        for channel in channels_data[:20]  # Limit to 20 channels
    ]

    return {"success": True, "results": results, "count": len(results)}


async def _create_channel(
    client: "httpx.AsyncClient",
    team_id: Optional[str],
    channel_name: Optional[str],
    channel_description: Optional[str],
) -> Dict[str, Any]:
    """Create a new channel in a team."""
    if not team_id or not channel_name:
        return {
            "success": False,
            "error": "Missing required parameters: team_id and channel_name",
        }

    logger.info(f"[teams] Creating channel '{channel_name}' in team '{team_id}'")

    url = f"{_GRAPH_API_BASE}/teams/{team_id}/channels"
    response = await client.post(
        url,
        headers=_get_headers(),
        json={
            "displayName": channel_name,
            "description": channel_description or "",
            "membershipType": "standard",
        },
    )
    response.raise_for_status()
    data = response.json()

    return {
        "success": True,
        "channel_id": data.get("id"),
        "name": data.get("displayName"),
        "message": f"Channel '{channel_name}' created successfully",
    }
