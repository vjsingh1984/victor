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

"""Jira integration tool.

This tool provides:
1. Issue creation
2. Issue retrieval
3. Issue search
4. Commenting on issues
"""

import logging
from typing import Any, Dict, Optional

try:
    from jira import JIRA, JIRAError

    JIRA_AVAILABLE = True
except ImportError:
    JIRA = None  # type: ignore
    JIRAError = Exception  # type: ignore
    JIRA_AVAILABLE = False

from victor.tools.base import AccessMode, CostTier, DangerLevel, Priority
from victor.tools.decorators import tool

logger = logging.getLogger(__name__)

_jira_client: Optional["JIRA"] = None
_config: Dict[str, Optional[str]] = {
    "server": None,
    "username": None,
    "api_token": None,
}


def set_jira_config(
    server: Optional[str] = None,
    username: Optional[str] = None,
    api_token: Optional[str] = None,
) -> bool:
    """Set Jira configuration.

    DEPRECATED: Use ToolConfig via executor context instead.
    This function will be removed in v2.0.

    Args:
        server: Jira server URL (e.g., https://company.atlassian.net)
        username: Jira username (email)
        api_token: Jira API token

    Returns:
        True if configuration was successful, False otherwise
    """
    import warnings

    warnings.warn(
        "set_jira_config() is deprecated. Use ToolConfig via executor.update_context() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    global _jira_client

    if not JIRA_AVAILABLE:
        logger.error("Jira library not available. Install with: pip install jira")
        return False

    if server:
        _config["server"] = server
    if username:
        _config["username"] = username
    if api_token:
        _config["api_token"] = api_token

    if _config["server"] and _config["username"] and _config["api_token"]:
        try:
            _jira_client = JIRA(
                server=_config["server"],
                basic_auth=(_config["username"], _config["api_token"]),
            )
            logger.info(f"Successfully connected to Jira at {_config['server']}")
            return True
        except JIRAError as e:
            logger.error(f"Failed to connect to Jira: {e}")
            _jira_client = None
            return False
    return False


def is_jira_configured() -> bool:
    """Check if Jira client is configured and connected."""
    return _jira_client is not None


@tool(
    cost_tier=CostTier.MEDIUM,
    category="jira",
    priority=Priority.MEDIUM,
    access_mode=AccessMode.NETWORK,
    danger_level=DangerLevel.LOW,
    progress_params=["jql"],
    stages=["planning", "execution"],
    task_types=["action", "search"],
    execution_category="network",
    keywords=["jira", "issue", "ticket", "bug", "task", "sprint", "backlog", "story"],
    mandatory_keywords=["search issues", "find tickets", "create ticket", "jira search"],
)
async def jira(
    operation: str,
    jql: Optional[str] = None,
    issue_key: Optional[str] = None,
    summary: Optional[str] = None,
    project: Optional[str] = None,
    issue_type: str = "Task",
    description: Optional[str] = None,
    comment: Optional[str] = None,
    max_results: int = 10,
) -> Dict[str, Any]:
    """Perform operations on Jira issues.

    Args:
        operation: The operation to perform: 'search_issues', 'create_issue', 'get_issue', 'add_comment'.
        jql: The JQL query for searching issues (required for search_issues).
        issue_key: The key of the issue (e.g., 'PROJ-123') for 'get_issue' and 'add_comment'.
        summary: The summary/title of the issue for 'create_issue'.
        project: The project key for 'create_issue' (e.g., 'PROJ').
        issue_type: The type of issue for 'create_issue' (default: 'Task'). Options: Task, Bug, Story, Epic.
        description: Optional description for 'create_issue'.
        comment: The comment to add for 'add_comment'.
        max_results: The maximum number of results to return for 'search_issues' (default: 10).

    Returns:
        A dictionary with the result of the operation.
    """
    if not JIRA_AVAILABLE:
        return {
            "success": False,
            "error": "Jira library not installed. Install with: pip install jira",
        }

    if not _jira_client:
        return {
            "success": False,
            "error": "Jira client is not configured. Call set_jira_config() first.",
        }

    try:
        if operation == "search_issues":
            if not jql:
                return {"success": False, "error": "Missing required parameter: jql"}
            logger.info(f"[jira] Searching issues with jql='{jql}'")
            issues = _jira_client.search_issues(jql, maxResults=max_results)
            results = [
                {
                    "key": issue.key,
                    "summary": issue.fields.summary,
                    "status": issue.fields.status.name,
                    "assignee": getattr(issue.fields.assignee, "displayName", "Unassigned"),
                    "priority": getattr(issue.fields.priority, "name", "None"),
                }
                for issue in issues
            ]
            return {"success": True, "results": results, "count": len(results)}

        elif operation == "create_issue":
            if not project or not summary:
                return {
                    "success": False,
                    "error": "Missing required parameters: project and summary",
                }
            logger.info(f"[jira] Creating issue in project '{project}' with summary '{summary}'")
            issue_dict: Dict[str, Any] = {
                "project": {"key": project},
                "summary": summary,
                "issuetype": {"name": issue_type},
            }
            if description:
                issue_dict["description"] = description
            new_issue = _jira_client.create_issue(fields=issue_dict)
            return {
                "success": True,
                "key": new_issue.key,
                "summary": new_issue.fields.summary,
                "url": f"{_config['server']}/browse/{new_issue.key}",
            }

        elif operation == "get_issue":
            if not issue_key:
                return {"success": False, "error": "Missing required parameter: issue_key"}
            logger.info(f"[jira] Getting issue '{issue_key}'")
            issue = _jira_client.issue(issue_key)
            comments = []
            if hasattr(issue.fields, "comment") and issue.fields.comment:
                comments = [
                    {
                        "author": getattr(comment.author, "displayName", "Unknown"),
                        "body": comment.body,
                        "created": str(comment.created),
                    }
                    for comment in issue.fields.comment.comments[:5]  # Limit to 5 most recent
                ]
            return {
                "success": True,
                "key": issue.key,
                "summary": issue.fields.summary,
                "status": issue.fields.status.name,
                "description": issue.fields.description or "",
                "assignee": getattr(issue.fields.assignee, "displayName", "Unassigned"),
                "reporter": getattr(issue.fields.reporter, "displayName", "Unknown"),
                "priority": getattr(issue.fields.priority, "name", "None"),
                "comments": comments,
                "url": f"{_config['server']}/browse/{issue.key}",
            }

        elif operation == "add_comment":
            if not issue_key or not comment:
                return {
                    "success": False,
                    "error": "Missing required parameters: issue_key and comment",
                }
            logger.info(f"[jira] Adding comment to issue '{issue_key}'")
            _jira_client.add_comment(issue_key, comment)
            return {"success": True, "message": f"Comment added to {issue_key}"}

        else:
            return {
                "success": False,
                "error": f"Unsupported operation: {operation}. "
                f"Valid operations: search_issues, create_issue, get_issue, add_comment",
            }

    except JIRAError as e:
        logger.error(f"[jira] Error during operation '{operation}': {e}")
        return {"success": False, "error": str(getattr(e, "text", str(e)))}
    except Exception as e:
        logger.error(f"[jira] Unexpected error during operation '{operation}': {e}")
        return {"success": False, "error": str(e)}
