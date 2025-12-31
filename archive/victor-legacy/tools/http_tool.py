"""HTTP/API testing tool for making HTTP requests and testing APIs.

Features:
- All HTTP methods (GET, POST, PUT, PATCH, DELETE)
- Headers and authentication
- Request/response inspection
- JSON, form data, file uploads
- Response validation
- Performance metrics
"""

import json
import time
from typing import Any, Dict, List, Optional

import httpx

from victor.tools.base import BaseTool, ToolParameter, ToolResult


class HTTPTool(BaseTool):
    """Tool for HTTP requests and API testing."""

    def __init__(self, timeout: int = 30):
        """Initialize HTTP tool.

        Args:
            timeout: Request timeout in seconds
        """
        super().__init__()
        self.timeout = timeout

    @property
    def name(self) -> str:
        """Get tool name."""
        return "http"

    @property
    def description(self) -> str:
        """Get tool description."""
        return """HTTP requests and API testing.

Make HTTP requests to test APIs and web endpoints.

Operations:
- request: Make HTTP request (GET, POST, PUT, PATCH, DELETE)
- test: Test API endpoint with validation

Supports:
- All HTTP methods
- Custom headers
- Authentication (Bearer, Basic)
- JSON and form data
- Query parameters
- Response validation

Example workflows:
1. Simple GET request:
   http(operation="request", method="GET", url="https://api.github.com/users/octocat")

2. POST with JSON:
   http(operation="request", method="POST", url="https://api.example.com/data",
        headers={"Content-Type": "application/json"},
        json={"key": "value"})

3. API testing:
   http(operation="test", method="GET", url="https://api.example.com/health",
        expected_status=200)
"""

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameters."""
        return self.convert_parameters_to_schema(
            [
                ToolParameter(
                    name="operation",
                    type="string",
                    description="Operation: request, test",
                    required=True,
                ),
                ToolParameter(
                    name="method",
                    type="string",
                    description="HTTP method: GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS",
                    required=True,
                ),
                ToolParameter(
                    name="url",
                    type="string",
                    description="Request URL",
                    required=True,
                ),
                ToolParameter(
                    name="headers",
                    type="object",
                    description="Request headers (dict)",
                    required=False,
                ),
                ToolParameter(
                    name="params",
                    type="object",
                    description="Query parameters (dict)",
                    required=False,
                ),
                ToolParameter(
                    name="json",
                    type="object",
                    description="JSON body (dict)",
                    required=False,
                ),
                ToolParameter(
                    name="data",
                    type="object",
                    description="Form data (dict)",
                    required=False,
                ),
                ToolParameter(
                    name="auth",
                    type="string",
                    description="Authentication: 'Bearer TOKEN' or 'Basic USER:PASS'",
                    required=False,
                ),
                ToolParameter(
                    name="expected_status",
                    type="integer",
                    description="Expected status code (for test operation)",
                    required=False,
                ),
                ToolParameter(
                    name="follow_redirects",
                    type="boolean",
                    description="Follow redirects (default: true)",
                    required=False,
                ),
            ]
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute HTTP operation.

        Args:
            operation: Operation to perform
            **kwargs: Operation-specific parameters

        Returns:
            Tool result with response or error
        """
        operation = kwargs.get("operation")

        if not operation:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: operation",
            )

        try:
            if operation == "request":
                return await self._request(kwargs)
            elif operation == "test":
                return await self._test(kwargs)
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown operation: {operation}",
                )

        except Exception as e:
            return ToolResult(success=False, output="", error=f"HTTP error: {str(e)}")

    async def _request(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Make HTTP request."""
        method = kwargs.get("method", "GET").upper()
        url = kwargs.get("url")

        if not url:
            return ToolResult(success=False, output="", error="Missing required parameter: url")

        # Prepare request
        headers = kwargs.get("headers", {})
        params = kwargs.get("params")
        json_data = kwargs.get("json")
        form_data = kwargs.get("data")
        follow_redirects = kwargs.get("follow_redirects", True)

        # Handle authentication
        auth = kwargs.get("auth")
        if auth:
            if auth.startswith("Bearer "):
                headers["Authorization"] = auth
            elif auth.startswith("Basic "):
                headers["Authorization"] = auth

        try:
            # Make request
            start_time = time.time()

            async with httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=follow_redirects
            ) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json_data,
                    data=form_data,
                )

            duration = time.time() - start_time

            # Parse response
            try:
                response_json = response.json()
            except:
                response_json = None

            # Build result
            result = {
                "status_code": response.status_code,
                "status": response.reason_phrase,
                "headers": dict(response.headers),
                "body": response_json if response_json else response.text[:1000],  # Limit text
                "duration_ms": int(duration * 1000),
                "url": str(response.url),
            }

            return ToolResult(
                success=True,
                output=json.dumps(result, indent=2),
                error="",
            )

        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                output="",
                error=f"Request timed out after {self.timeout} seconds",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Request failed: {str(e)}")

    async def _test(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Test API endpoint with validation."""
        # Make request first
        request_result = await self._request(kwargs)

        if not request_result.success:
            return request_result

        # Parse response
        response = json.loads(request_result.output)

        # Validate
        validations = []
        all_passed = True

        # Check status code
        expected_status = kwargs.get("expected_status")
        if expected_status is not None:
            passed = response["status_code"] == expected_status
            all_passed = all_passed and passed
            validations.append(
                {
                    "test": "Status code",
                    "expected": expected_status,
                    "actual": response["status_code"],
                    "passed": passed,
                }
            )

        # Build test result
        test_result = {
            "url": response["url"],
            "method": kwargs.get("method", "GET").upper(),
            "status_code": response["status_code"],
            "duration_ms": response["duration_ms"],
            "validations": validations,
            "all_passed": all_passed,
        }

        return ToolResult(
            success=all_passed,
            output=json.dumps(test_result, indent=2),
            error="" if all_passed else "Some validations failed",
        )
