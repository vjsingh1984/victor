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
from typing import Any, Dict, Optional

import httpx

from victor.tools.decorators import tool


@tool
async def http_request(
    method: str,
    url: str,
    headers: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    auth: Optional[str] = None,
    follow_redirects: bool = True,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Make an HTTP request to a URL.

    Supports all HTTP methods with headers, authentication, query parameters,
    JSON body, form data, and performance metrics.

    Args:
        method: HTTP method (GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS).
        url: Request URL.
        headers: Request headers (optional).
        params: Query parameters (optional).
        json: JSON body as dictionary (optional).
        data: Form data as dictionary (optional).
        auth: Authentication - 'Bearer TOKEN' or 'Basic USER:PASS' (optional).
        follow_redirects: Follow redirects (default: true).
        timeout: Request timeout in seconds (default: 30).

    Returns:
        Dictionary containing:
        - success: Whether request succeeded
        - status_code: HTTP status code
        - status: Status reason phrase
        - headers: Response headers
        - body: Parsed JSON or text response
        - duration_ms: Request duration in milliseconds
        - url: Final URL (after redirects)
        - error: Error message if failed
    """
    if not url:
        return {
            "success": False,
            "error": "Missing required parameter: url"
        }

    method = method.upper()
    request_headers = headers or {}

    # Handle authentication
    if auth:
        if auth.startswith("Bearer ") or auth.startswith("Basic "):
            request_headers["Authorization"] = auth

    try:
        # Make request
        start_time = time.time()

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=follow_redirects) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=request_headers,
                params=params,
                json=json,
                data=data,
            )

        duration = time.time() - start_time

        # Parse response
        try:
            response_json = response.json()
        except (ValueError, TypeError, AttributeError):
            response_json = None

        # Build result
        return {
            "success": True,
            "status_code": response.status_code,
            "status": response.reason_phrase,
            "headers": dict(response.headers),
            "body": response_json if response_json else response.text[:1000],  # Limit text
            "duration_ms": int(duration * 1000),
            "url": str(response.url),
        }

    except httpx.TimeoutException:
        return {
            "success": False,
            "error": f"Request timed out after {timeout} seconds"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}"
        }


@tool
async def http_test(
    method: str,
    url: str,
    expected_status: Optional[int] = None,
    headers: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    auth: Optional[str] = None,
    follow_redirects: bool = True,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Test an API endpoint with validation.

    Makes an HTTP request and validates the response against expected values.
    Currently validates status code, can be extended for more validations.

    Args:
        method: HTTP method (GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS).
        url: Request URL.
        expected_status: Expected HTTP status code (optional).
        headers: Request headers (optional).
        params: Query parameters (optional).
        json: JSON body as dictionary (optional).
        data: Form data as dictionary (optional).
        auth: Authentication - 'Bearer TOKEN' or 'Basic USER:PASS' (optional).
        follow_redirects: Follow redirects (default: true).
        timeout: Request timeout in seconds (default: 30).

    Returns:
        Dictionary containing:
        - success: Whether all validations passed
        - url: Final URL tested
        - method: HTTP method used
        - status_code: HTTP status code received
        - duration_ms: Request duration in milliseconds
        - validations: List of validation results
        - all_passed: Whether all validations passed
        - error: Error message if request failed
    """
    # Make request first using http_request
    # We need to call the underlying logic directly to avoid double decoration
    if not url:
        return {
            "success": False,
            "error": "Missing required parameter: url"
        }

    method_upper = method.upper()
    request_headers = headers or {}

    # Handle authentication
    if auth:
        if auth.startswith("Bearer ") or auth.startswith("Basic "):
            request_headers["Authorization"] = auth

    try:
        # Make request
        start_time = time.time()

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=follow_redirects) as client:
            response = await client.request(
                method=method_upper,
                url=url,
                headers=request_headers,
                params=params,
                json=json,
                data=data,
            )

        duration = time.time() - start_time

        # Validate
        validations = []
        all_passed = True

        # Check status code
        if expected_status is not None:
            passed = response.status_code == expected_status
            all_passed = all_passed and passed
            validations.append({
                "test": "Status code",
                "expected": expected_status,
                "actual": response.status_code,
                "passed": passed,
            })

        # Build test result
        return {
            "success": all_passed,
            "url": str(response.url),
            "method": method_upper,
            "status_code": response.status_code,
            "duration_ms": int(duration * 1000),
            "validations": validations,
            "all_passed": all_passed,
            "error": "" if all_passed else "Some validations failed"
        }

    except httpx.TimeoutException:
        return {
            "success": False,
            "error": f"Request timed out after {timeout} seconds"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}"
        }
