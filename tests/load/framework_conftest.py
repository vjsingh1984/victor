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

"""Configuration for load_test tests."""

import asyncio
import pytest
import httpx


@pytest.fixture(scope="session")
def api_server_available():
    """Check if the API server is running.

    Skips load tests if the API server is not available at localhost:8000.
    """
    try:
        # Try to connect to the API server with a short timeout
        response = asyncio.run(httpx.AsyncClient(timeout=2.0).get("http://localhost:8000/health"))
        if response.status_code == 200:
            return True
    except Exception:
        pass

    # Server not available, skip tests
    pytest.skip("API server not running at localhost:8000. Start with: victor api server")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
