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

"""Load testing and scalability test suite for Victor AI.

This module provides comprehensive load testing infrastructure including:
- Locust-based HTTP load testing
- Pytest-based scalability tests
- Performance benchmarking
- Memory leak detection
- Stress testing

Example Usage:
    # Run Locust web interface
    make load-test

    # Run quick pytest test
    pytest tests/load_test/test_scalability.py::TestConcurrentRequests::test_10_concurrent_requests -v

    # Run headless load test
    make load-test-headless
"""

__all__ = [
    "load_test_framework",
    "test_scalability",
]
