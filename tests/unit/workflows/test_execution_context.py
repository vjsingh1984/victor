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

"""Tests for victor.workflows.execution_context module."""

import pytest

from victor.workflows.execution_context import ExecutionContext


class TestExecutionContext:
    """Test ExecutionContext class."""

    def test_init_with_all_parameters(self):
        """Test initialization with all parameters."""
        orchestrator = mock_orchestrator()
        settings = {"temperature": 0.7}
        services = {"service1": "value1"}

        context = ExecutionContext(
            orchestrator=orchestrator,
            settings=settings,
            services=services,
        )

        assert context.orchestrator == orchestrator
        assert context.settings == settings
        assert context.services == services

    def test_init_with_partial_parameters(self):
        """Test initialization with partial parameters."""
        context = ExecutionContext(orchestrator=mock_orchestrator())

        assert context.orchestrator is not None
        assert context.settings is None
        assert context.services is None

    def test_init_with_no_parameters(self):
        """Test initialization with no parameters."""
        context = ExecutionContext()

        assert context.orchestrator is None
        assert context.settings is None
        assert context.services is None

    def test_orchestrator_property(self):
        """Test orchestrator property."""
        orchestrator = mock_orchestrator()
        context = ExecutionContext(orchestrator=orchestrator)

        assert context.orchestrator == orchestrator

    def test_settings_property(self):
        """Test settings property."""
        settings = {"temperature": 0.7, "max_tokens": 1000}
        context = ExecutionContext(settings=settings)

        assert context.settings == settings

    def test_services_property(self):
        """Test services property."""
        services = {"service1": "value1", "service2": "value2"}
        context = ExecutionContext(services=services)

        assert context.services == services

    def test_property_immutability(self):
        """Test that properties return the same values."""
        orchestrator = mock_orchestrator()
        settings = {"temperature": 0.7}
        services = {"service1": "value1"}

        context = ExecutionContext(
            orchestrator=orchestrator,
            settings=settings,
            services=services,
        )

        # Access properties multiple times
        assert context.orchestrator is context.orchestrator
        assert context.settings is context.settings
        assert context.services is context.services

    def test_with_complex_settings(self):
        """Test with complex settings object."""
        settings = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "model": "gpt-4",
            "nested": {
                "key1": "value1",
                "key2": "value2",
            },
        }

        context = ExecutionContext(settings=settings)

        assert context.settings == settings
        assert context.settings["nested"]["key1"] == "value1"

    def test_with_complex_services(self):
        """Test with complex services object."""

        class Service1:
            def method1(self):
                return "service1"

        class Service2:
            def method2(self):
                return "service2"

        services = {
            "service1": Service1(),
            "service2": Service2(),
        }

        context = ExecutionContext(services=services)

        assert context.services == services
        assert context.services["service1"].method1() == "service1"
        assert context.services["service2"].method2() == "service2"


# Helper function to create a mock orchestrator
def mock_orchestrator():
    """Create a mock orchestrator for testing."""

    class MockOrchestrator:
        def __init__(self):
            self.name = "mock_orchestrator"

    return MockOrchestrator()
