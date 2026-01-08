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

"""Base capability provider for verticals.

This module provides the abstract base class for vertical capability providers,
enabling consistent capability registration and discovery across all verticals.

Design Pattern: Generic Provider
- Type-safe capability registration via generics
- Metadata-driven capability discovery
- Consistent interface across all verticals

Example:
    class CodingCapabilityProvider(BaseCapabilityProvider[CodingCapability]):
        def get_capabilities(self) -> Dict[str, CodingCapability]:
            return {"code_review": self._code_review_capability}

        def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
            return {
                "code_review": CapabilityMetadata(
                    name="code_review",
                    description="Review code for quality and issues",
                    version="1.0",
                    tags=["review", "quality"]
                )
            }
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")


@dataclass
class CapabilityMetadata:
    """Metadata for a registered capability.

    Attributes:
        name: Unique identifier for the capability.
        description: Human-readable description of what the capability does.
        version: Semantic version string (default: "1.0").
        dependencies: List of capability names this capability depends on.
        tags: List of tags for categorization and discovery.

    Example:
        metadata = CapabilityMetadata(
            name="code_review",
            description="Automated code review capability",
            version="2.0",
            dependencies=["ast_analysis"],
            tags=["review", "quality", "static-analysis"]
        )
    """

    name: str
    description: str
    version: str = "1.0"
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class BaseCapabilityProvider(ABC, Generic[T]):
    """Abstract base for vertical capability providers.

    This class provides a consistent interface for registering and discovering
    capabilities within a vertical. Each vertical should implement a concrete
    provider that returns its specific capabilities.

    Type Parameters:
        T: The type of capability objects this provider manages.

    Example:
        class CodingCapabilityProvider(BaseCapabilityProvider[CodingCapability]):
            def __init__(self):
                self._code_review = CodeReviewCapability()
                self._test_gen = TestGenerationCapability()

            def get_capabilities(self) -> Dict[str, CodingCapability]:
                return {
                    "code_review": self._code_review,
                    "test_generation": self._test_gen
                }

            def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
                return {
                    "code_review": CapabilityMetadata(
                        name="code_review",
                        description="Review code for quality issues",
                        version="1.0",
                        tags=["review"]
                    ),
                    "test_generation": CapabilityMetadata(
                        name="test_generation",
                        description="Generate unit tests",
                        version="1.0",
                        dependencies=["code_review"],
                        tags=["testing"]
                    )
                }
    """

    @abstractmethod
    def get_capabilities(self) -> Dict[str, T]:
        """Return all registered capabilities.

        Returns:
            Dictionary mapping capability names to capability objects.
        """
        pass

    @abstractmethod
    def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
        """Return metadata for all registered capabilities.

        Returns:
            Dictionary mapping capability names to their metadata.
        """
        pass

    def get_capability(self, name: str) -> Optional[T]:
        """Retrieve a specific capability by name.

        Args:
            name: The name of the capability to retrieve.

        Returns:
            The capability object if found, None otherwise.
        """
        return self.get_capabilities().get(name)

    def list_capabilities(self) -> List[str]:
        """List all available capability names.

        Returns:
            List of capability names.
        """
        return list(self.get_capabilities().keys())

    def has_capability(self, name: str) -> bool:
        """Check if a capability is available.

        Args:
            name: The name of the capability to check.

        Returns:
            True if the capability exists, False otherwise.
        """
        return name in self.get_capabilities()
