"""
Base classes for language processors.

Shared by both indexing extractors and validation validators.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..registry import LanguageCapabilityRegistry
    from ..types import UnifiedLanguageCapability


class BaseLanguageProcessor(ABC):
    """
    Base class for language processors.

    Shared by both indexing extractors and validation validators.
    Uses the unified capability registry.
    """

    def __init__(self, registry: Optional["LanguageCapabilityRegistry"] = None) -> None:
        # Lazy import to avoid circular dependency
        if registry is None:
            from ..registry import LanguageCapabilityRegistry

            registry = LanguageCapabilityRegistry.instance()
        self._registry = registry

    @abstractmethod
    def process(
        self,
        code: str,
        file_path: Path,
        language: Optional[str] = None,
    ) -> Any:
        """
        Process code and return results.

        Args:
            code: Source code to process
            file_path: Path to the source file
            language: Optional language override

        Returns:
            Processing results (type varies by subclass)
        """
        pass

    def _get_capability(
        self, file_path: Path, language: Optional[str] = None
    ) -> Optional["UnifiedLanguageCapability"]:
        """Get capability for file/language."""
        if language:
            return self._registry.get(language)
        return self._registry.get_for_file(file_path)

    def _detect_language(self, file_path: Path) -> Optional[str]:
        """Detect language from file."""
        cap = self._registry.get_for_file(file_path)
        return cap.name if cap else None

    @property
    def registry(self) -> "LanguageCapabilityRegistry":
        """Get the capability registry."""
        return self._registry
