"""Per-capability version contracts for step handler boundaries.

Allows individual capabilities (Tools, Safety, Middleware) to version
independently, so verticals can declare "I need Tools v2, Safety v1".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CapabilityContract:
    """Version contract for a specific capability.

    Attributes:
        name: Capability name (e.g., "tools", "safety", "prompt")
        version: Integer version of this capability's contract
        min_sdk_version: Minimum SDK version required (PEP 440 specifier)
    """

    name: str
    version: int = 1
    min_sdk_version: Optional[str] = None

    def is_sdk_compatible(self, sdk_version: str) -> bool:
        """Check if the given SDK version satisfies this contract."""
        if self.min_sdk_version is None:
            return True
        try:
            from packaging.specifiers import SpecifierSet
            spec = SpecifierSet(self.min_sdk_version)
            return sdk_version in spec
        except ImportError:
            # Without packaging library, do simple string comparison
            min_ver = self.min_sdk_version.lstrip(">=<! ")
            return sdk_version >= min_ver
        except Exception:
            return True


@dataclass
class CapabilityCheckResult:
    """Result of checking a capability contract against the SDK."""

    contract_name: str
    compatible: bool
    message: str = ""


class CapabilityContractRegistry:
    """Registry of per-capability version contracts."""

    def __init__(self) -> None:
        self._contracts: Dict[str, CapabilityContract] = {}

    def register(self, contract: CapabilityContract) -> None:
        """Register a capability contract."""
        self._contracts[contract.name] = contract

    def get(self, name: str) -> Optional[CapabilityContract]:
        """Get a registered contract by name."""
        return self._contracts.get(name)

    def check_all(self, sdk_version: str) -> List[CapabilityCheckResult]:
        """Check all registered contracts against the given SDK version."""
        results = []
        for name, contract in self._contracts.items():
            compatible = contract.is_sdk_compatible(sdk_version)
            msg = "" if compatible else (
                f"{name} v{contract.version} requires SDK {contract.min_sdk_version}, "
                f"got {sdk_version}"
            )
            results.append(CapabilityCheckResult(
                contract_name=name,
                compatible=compatible,
                message=msg,
            ))
        return results
