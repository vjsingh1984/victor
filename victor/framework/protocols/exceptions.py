"""Protocol-related exceptions."""

from __future__ import annotations


class IncompatibleVersionError(Exception):
    """Raised when a capability version is incompatible with requirements.

    This exception is raised when invoking a capability that does not meet
    the minimum version requirement specified by the caller.

    Attributes:
        capability_name: Name of the capability
        required_version: Minimum version required
        actual_version: Actual version of the capability
    """

    def __init__(
        self,
        capability_name: str,
        required_version: str,
        actual_version: str,
    ) -> None:
        self.capability_name = capability_name
        self.required_version = required_version
        self.actual_version = actual_version
        super().__init__(
            f"Capability '{capability_name}' version {actual_version} "
            f"is incompatible with required version {required_version}"
        )
