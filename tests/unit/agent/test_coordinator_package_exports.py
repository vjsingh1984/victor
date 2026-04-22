"""Tests for deprecated package-root coordinator exports."""

import pytest


def test_session_package_exports_warn():
    """Package-root session coordinator exports should be compatibility-only."""
    from victor.agent.coordinators.session_coordinator import (
        SessionCoordinator,
        create_session_coordinator,
    )

    with pytest.warns(
        DeprecationWarning,
        match="victor.agent.coordinators.SessionCoordinator is deprecated compatibility surface",
    ):
        from victor.agent.coordinators import SessionCoordinator as package_session_coordinator

    with pytest.warns(
        DeprecationWarning,
        match=(
            "victor.agent.coordinators.create_session_coordinator is deprecated "
            "compatibility surface"
        ),
    ):
        from victor.agent.coordinators import (
            create_session_coordinator as package_create_session_coordinator,
        )

    assert package_session_coordinator is SessionCoordinator
    assert package_create_session_coordinator is create_session_coordinator
