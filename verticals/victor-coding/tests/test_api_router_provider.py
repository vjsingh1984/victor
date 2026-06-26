"""Tests for victor-coding FastAPI router provider."""

import pytest

# Check if FastAPI is available
try:
    import fastapi  # noqa: F401

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Only import if FastAPI is available
if FASTAPI_AVAILABLE:
    from victor_coding.api.router_provider import (
        create_coding_lsp_router,
        get_fastapi_router_provider,
    )
else:
    # Create dummy imports for pytestmark
    create_coding_lsp_router = None
    get_fastapi_router_provider = None

pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE, reason="FastAPI not installed. Install with: pip install fastapi"
)


def _route_paths(app) -> set[str]:
    return {getattr(route, "path", "") for route in app.routes}


def test_create_coding_lsp_router_registers_expected_routes() -> None:
    router = create_coding_lsp_router()
    paths = _route_paths(router)

    assert "/lsp/completions" in paths
    assert "/lsp/hover" in paths
    assert "/lsp/definition" in paths
    assert "/lsp/references" in paths
    assert "/lsp/diagnostics" in paths


def test_entry_point_provider_returns_router() -> None:
    router = get_fastapi_router_provider(workspace_root="/tmp")
    assert "/lsp/completions" in _route_paths(router)
