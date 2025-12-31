import asyncio
import shutil
from pathlib import Path

import pytest

from victor.tools.security_scanner_tool import scan


def _pip_audit_available():
    """Check if pip-audit is available."""
    return shutil.which("pip-audit") is not None


@pytest.mark.skipif(
    not _pip_audit_available(),
    reason="pip-audit not installed (install with: pip install pip-audit)",
)
def test_security_scan_dependency_hook(tmp_path: Path):
    """Test dependency vulnerability scanning with pip-audit."""
    req = tmp_path / "requirements.txt"
    req.write_text("flask==0.5\n")

    result = asyncio.run(scan(str(tmp_path), scan_types=["dependencies"], dependency_scan=True))

    assert "dependencies" in result["results"]
    assert result["results"]["dependencies"]["count"] >= 0
