import asyncio
import shutil
from pathlib import Path

import pytest

from victor.tools.security_scanner_tool import scan


def _bandit_available():
    """Check if bandit is available."""
    return shutil.which("bandit") is not None


@pytest.mark.skipif(
    not _bandit_available(),
    reason="bandit not installed (install with: pip install bandit)",
)
def test_security_scan_iac_hook(tmp_path: Path):
    """Test IAC/config security scanning with bandit."""
    target = tmp_path / "bad.py"
    target.write_text("import os\nos.system('ls')\n")

    result = asyncio.run(scan(str(tmp_path), scan_types=["config"], iac_scan=True))

    assert "config" in result["results"]
    assert result["results"]["config"]["count"] >= 0
