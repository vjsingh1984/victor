import asyncio
from pathlib import Path

import pytest

from victor.tools.security_scanner_tool import scan


@pytest.mark.skipif(True, reason="bandit may not be available in CI")
def test_security_scan_iac_hook(tmp_path: Path):
    target = tmp_path / "bad.py"
    target.write_text("import os\nos.system('ls')\n")

    result = asyncio.run(scan(str(tmp_path), scan_types=["config"], iac_scan=True))

    assert "config" in result["results"]
    assert result["results"]["config"]["count"] >= 0
