import asyncio
from pathlib import Path

import pytest

from victor.tools.security_scanner_tool import scan


@pytest.mark.skipif(True, reason="pip-audit may not be available in CI")
def test_security_scan_dependency_hook(tmp_path: Path):
    req = tmp_path / "requirements.txt"
    req.write_text("flask==0.5\n")

    result = asyncio.run(scan(str(tmp_path), scan_types=["dependencies"], dependency_scan=True))

    assert "dependencies" in result["results"]
    assert result["results"]["dependencies"]["count"] >= 0
