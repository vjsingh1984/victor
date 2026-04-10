import asyncio
from pathlib import Path

from victor.tools.cicd_tool import cicd


def test_cicd_generate_and_validate(tmp_path: Path):
    out_file = tmp_path / "workflow.yml"
    result = asyncio.run(cicd(operation="generate", workflow="python-test", output=str(out_file)))
    assert result["success"]
    assert out_file.exists()

    validate = asyncio.run(cicd(operation="validate", file=str(out_file)))
    assert validate["success"]
    assert not validate["issues"]


def test_cicd_validate_errors(tmp_path: Path):
    bad = tmp_path / "bad.yml"
    bad.write_text("name: test\njobs: {}\n")
    validate = asyncio.run(cicd(operation="validate", file=str(bad)))
    assert not validate["success"]
    assert "Missing 'on' field" in "\n".join(validate["issues"])
