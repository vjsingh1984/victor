#!/usr/bin/env python3
"""Test code_review tool fixes."""

import os
import pytest
import asyncio
from victor.tools.code_review_tool import code_review

# Get project root for absolute paths
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


@pytest.mark.asyncio
async def test_code_review_with_json_string_aspects():
    """Test code_review handles aspects as JSON string."""
    # This simulates how LLM might pass the parameter
    result = await code_review(
        path=os.path.join(PROJECT_ROOT, "victor/tools"),
        aspects='["security", "complexity"]',  # JSON string instead of list
        max_issues=5,
    )

    assert result["success"] is True, f"Expected success, got: {result.get('error', 'No error')}"
    assert "security" in result["aspects_checked"]
    assert "complexity" in result["aspects_checked"]


@pytest.mark.asyncio
async def test_code_review_with_list_aspects():
    """Test code_review handles aspects as proper list."""
    result = await code_review(
        path=os.path.join(PROJECT_ROOT, "victor/tools"),
        aspects=["best_practices", "documentation"],  # Proper list
        max_issues=5,
    )

    assert result["success"] is True
    assert "best_practices" in result["aspects_checked"]
    assert "documentation" in result["aspects_checked"]


@pytest.mark.asyncio
async def test_code_review_with_single_aspect_string():
    """Test code_review handles single aspect as string."""
    result = await code_review(
        path=os.path.join(PROJECT_ROOT, "victor/tools"),
        aspects="security",
        max_issues=5,  # Single string
    )

    assert result["success"] is True
    assert "security" in result["aspects_checked"]


@pytest.mark.asyncio
async def test_code_review_with_invalid_aspect():
    """Test code_review rejects invalid aspects."""
    result = await code_review(
        path=os.path.join(PROJECT_ROOT, "victor/tools"),
        aspects=["invalid_aspect", "security"],
        max_issues=5,
    )

    assert result["success"] is False
    assert "Invalid aspect" in result["error"]


@pytest.mark.asyncio
async def test_code_review_all_aspects():
    """Test code_review with 'all' aspects."""
    result = await code_review(
        path=os.path.join(PROJECT_ROOT, "victor/tools/base.py"),
        aspects=["all"],
        max_issues=10,
    )

    assert result["success"] is True
    assert set(result["aspects_checked"]) == {
        "security",
        "complexity",
        "best_practices",
        "documentation",
    }


@pytest.mark.asyncio
async def test_code_review_report_format():
    """Test that report uses correct field names."""
    result = await code_review(
        path=os.path.join(PROJECT_ROOT, "victor/tools/base.py"),
        aspects=["security", "best_practices"],
        max_issues=5,
    )

    assert result["success"] is True
    assert "formatted_report" in result
    report = result["formatted_report"]

    # Report should not have empty placeholder fields
    assert "None" not in report or report.count("None") < 3  # Allow some None in paths

    # Should contain actual issue information
    if result["total_issues"] > 0:
        assert any(
            keyword in report.lower() for keyword in ["security", "best", "practice", "issue"]
        )


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_code_review_with_json_string_aspects())
    asyncio.run(test_code_review_with_list_aspects())
    asyncio.run(test_code_review_with_single_aspect_string())
    asyncio.run(test_code_review_with_invalid_aspect())
    asyncio.run(test_code_review_all_aspects())
    asyncio.run(test_code_review_report_format())
    print("✓ All tests passed!")
