import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from victor.tools.filesystem import overview

@pytest.mark.asyncio
async def test_overview_with_directory_summaries():
    # Setup mocks
    mock_db = MagicMock()
    mock_db.table_exists.return_value = True
    # Mock some symbols in the DB
    mock_db.query.return_value = [
        {"type": "class", "name": "AgentOrchestrator"},
        {"type": "function", "name": "execute_task"}
    ]
    
    # We need to mock Path objects carefully as overview uses them extensively
    with patch("victor.core.database.get_project_database", return_value=mock_db), \
         patch("victor.tools.filesystem.Path.iterdir") as mock_iterdir, \
         patch("victor.tools.filesystem.Path.exists", return_value=True), \
         patch("victor.tools.filesystem.Path.is_dir", return_value=True), \
         patch("victor.tools.filesystem.Path.stat") as mock_stat:
        
        # Mock directory structure: root/dir1
        mock_dir1 = MagicMock(spec=Path)
        mock_dir1.name = "dir1"
        mock_dir1.is_dir.return_value = True
        mock_dir1.relative_to.return_value = Path("dir1")
        mock_dir1.suffix = ""
        
        mock_iterdir.return_value = [mock_dir1]
        mock_stat.return_value.st_size = 100
        
        # Call overview
        result = await overview(path=".", max_depth=1)
        
        assert result["success"] is True
        assert len(result["directories"]) >= 1
        
        # Find our mocked directory
        dir_entry = next((d for d in result["directories"] if d["path"] == "dir1"), None)
        assert dir_entry is not None
        assert "summary" in dir_entry
        assert "AgentOrchestrator" in dir_entry["summary"]
        assert "execute_task" in dir_entry["summary"]
