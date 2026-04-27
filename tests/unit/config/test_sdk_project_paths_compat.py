from pathlib import Path

from victor_sdk.verticals.protocols.config import ProjectPathsData


def test_sdk_project_paths_return_path_objects_for_filesystem_consumers() -> None:
    paths = ProjectPathsData(project_root="/tmp/example-project")

    assert isinstance(paths.victor_dir, Path)
    assert isinstance(paths.project_db, Path)
    assert isinstance(paths.project_context_file, Path)
    assert paths.project_db.parent == paths.victor_dir
    assert paths.project_context_file.parent == paths.victor_dir
    assert not hasattr(paths, "conversation_db")
