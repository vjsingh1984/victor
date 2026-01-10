# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for vertical registry commands."""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from typer.testing import CliRunner

from victor.core.verticals.registry_manager import (
    InstalledVertical,
    PackageSpec,
    PackageSourceType,
    VerticalRegistryManager,
)
from victor.core.verticals.package_schema import (
    VerticalPackageMetadata,
    VerticalClassSpec,
    AuthorInfo,
)
from victor.ui.commands.vertical import vertical_app

runner = CliRunner()


class TestPackageSpec:
    """Tests for PackageSpec class."""

    def test_parse_simple_name(self):
        """Test parsing simple package name."""
        spec = PackageSpec.parse("victor-security")
        assert spec.name == "victor-security"
        assert spec.version is None
        assert spec.source == PackageSourceType.PYPI
        assert spec.url is None
        assert spec.extras == []

    def test_parse_with_version(self):
        """Test parsing package with version constraint."""
        spec = PackageSpec.parse("victor-security>=1.0.0")
        assert spec.name == "victor-security"
        assert spec.version == ">=1.0.0"
        assert spec.source == PackageSourceType.PYPI

    def test_parse_git_url(self):
        """Test parsing git URL."""
        url = "https://github.com/user/victor-security.git"
        spec = PackageSpec.parse(f"git+{url}")
        assert spec.name == "victor-security"
        assert spec.source == PackageSourceType.GIT
        assert spec.url == url

    def test_parse_local_path(self):
        """Test parsing local path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "victor-security"
            path.mkdir()

            spec = PackageSpec.parse(str(path))
            assert spec.name == "victor-security"
            assert spec.source == PackageSourceType.LOCAL
            # Resolve paths for comparison (macOS adds /private prefix)
            assert Path(spec.url).resolve() == path.resolve()

    def test_parse_with_extras(self):
        """Test parsing package with extras."""
        spec = PackageSpec.parse("victor-security[full,dev]")
        assert spec.name == "victor-security"
        assert spec.extras == ["full", "dev"]

    def test_to_pip_string_pypi(self):
        """Test converting PyPI spec to pip string."""
        spec = PackageSpec(name="victor-security", version=">=1.0.0")
        assert spec.to_pip_string() == "victor-security>=1.0.0"

    def test_to_pip_string_git(self):
        """Test converting git spec to pip string."""
        spec = PackageSpec(
            name="victor-security",
            source=PackageSourceType.GIT,
            url="https://github.com/user/victor-security.git",
        )
        assert spec.to_pip_string() == "git+https://github.com/user/victor-security.git"

    def test_to_pip_string_local(self):
        """Test converting local spec to pip string."""
        spec = PackageSpec(
            name="victor-security",
            source=PackageSourceType.LOCAL,
            url="/path/to/package",
        )
        assert spec.to_pip_string() == "/path/to/package"


class TestVerticalRegistryManager:
    """Tests for VerticalRegistryManager class."""

    def test_init_default(self):
        """Test initialization with defaults."""
        manager = VerticalRegistryManager()
        assert manager.dry_run is False
        assert manager.cache_dir.exists()

    def test_init_dry_run(self):
        """Test initialization with dry_run."""
        manager = VerticalRegistryManager(dry_run=True)
        assert manager.dry_run is True

    def test_list_builtin_verticals(self):
        """Test listing built-in verticals."""
        # Create a temporary directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            victor_dir = Path(tmpdir)

            # Create fake vertical directories
            (victor_dir / "coding").mkdir()
            (victor_dir / "devops").mkdir()

            manager = VerticalRegistryManager()

            with patch.object(manager, "BUILTIN_VERTICALS", ["coding", "devops"]):
                verticals = manager._list_builtin_verticals(victor_dir)

        assert len(verticals) == 2
        assert all(v.is_builtin for v in verticals)

    def test_list_installed_verticals(self):
        """Test listing installed verticals."""
        manager = VerticalRegistryManager()

        # Mock importlib.metadata
        with patch("importlib.metadata.distributions") as mock_dists:
            mock_dist = Mock()
            mock_dist.version = "1.0.0"
            mock_dist.locate_file.return_value = Path("/fake/location")

            mock_ep = Mock()
            mock_ep.name = "security"
            mock_dist.entry_points.select.return_value = [mock_ep]
            mock_dists.return_value = [mock_dist]

            verticals = manager._list_installed_verticals()

        assert len(verticals) == 1
        assert verticals[0].name == "security"
        assert verticals[0].version == "1.0.0"
        assert not verticals[0].is_builtin

    def test_search_by_name(self):
        """Test searching verticals by name."""
        manager = VerticalRegistryManager()

        # Mock list_verticals
        verticals = [
            InstalledVertical(
                name="security",
                version="1.0.0",
                location=Path("/fake"),
                metadata=VerticalPackageMetadata(
                    name="security",
                    version="1.0.0",
                    description="Security analysis",
                    authors=[AuthorInfo(name="Test")],
                    license="Apache-2.0",
                    requires_victor=">=0.5.0",
                    class_spec=VerticalClassSpec(
                        module="test",
                        class_name="Test",
                    ),
                ),
            ),
            InstalledVertical(
                name="coding",
                version="builtin",
                location=Path("/fake"),
                metadata=None,
            ),
        ]

        with patch.object(manager, "list_verticals", return_value=verticals):
            results = manager.search("security")

        assert len(results) == 1
        assert results[0].name == "security"

    def test_search_by_description(self):
        """Test searching verticals by description."""
        manager = VerticalRegistryManager()

        verticals = [
            InstalledVertical(
                name="security",
                version="1.0.0",
                location=Path("/fake"),
                metadata=VerticalPackageMetadata(
                    name="security",
                    version="1.0.0",
                    description="Security analysis and vulnerability scanning",
                    authors=[AuthorInfo(name="Test")],
                    license="Apache-2.0",
                    requires_victor=">=0.5.0",
                    class_spec=VerticalClassSpec(
                        module="test",
                        class_name="Test",
                    ),
                ),
            ),
        ]

        with patch.object(manager, "list_verticals", return_value=verticals):
            results = manager.search("vulnerability")

        assert len(results) == 1
        assert results[0].name == "security"

    def test_search_by_tags(self):
        """Test searching verticals by tags."""
        manager = VerticalRegistryManager()

        verticals = [
            InstalledVertical(
                name="security",
                version="1.0.0",
                location=Path("/fake"),
                metadata=VerticalPackageMetadata(
                    name="security",
                    version="1.0.0",
                    description="Security analysis",
                    authors=[AuthorInfo(name="Test")],
                    license="Apache-2.0",
                    requires_victor=">=0.5.0",
                    class_spec=VerticalClassSpec(
                        module="test",
                        class_name="Test",
                    ),
                    tags=["security", "analysis", "scanning"],
                ),
            ),
        ]

        with patch.object(manager, "list_verticals", return_value=verticals):
            results = manager.search("scanning")

        assert len(results) == 1
        assert results[0].name == "security"

    def test_get_info_found(self):
        """Test getting info for existing vertical."""
        manager = VerticalRegistryManager()

        verticals = [
            InstalledVertical(
                name="security",
                version="1.0.0",
                location=Path("/fake"),
                metadata=None,
            ),
        ]

        with patch.object(manager, "list_verticals", return_value=verticals):
            result = manager.get_info("security")

        assert result is not None
        assert result.name == "security"

    def test_get_info_not_found(self):
        """Test getting info for non-existent vertical."""
        manager = VerticalRegistryManager()

        with patch.object(manager, "list_verticals", return_value=[]):
            result = manager.get_info("nonexistent")

        assert result is None

    def test_install_dry_run(self):
        """Test installation in dry-run mode."""
        manager = VerticalRegistryManager(dry_run=True)
        spec = PackageSpec(name="victor-security")

        success, message = manager.install(spec)

        assert success is True
        assert "Would install" in message

    @patch("subprocess.run")
    def test_install_success(self, mock_run):
        """Test successful installation."""
        manager = VerticalRegistryManager(dry_run=False)
        spec = PackageSpec(name="victor-security")

        mock_run.return_value = Mock(
            returncode=0,
            stdout="Installing...",
            stderr="",
        )

        success, message = manager.install(spec)

        assert success is True
        assert "Successfully installed" in message
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_install_failure(self, mock_run):
        """Test failed installation."""
        manager = VerticalRegistryManager(dry_run=False)
        spec = PackageSpec(name="victor-security")

        mock_run.side_effect = subprocess.CalledProcessError(
            1, "pip", stderr="Installation failed"
        )

        success, message = manager.install(spec)

        assert success is False
        assert "Installation failed" in message

    def test_uninstall_builtin(self):
        """Test uninstalling built-in vertical (should fail)."""
        manager = VerticalRegistryManager()

        success, message = manager.uninstall("coding")

        assert success is False
        assert "Cannot uninstall built-in" in message

    def test_uninstall_dry_run(self):
        """Test uninstallation in dry-run mode."""
        manager = VerticalRegistryManager(dry_run=True)

        success, message = manager.uninstall("victor-security")

        assert success is True
        assert "Would uninstall" in message

    @patch("subprocess.run")
    def test_uninstall_success(self, mock_run):
        """Test successful uninstallation."""
        manager = VerticalRegistryManager(dry_run=False)

        mock_run.return_value = Mock(
            returncode=0,
            stdout="Uninstalling...",
            stderr="",
        )

        success, message = manager.uninstall("victor-security")

        assert success is True
        assert "Successfully uninstalled" in message

    def test_validate_package_builtin_name(self):
        """Test validation fails for built-in name."""
        manager = VerticalRegistryManager()
        spec = PackageSpec(name="coding")

        errors = manager._validate_package(spec)

        assert len(errors) > 0
        assert "conflicts with built-in" in errors[0]

    def test_validate_package_invalid_name(self):
        """Test validation fails for invalid package name."""
        manager = VerticalRegistryManager()
        spec = PackageSpec(name="invalid-package")

        # No errors for non-builtin names
        errors = manager._validate_package(spec)
        assert len(errors) == 0

    def test_clear_cache(self):
        """Test clearing metadata cache."""
        manager = VerticalRegistryManager()

        # Create a fake cache file
        cache_file = manager.cache_dir / "available.json"
        cache_file.write_text(json.dumps({"test": "data"}))

        # Clear cache
        manager.clear_cache()

        # Cache should be gone
        assert not cache_file.exists()


class TestVerticalCommands:
    """Tests for vertical CLI commands."""

    def test_install_command_missing_package(self, tmp_path):
        """Test install command requires package argument."""
        result = runner.invoke(vertical_app, ["install"])
        assert result.exit_code != 0
        # Typer puts error messages in result.output, not result.stdout
        assert "Missing argument" in result.output or "requires an argument" in result.output

    @patch("victor.core.verticals.registry_manager.VerticalRegistryManager.install")
    def test_install_command_success(self, mock_install):
        """Test install command succeeds."""
        mock_install.return_value = (True, "Successfully installed victor-security")

        result = runner.invoke(vertical_app, ["install", "victor-security"])

        assert result.exit_code == 0
        assert "Successfully installed" in result.stdout

    @patch("victor.core.verticals.registry_manager.VerticalRegistryManager.install")
    def test_install_command_failure(self, mock_install):
        """Test install command handles failure."""
        mock_install.return_value = (False, "Installation failed")

        result = runner.invoke(vertical_app, ["install", "victor-security"])

        assert result.exit_code == 1
        assert "Installation failed" in result.stdout

    @patch("victor.core.verticals.registry_manager.VerticalRegistryManager.uninstall")
    def test_uninstall_command_builtin(self, mock_uninstall):
        """Test uninstall command fails for built-in."""
        mock_uninstall.return_value = (False, "Cannot uninstall built-in vertical: coding")

        result = runner.invoke(vertical_app, ["uninstall", "coding"])

        assert result.exit_code == 1
        assert "Cannot uninstall built-in" in result.stdout

    @patch("victor.core.verticals.registry_manager.VerticalRegistryManager.list_verticals")
    def test_list_command(self, mock_list):
        """Test list command."""
        mock_list.return_value = [
            InstalledVertical(
                name="coding",
                version="builtin",
                location=Path("/fake"),
                is_builtin=True,
                metadata=None,
            ),
            InstalledVertical(
                name="security",
                version="1.0.0",
                location=Path("/fake"),
                is_builtin=False,
                metadata=None,
            ),
        ]

        result = runner.invoke(vertical_app, ["list"])

        assert result.exit_code == 0
        assert "coding" in result.stdout
        assert "security" in result.stdout

    def test_list_command_invalid_source(self):
        """Test list command with invalid source."""
        result = runner.invoke(vertical_app, ["list", "--source", "invalid"])

        assert result.exit_code == 1
        assert "Invalid source" in result.stdout

    @patch("victor.core.verticals.registry_manager.VerticalRegistryManager.search")
    def test_search_command(self, mock_search):
        """Test search command."""
        mock_search.return_value = [
            InstalledVertical(
                name="security",
                version="1.0.0",
                location=Path("/fake"),
                metadata=VerticalPackageMetadata(
                    name="security",
                    version="1.0.0",
                    description="Security analysis",
                    authors=[AuthorInfo(name="Test")],
                    license="Apache-2.0",
                    requires_victor=">=0.5.0",
                    class_spec=VerticalClassSpec(
                        module="test",
                        class_name="Test",
                    ),
                ),
            ),
        ]

        result = runner.invoke(vertical_app, ["search", "security"])

        assert result.exit_code == 0
        assert "security" in result.stdout

    @patch("victor.core.verticals.registry_manager.VerticalRegistryManager.get_info")
    def test_info_command_found(self, mock_info):
        """Test info command for existing vertical."""
        mock_info.return_value = InstalledVertical(
            name="security",
            version="1.0.0",
            location=Path("/fake"),
            metadata=VerticalPackageMetadata(
                name="security",
                version="1.0.0",
                description="Security analysis",
                authors=[AuthorInfo(name="Test")],
                license="Apache-2.0",
                requires_victor=">=0.5.0",
                class_spec=VerticalClassSpec(
                    module="test",
                    class_name="Test",
                ),
            ),
        )

        result = runner.invoke(vertical_app, ["info", "security"])

        assert result.exit_code == 0
        assert "security" in result.stdout
        assert "1.0.0" in result.stdout

    @patch("victor.core.verticals.registry_manager.VerticalRegistryManager.get_info")
    def test_info_command_not_found(self, mock_info):
        """Test info command for non-existent vertical."""
        mock_info.return_value = None

        result = runner.invoke(vertical_app, ["info", "nonexistent"])

        assert result.exit_code == 1
        assert "not found" in result.stdout

    @patch("victor.ui.commands.scaffold.new_vertical")
    def test_create_command(self, mock_create):
        """Test create command (alias to scaffold)."""
        result = runner.invoke(
            vertical_app,
            ["create", "test-vertical", "--description", "Test vertical"],
        )

        assert result.exit_code == 0
        mock_create.assert_called_once()


class TestVerticalFiltering:
    """Tests for vertical filtering functionality."""

    @patch("victor.core.verticals.registry_manager.VerticalRegistryManager.list_verticals")
    def test_list_with_category_filter(self, mock_list):
        """Test list command with category filter."""
        mock_list.return_value = [
            InstalledVertical(
                name="security",
                version="1.0.0",
                location=Path("/fake"),
                is_builtin=False,
                metadata=VerticalPackageMetadata(
                    name="security",
                    version="1.0.0",
                    description="Security analysis",
                    authors=[AuthorInfo(name="Test")],
                    license="Apache-2.0",
                    requires_victor=">=0.5.0",
                    category="security",
                    class_spec=VerticalClassSpec(
                        module="test",
                        class_name="Test",
                    ),
                ),
            ),
            InstalledVertical(
                name="coding",
                version="0.5.1",
                location=Path("/fake"),
                is_builtin=True,
                metadata=VerticalPackageMetadata(
                    name="coding",
                    version="0.5.1",
                    description="Coding assistant",
                    authors=[AuthorInfo(name="Test")],
                    license="Apache-2.0",
                    requires_victor=">=0.5.0",
                    category="development",
                    class_spec=VerticalClassSpec(
                        module="test",
                        class_name="Test",
                    ),
                ),
            ),
        ]

        result = runner.invoke(vertical_app, ["list", "--category", "security"])

        assert result.exit_code == 0
        assert "security" in result.stdout
        # Only security should be shown when filtering by security category

    @patch("victor.core.verticals.registry_manager.VerticalRegistryManager.list_verticals")
    def test_list_with_tags_filter(self, mock_list):
        """Test list command with tags filter."""
        mock_list.return_value = [
            InstalledVertical(
                name="security",
                version="1.0.0",
                location=Path("/fake"),
                is_builtin=False,
                metadata=VerticalPackageMetadata(
                    name="security",
                    version="1.0.0",
                    description="Security analysis",
                    authors=[AuthorInfo(name="Test")],
                    license="Apache-2.0",
                    requires_victor=">=0.5.0",
                    tags=["security", "scanning", "sast"],
                    class_spec=VerticalClassSpec(
                        module="test",
                        class_name="Test",
                    ),
                ),
            ),
        ]

        result = runner.invoke(vertical_app, ["list", "--tags", "security,scanning"])

        assert result.exit_code == 0
        assert "security" in result.stdout

    @patch("victor.core.verticals.registry_manager.VerticalRegistryManager.list_verticals")
    def test_list_with_verbose_output(self, mock_list):
        """Test list command with verbose output."""
        mock_list.return_value = [
            InstalledVertical(
                name="security",
                version="1.0.0",
                location=Path("/fake"),
                is_builtin=False,
                metadata=VerticalPackageMetadata(
                    name="security",
                    version="1.0.0",
                    description="Security analysis",
                    authors=[AuthorInfo(name="Test")],
                    license="Apache-2.0",
                    requires_victor=">=0.5.0",
                    category="security",
                    class_spec=VerticalClassSpec(
                        module="test",
                        class_name="Test",
                        provides_tools=["scan", "audit"],
                        provides_workflows=["security_review"],
                    ),
                ),
            ),
        ]

        result = runner.invoke(vertical_app, ["list", "--verbose"])

        assert result.exit_code == 0
        assert "security" in result.stdout
        # Verbose output should include additional columns
        assert "Category" in result.stdout or "Tools" in result.stdout

    @patch("victor.core.verticals.registry_manager.VerticalRegistryManager.list_verticals")
    def test_list_no_results_after_filtering(self, mock_list):
        """Test list command when filtering returns no results."""
        mock_list.return_value = [
            InstalledVertical(
                name="coding",
                version="0.5.1",
                location=Path("/fake"),
                is_builtin=True,
                metadata=VerticalPackageMetadata(
                    name="coding",
                    version="0.5.1",
                    description="Coding assistant",
                    authors=[AuthorInfo(name="Test")],
                    license="Apache-2.0",
                    requires_victor=">=0.5.0",
                    category="development",
                    class_spec=VerticalClassSpec(
                        module="test",
                        class_name="Test",
                    ),
                ),
            ),
        ]

        result = runner.invoke(vertical_app, ["list", "--category", "security"])

        assert result.exit_code == 0
        assert "No verticals found" in result.stdout or "no verticals" in result.stdout.lower()


class TestVerticalInstallation:
    """Tests for vertical installation workflow."""

    @patch("subprocess.run")
    @patch("victor.core.verticals.registry_manager.VerticalRegistryManager._validate_package")
    def test_install_with_validation_success(self, mock_validate, mock_run):
        """Test installation with validation enabled passes."""
        mock_validate.return_value = []  # No errors
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Successfully installed",
            stderr="",
        )

        result = runner.invoke(vertical_app, ["install", "victor-security"])

        assert result.exit_code == 0
        assert "Successfully installed" in result.stdout or "Success" in result.stdout
        mock_validate.assert_called_once()

    @patch("subprocess.run")
    @patch("victor.core.verticals.registry_manager.VerticalRegistryManager._validate_package")
    def test_install_skip_validation(self, mock_validate, mock_run):
        """Test installation with validation skipped."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Successfully installed",
            stderr="",
        )

        result = runner.invoke(vertical_app, ["install", "victor-security", "--no-validate"])

        assert result.exit_code == 0
        # Validation should not be called when --no-validate is used
        mock_validate.assert_not_called()

    @patch("victor.core.verticals.registry_manager.VerticalRegistryManager.install")
    def test_install_dry_run_mode(self, mock_install):
        """Test installation in dry-run mode."""
        mock_install.return_value = (True, "Would install: pip install victor-security")

        result = runner.invoke(vertical_app, ["install", "victor-security", "--dry-run"])

        assert result.exit_code == 0
        assert "Would install" in result.stdout

    @patch("subprocess.run")
    @patch("victor.core.verticals.registry_manager.VerticalRegistryManager._validate_package")
    def test_install_validation_failure(self, mock_validate, mock_run):
        """Test installation fails validation."""
        from victor.core.verticals.registry_manager import PackageSpec

        mock_validate.return_value = ["Package name conflicts with built-in vertical"]
        result = runner.invoke(vertical_app, ["install", "victor-security"])

        assert result.exit_code == 1
        assert "Validation failed" in result.stdout or "conflicts" in result.stdout.lower()

    @patch("subprocess.run")
    @patch("victor.core.verticals.registry_manager.VerticalRegistryManager._validate_package")
    def test_install_pip_failure(self, mock_validate, mock_run):
        """Test installation when pip install fails."""
        mock_validate.return_value = []
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "pip", stderr="Package not found"
        )

        result = runner.invoke(vertical_app, ["install", "nonexistent-package"])

        assert result.exit_code == 1
        assert "failed" in result.stdout.lower()


class TestVerticalUninstallation:
    """Tests for vertical uninstallation workflow."""

    @patch("subprocess.run")
    def test_uninstall_builtin_fails(self, mock_run):
        """Test uninstalling built-in vertical fails."""
        result = runner.invoke(vertical_app, ["uninstall", "coding"])

        assert result.exit_code == 1
        assert "Cannot uninstall built-in" in result.stdout

    @patch("subprocess.run")
    def test_uninstall_external_success(self, mock_run):
        """Test uninstalling external vertical succeeds."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Successfully uninstalled",
            stderr="",
        )

        result = runner.invoke(vertical_app, ["uninstall", "victor-security"])

        assert result.exit_code == 0
        assert "Successfully uninstalled" in result.stdout or "Success" in result.stdout

    @patch("subprocess.run")
    def test_uninstall_dry_run(self, mock_run):
        """Test uninstallation in dry-run mode."""
        result = runner.invoke(vertical_app, ["uninstall", "victor-security", "--dry-run"])

        assert result.exit_code == 0
        assert "Would uninstall" in result.stdout


class TestVerticalSearch:
    """Tests for vertical search functionality."""

    @patch("victor.core.verticals.registry_manager.VerticalRegistryManager.search")
    def test_search_by_name(self, mock_search):
        """Test searching by vertical name."""
        mock_search.return_value = [
            InstalledVertical(
                name="security",
                version="1.0.0",
                location=Path("/fake"),
                metadata=VerticalPackageMetadata(
                    name="security",
                    version="1.0.0",
                    description="Security analysis",
                    authors=[AuthorInfo(name="Test")],
                    license="Apache-2.0",
                    requires_victor=">=0.5.0",
                    class_spec=VerticalClassSpec(
                        module="test",
                        class_name="Test",
                    ),
                ),
            ),
        ]

        result = runner.invoke(vertical_app, ["search", "security"])

        assert result.exit_code == 0
        assert "security" in result.stdout
        mock_search.assert_called_once_with("security")

    @patch("victor.core.verticals.registry_manager.VerticalRegistryManager.search")
    def test_search_no_results(self, mock_search):
        """Test search returns no results."""
        mock_search.return_value = []

        result = runner.invoke(vertical_app, ["search", "nonexistent"])

        assert result.exit_code == 0
        assert "No verticals found" in result.stdout or "not found" in result.stdout.lower()

    @patch("victor.core.verticals.registry_manager.VerticalRegistryManager.search")
    def test_search_multiple_results(self, mock_search):
        """Test search returns multiple results."""
        mock_search.return_value = [
            InstalledVertical(
                name="security",
                version="1.0.0",
                location=Path("/fake"),
                metadata=VerticalPackageMetadata(
                    name="security",
                    version="1.0.0",
                    description="Security analysis",
                    authors=[AuthorInfo(name="Test")],
                    license="Apache-2.0",
                    requires_victor=">=0.5.0",
                    class_spec=VerticalClassSpec(
                        module="test",
                        class_name="Test",
                    ),
                ),
            ),
            InstalledVertical(
                name="security_audit",
                version="2.0.0",
                location=Path("/fake2"),
                metadata=VerticalPackageMetadata(
                    name="security_audit",
                    version="2.0.0",
                    description="Security audit tools",
                    authors=[AuthorInfo(name="Test")],
                    license="Apache-2.0",
                    requires_victor=">=0.5.0",
                    class_spec=VerticalClassSpec(
                        module="test2",
                        class_name="Test2",
                    ),
                ),
            ),
        ]

        result = runner.invoke(vertical_app, ["search", "security"])

        assert result.exit_code == 0
        assert "security" in result.stdout
        assert "Found 2 result" in result.stdout or "2 result" in result.stdout
