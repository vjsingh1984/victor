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

"""Vertical Registry Manager for package discovery and installation.

This module provides the VerticalRegistryManager class that handles discovery,
installation, validation, and management of vertical packages from multiple
sources including PyPI, git repositories, and local paths.

Design Principles:
- Multi-source support: Built-in, PyPI, git, local paths
- Validation: Security checks and version compatibility
- Metadata caching: Fast lookup via victor-vertical.toml
- Dry-run mode: Safe preview of install/uninstall operations

Key Classes:
    VerticalRegistryManager: Main class for vertical package management
    PackageSpec: Specification for parsing package strings
    InstalledVertical: Information about installed verticals
    PackageSourceType: Enum of supported package sources

Usage:
    from victor.core.verticals.registry_manager import VerticalRegistryManager, PackageSpec

    # List all available verticals
    manager = VerticalRegistryManager()
    verticals = manager.list_verticals(source="all")

    # Search for verticals
    results = manager.search("security")

    # Install a vertical
    spec = PackageSpec.parse("victor-security>=1.0.0")
    success, message = manager.install(spec)

    # Install from git
    git_spec = PackageSpec.parse("git+https://github.com/user/victor-security.git")
    success, message = manager.install(git_spec)

    # Get detailed info
    info = manager.get_info("security")
    print(f"Version: {info.version}, Location: {info.location}")

Related Modules:
    victor.core.verticals.vertical_loader: Runtime vertical activation
    victor.core.verticals.package_schema: Metadata schema definitions
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import httpx

from victor.core.verticals.package_schema import VerticalPackageMetadata

logger = logging.getLogger(__name__)


class PackageSourceType(Enum):
    """Type of package source."""

    PYPI = "pypi"
    GIT = "git"
    LOCAL = "local"
    BUILTIN = "builtin"


@dataclass
class PackageSpec:
    """Specification for a vertical package.

    Attributes:
        name: Package name (e.g., "victor-security")
        version: Optional version constraint
        source: Source type (pypi, git, local, builtin)
        url: URL for git or local path
        extras: Optional list of extras
    """

    name: str
    version: Optional[str] = None
    source: PackageSourceType = PackageSourceType.PYPI
    url: Optional[str] = None
    extras: List[str] = field(default_factory=list)

    @classmethod
    def parse(cls, spec_str: str) -> "PackageSpec":
        """Parse a package specification string.

        Args:
            spec_str: Package spec like:
                - "victor-security"
                - "victor-security>=1.0.0"
                - "git+https://github.com/user/victor-security.git"
                - "/path/to/local/package"
                - "victor-security[extra1,extra2]"

        Returns:
            PackageSpec instance

        Raises:
            ValueError: If spec string is invalid
        """
        spec_str = spec_str.strip()

        # Git URL
        if spec_str.startswith("git+"):
            url = spec_str[4:]
            name = cls._extract_name_from_git_url(url)
            return cls(name=name, source=PackageSourceType.GIT, url=url)

        # Local path
        if spec_str.startswith("/") or spec_str.startswith("./") or spec_str.startswith("../"):
            path = Path(spec_str).resolve()
            name = path.name
            return cls(name=name, source=PackageSourceType.LOCAL, url=str(path))

        # PyPI package with extras
        if "[" in spec_str and "]" in spec_str:
            base_part = spec_str[: spec_str.index("[")]
            extras_part = spec_str[spec_str.index("[") + 1 : spec_str.index("]")]
            extras = [e.strip() for e in extras_part.split(",")]
            # Parse base part first
            base_spec = cls.parse(base_part)
            # Return new spec with extras
            return cls(
                name=base_spec.name,
                version=base_spec.version,
                source=base_spec.source,
                url=base_spec.url,
                extras=extras,
            )

        # PyPI package with version
        if any(op in spec_str for op in [">", "<", "=", "~", "!", "^"]):
            for op in [">=", "<=", "==", "!=", "~=", "^", ">", "<", "=="]:
                if op in spec_str:
                    name, version = spec_str.split(op, 1)
                    return cls(name=name.strip(), version=f"{op}{version.strip()}")
            return cls(name=spec_str)

        # Simple package name
        return cls(name=spec_str)

    @staticmethod
    def _extract_name_from_git_url(url: str) -> str:
        """Extract package name from git URL."""
        # Remove .git suffix if present
        if url.endswith(".git"):
            url = url[:-4]

        # Get last part of path
        path = urlparse(url).path
        name = path.rstrip("/").split("/")[-1]
        return name

    def to_pip_string(self) -> str:
        """Convert to pip install string.

        Returns:
            String suitable for pip install
        """
        if self.source == PackageSourceType.GIT:
            return f"git+{self.url}"
        elif self.source == PackageSourceType.LOCAL:
            return str(self.url)
        elif self.source == PackageSourceType.PYPI:
            result = self.name
            if self.version:
                result += self.version
            if self.extras:
                result += f"[{','.join(self.extras)}]"
            return result
        else:
            return self.name


@dataclass
class InstalledVertical:
    """Information about an installed vertical.

    Attributes:
        name: Vertical name
        version: Installed version
        location: Installation location
        metadata: Package metadata (if available)
        is_builtin: Whether this is a built-in vertical
    """

    name: str
    version: str
    location: Path
    metadata: Optional[VerticalPackageMetadata] = None
    is_builtin: bool = False


class VerticalRegistryManager:
    """Manager for vertical package discovery and installation.

    This class handles all operations related to vertical packages:
    - Discovery of built-in and installed verticals
    - Installation from PyPI, git, or local paths
    - Validation and security checks
    - Uninstallation
    - Search functionality
    """

    # Built-in verticals
    BUILTIN_VERTICALS = [
        "coding",
        "devops",
        "rag",
        "dataanalysis",
        "research",
        "benchmark",
    ]

    # Default registry URL (can be overridden via settings)
    DEFAULT_REGISTRY_URL = (
        "https://raw.githubusercontent.com/vjsingh1984/victor-registry/main/index.json"
    )

    def __init__(
        self,
        registry_url: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        dry_run: bool = False,
    ) -> None:
        """Initialize the registry manager.

        Args:
            registry_url: Optional custom registry URL
            cache_dir: Optional cache directory for metadata
            dry_run: If True, don't actually install/uninstall
        """
        self.registry_url = registry_url or self.DEFAULT_REGISTRY_URL
        self.dry_run = dry_run

        # Setup cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".victor" / "cache" / "verticals"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._metadata_cache: Dict[str, VerticalPackageMetadata] = {}

    def list_verticals(
        self,
        source: str = "all",
    ) -> List[InstalledVertical]:
        """List verticals.

        Args:
            source: One of "all", "installed", "builtin", "available"

        Returns:
            List of InstalledVertical instances
        """
        victor_dir = Path(__file__).parent.parent.parent

        if source == "builtin":
            return self._list_builtin_verticals(victor_dir)
        elif source == "installed":
            return self._list_installed_verticals()
        elif source == "available":
            return self._list_available_verticals()
        else:  # "all"
            builtin = self._list_builtin_verticals(victor_dir)
            installed = self._list_installed_verticals()
            # Merge, removing duplicates
            return self._merge_verticals(builtin, installed)

    def _list_builtin_verticals(self, victor_dir: Path) -> List[InstalledVertical]:
        """List built-in verticals."""
        verticals = []

        for name in self.BUILTIN_VERTICALS:
            vertical_dir = victor_dir / name
            if vertical_dir.exists():
                # Try to load metadata
                metadata = None
                metadata_file = vertical_dir / "victor-vertical.toml"
                if metadata_file.exists():
                    try:
                        metadata = VerticalPackageMetadata.from_toml(metadata_file)
                    except Exception as e:
                        logger.warning(f"Failed to load metadata for {name}: {e}")

                verticals.append(
                    InstalledVertical(
                        name=name,
                        version="builtin",
                        location=vertical_dir,
                        metadata=metadata,
                        is_builtin=True,
                    )
                )

        return verticals

    def _list_installed_verticals(self) -> List[InstalledVertical]:
        """List installed external verticals.

        Uses importlib.metadata to discover packages that register
        verticals via entry points.
        """
        verticals = []

        try:
            from importlib.metadata import distributions

            for dist in distributions():
                # Check for victor.verticals entry point
                entry_points = list(dist.entry_points.select(group="victor.verticals"))
                if entry_points:
                    for ep in entry_points:
                        # Try to load metadata from package
                        metadata = self._load_metadata_from_dist(ep, dist)

                        verticals.append(
                            InstalledVertical(
                                name=ep.name,
                                version=dist.version,
                                location=Path(dist.locate_file("")),
                                metadata=metadata,
                                is_builtin=False,
                            )
                        )
        except Exception as e:
            logger.warning(f"Failed to list installed verticals: {e}")

        return verticals

    def _load_metadata_from_dist(
        self,
        entry_point: Any,
        dist: Any,
    ) -> Optional[VerticalPackageMetadata]:
        """Load metadata from an installed distribution.

        Args:
            entry_point: Entry point object
            dist: Distribution object

        Returns:
            VerticalPackageMetadata if found, None otherwise
        """
        try:
            # Check for victor-vertical.toml in package
            package_dir = Path(dist.locate_file(""))
            metadata_file = package_dir / "victor-vertical.toml"

            if metadata_file.exists():
                return VerticalPackageMetadata.from_toml(metadata_file)

        except Exception as e:
            logger.debug(f"Failed to load metadata for {entry_point.name}: {e}")

        return None

    def _list_available_verticals(self) -> List[InstalledVertical]:
        """List verticals available in the remote registry.

        Returns:
            List of available verticals (cached or from registry)
        """
        cache_file = self.cache_dir / "available.json"

        # Try to load from cache
        if cache_file.exists():
            try:
                cached_data = json.loads(cache_file.read_text())
                # Check cache age (1 hour)
                import time

                if time.time() - cached_data.get("timestamp", 0) < 3600:
                    verticals = []
                    for item in cached_data.get("verticals", []):
                        verticals.append(
                            InstalledVertical(
                                name=item["name"],
                                version=item.get("version", "unknown"),
                                location=Path(item.get("location", "")),
                                is_builtin=False,
                            )
                        )
                    return verticals
            except Exception as e:
                logger.debug(f"Failed to load cached verticals: {e}")

        # Fetch from registry
        try:
            verticals = self._fetch_available_verticals()
            # Cache the results
            self._cache_available_verticals(verticals)
            return verticals
        except Exception as e:
            logger.warning(f"Failed to fetch available verticals: {e}")
            return []

    def _fetch_available_verticals(self) -> List[InstalledVertical]:
        """Fetch available verticals from remote registry.

        Returns:
            List of available verticals

        Raises:
            httpx.HTTPError: If registry query fails
        """
        try:
            response = httpx.get(
                self.registry_url,
                timeout=10.0,
                follow_redirects=True,
            )
            response.raise_for_status()

            data = response.json()
            verticals = []

            for item in data.get("verticals", []):
                verticals.append(
                    InstalledVertical(
                        name=item["name"],
                        version=item.get("version", "unknown"),
                        location=Path(item.get("repository", "")),
                        is_builtin=False,
                    )
                )

            return verticals

        except Exception as e:
            logger.debug(f"Registry query failed: {e}")
            # Return empty list on error
            return []

    def _cache_available_verticals(self, verticals: List[InstalledVertical]) -> None:
        """Cache available verticals to disk.

        Args:
            verticals: List of verticals to cache
        """
        try:
            import time

            cache_file = self.cache_dir / "available.json"
            data = {
                "timestamp": time.time(),
                "verticals": [
                    {
                        "name": v.name,
                        "version": v.version,
                        "location": str(v.location),
                    }
                    for v in verticals
                ],
            }
            cache_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"Failed to cache available verticals: {e}")

    def _merge_verticals(
        self,
        *lists: List[InstalledVertical],
    ) -> List[InstalledVertical]:
        """Merge multiple lists of verticals, removing duplicates.

        Args:
            *lists: Variable number of vertical lists

        Returns:
            Merged list with duplicates removed (builtin takes precedence)
        """
        seen: Set[str] = set()
        merged = []

        # Process in order: builtin first, then installed
        for vertical_list in lists:
            for vertical in vertical_list:
                if vertical.name not in seen:
                    seen.add(vertical.name)
                    merged.append(vertical)

        return merged

    def search(self, query: str) -> List[InstalledVertical]:
        """Search for verticals matching query.

        Args:
            query: Search query (matches name, description, tags)

        Returns:
            List of matching verticals
        """
        query_lower = query.lower()
        all_verticals = self.list_verticals(source="all")
        results = []

        for vertical in all_verticals:
            # Search in name
            if query_lower in vertical.name.lower():
                results.append(vertical)
                continue

            # Search in metadata
            if vertical.metadata:
                if query_lower in vertical.metadata.description.lower():
                    results.append(vertical)
                    continue

                # Search in tags
                if any(query_lower in tag.lower() for tag in vertical.metadata.tags):
                    results.append(vertical)
                    continue

        return results

    def get_info(self, name: str) -> Optional[InstalledVertical]:
        """Get detailed information about a vertical.

        Args:
            name: Vertical name

        Returns:
            InstalledVertical if found, None otherwise
        """
        all_verticals = self.list_verticals(source="all")

        for vertical in all_verticals:
            if vertical.name == name:
                # If we don't have metadata, try to load it
                if not vertical.metadata:
                    metadata_file = vertical.location / "victor-vertical.toml"
                    if metadata_file.exists():
                        try:
                            vertical.metadata = VerticalPackageMetadata.from_toml(metadata_file)
                        except Exception:
                            pass
                return vertical

        return None

    def install(
        self,
        package_spec: PackageSpec,
        validate: bool = True,
        verbose: bool = False,
    ) -> Tuple[bool, str]:
        """Install a vertical package.

        Args:
            package_spec: Package specification
            validate: Whether to validate before installation
            verbose: Whether to show detailed output

        Returns:
            Tuple of (success, message)
        """
        # Validation
        if validate:
            validation_errors = self._validate_package(package_spec)
            if validation_errors:
                return False, "Validation failed:\n" + "\n".join(validation_errors)

        # Build pip command
        pip_string = package_spec.to_pip_string()
        cmd = ["pip", "install", pip_string]

        if verbose:
            cmd.append("-v")

        # Dry run mode
        if self.dry_run:
            return True, f"Would install: {' '.join(cmd)}"

        # Execute installation
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            # Success
            return True, f"Successfully installed {package_spec.name}"

        except subprocess.CalledProcessError as e:
            return False, f"Installation failed:\n{e.stderr}"

    def uninstall(self, name: str, verbose: bool = False) -> Tuple[bool, str]:
        """Uninstall a vertical package.

        Args:
            name: Package name to uninstall
            verbose: Whether to show detailed output

        Returns:
            Tuple of (success, message)
        """
        # Check if it's a built-in vertical
        if name in self.BUILTIN_VERTICALS:
            return False, f"Cannot uninstall built-in vertical: {name}"

        # Build pip command
        cmd = ["pip", "uninstall", "-y", name]

        if verbose:
            cmd.append("-v")

        # Dry run mode
        if self.dry_run:
            return True, f"Would uninstall: {' '.join(cmd)}"

        # Execute uninstallation
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            # Success
            return True, f"Successfully uninstalled {name}"

        except subprocess.CalledProcessError as e:
            return False, f"Uninstallation failed:\n{e.stderr}"

    def _validate_package(self, package_spec: PackageSpec) -> List[str]:
        """Validate a package before installation.

        Args:
            package_spec: Package specification

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check for reserved names
        if package_spec.name in self.BUILTIN_VERTICALS:
            errors.append(f"Package name '{package_spec.name}' conflicts with built-in vertical")

        # For git and local packages, try to load metadata
        if package_spec.source in [PackageSourceType.GIT, PackageSourceType.LOCAL]:
            try:
                metadata = self._load_remote_metadata(package_spec)
                if metadata:
                    # Check Victor version compatibility
                    current_version = self._get_victor_version()
                    if not self._check_version_compatibility(
                        current_version,
                        metadata.requires_victor,
                    ):
                        errors.append(
                            f"Package requires Victor {metadata.requires_victor}, "
                            f"but current version is {current_version}"
                        )
            except Exception as e:
                errors.append(f"Failed to load package metadata: {e}")

        return errors

    def _load_remote_metadata(
        self,
        package_spec: PackageSpec,
    ) -> Optional[VerticalPackageMetadata]:
        """Load metadata from a remote or local package.

        Args:
            package_spec: Package specification

        Returns:
            VerticalPackageMetadata if found, None otherwise
        """
        if package_spec.source == PackageSourceType.LOCAL:
            # Load from local path
            metadata_file = Path(package_spec.url) / "victor-vertical.toml"
            if metadata_file.exists():
                return VerticalPackageMetadata.from_toml(metadata_file)

        elif package_spec.source == PackageSourceType.GIT:
            # For git repos, we'd need to clone temporarily
            # This is expensive, so skip for now
            pass

        return None

    def _get_victor_version(self) -> str:
        """Get current Victor version.

        Returns:
            Version string
        """
        try:
            from importlib.metadata import version

            return version("victor-ai")
        except Exception:
            return "0.0.0"

    def _check_version_compatibility(
        self,
        current: str,
        required: str,
    ) -> bool:
        """Check if current version satisfies requirement.

        Args:
            current: Current version
            required: Required version spec (e.g., ">=0.5.0")

        Returns:
            True if compatible, False otherwise
        """
        try:
            from packaging.requirements import Requirement
            from packaging.version import Version

            req = Requirement(f"victor-ai{required}")
            return Version(current) in req.specifier
        except Exception:
            # Assume compatible if we can't check
            return True

    def clear_cache(self) -> None:
        """Clear the metadata cache."""
        try:
            cache_file = self.cache_dir / "available.json"
            if cache_file.exists():
                cache_file.unlink()
            self._metadata_cache.clear()
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
