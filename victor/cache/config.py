"""Cache configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CacheConfig:
    """Cache configuration settings."""

    # Memory cache settings
    memory_max_size: int = 1000  # Max items in memory
    memory_ttl: int = 300  # 5 minutes TTL for memory cache

    # Disk cache settings
    disk_max_size: int = 1024 * 1024 * 1024  # 1GB max size
    disk_ttl: int = 86400 * 7  # 7 days TTL for disk cache
    disk_path: Optional[Path] = None  # Cache directory

    # Cache behavior
    enable_memory: bool = True  # Enable L1 memory cache
    enable_disk: bool = True  # Enable L2 disk cache
    auto_evict: bool = True  # Auto-evict old entries

    def __post_init__(self):
        """Set default cache path if not provided."""
        if self.disk_path is None:
            self.disk_path = Path.home() / ".victor" / "cache"

        # Ensure cache directory exists
        if self.enable_disk:
            self.disk_path.mkdir(parents=True, exist_ok=True)
