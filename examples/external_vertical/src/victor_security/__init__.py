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

"""Victor Security Vertical - Example External Plugin Package.

This package demonstrates how to create an external Victor vertical that can be:
1. Installed via pip: `pip install victor-security` (or `pip install -e .` for development)
2. Automatically discovered by Victor through Python entry points
3. Used alongside built-in verticals (coding, research, devops, etc.)

The Security vertical provides:
- Security vulnerability scanning and analysis
- Dependency auditing
- Code security review
- Configuration security assessment
- Compliance checking

Usage:
    # After installing the package, Victor automatically discovers it
    victor --vertical security

    # Or programmatically:
    from victor.core.verticals import VerticalRegistry

    # Discover external verticals (happens automatically on startup)
    VerticalRegistry.discover_external_verticals()

    # Get the security vertical
    security = VerticalRegistry.get("security")

    # Use the security vertical
    config = security.get_config()
    extensions = security.get_extensions()

Entry Point Discovery:
    Victor discovers this vertical via the entry point defined in pyproject.toml:

    [project.entry-points."victor.verticals"]
    security = "victor_security:SecurityAssistant"

    This tells Victor to load SecurityAssistant from this module when looking
    for verticals named "security".

Package Structure:
    victor_security/
    |-- __init__.py     # This file - exports SecurityAssistant
    |-- assistant.py    # SecurityAssistant class (VerticalBase subclass)
    |-- safety.py       # Security-specific safety patterns
    |-- prompts.py      # Security task hints and prompt contributions
"""

from victor_security.assistant import SecurityAssistant
from victor_security.prompts import SecurityPromptContributor
from victor_security.safety import SecuritySafetyExtension

__version__ = "0.1.0"

__all__ = [
    "SecurityAssistant",
    "SecuritySafetyExtension",
    "SecurityPromptContributor",
]
