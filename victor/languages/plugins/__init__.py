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

"""Built-in language plugins.

Provides language support for:
- Python (pytest, black, ruff)
- JavaScript (jest, prettier, eslint)
- TypeScript (jest, prettier, eslint)
- Rust (cargo test, rustfmt, clippy)
- Go (go test, gofmt, golint)
- Java (junit, google-java-format, checkstyle)
"""

from victor.languages.plugins.python import PythonPlugin
from victor.languages.plugins.javascript import JavaScriptPlugin
from victor.languages.plugins.typescript import TypeScriptPlugin
from victor.languages.plugins.rust import RustPlugin
from victor.languages.plugins.go import GoPlugin
from victor.languages.plugins.java import JavaPlugin

__all__ = [
    "PythonPlugin",
    "JavaScriptPlugin",
    "TypeScriptPlugin",
    "RustPlugin",
    "GoPlugin",
    "JavaPlugin",
]
