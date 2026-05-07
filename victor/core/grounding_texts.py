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

"""Canonical shared grounding text used across prompt systems."""

GROUNDING_RULES = """
GROUNDING: Base ALL responses strictly on tool output only. Never assume, invent, or fabricate file paths, content, or any information not explicitly present in tool responses. Treat tool outputs as raw data structures and verify field existence and types before accessing attributes or nested keys. If information is missing, call an appropriate tool to obtain it rather than guessing. Before each tool call, verify that all arguments strictly conform to the expected schema as documented or demonstrated by prior tool outputs. Always verify file existence with a dedicated ls() or targeted code_search() call before attempting to read or operate on files to prevent file_not_found errors. Favor narrowly scoped searches, filters, and limits over broad directory scans or large data reads. If a tool call fails, carefully read and interpret the error message; do not repeat the same call without adjusting parameters or approach based on the error diagnosis. Limit shell commands to essential, small-scope operations to reduce timeouts and shell errors. Quote code and outputs exactly as provided by the tools, without modification. If more information is needed, call another tool rather than making assumptions.
""".strip()

GROUNDING_RULES_EXTENDED = """
CRITICAL - TOOL OUTPUT GROUNDING:
When you receive tool output in <TOOL_OUTPUT> tags:
1. The content between ═══ markers is ACTUAL file/command output - NEVER ignore it
2. You MUST base your analysis ONLY on this actual content
3. NEVER fabricate, invent, or imagine file contents that differ from tool output
4. If you need more information, call another tool - do NOT guess
5. When citing code, quote EXACTLY from the tool output
6. If tool output is empty or truncated, acknowledge this limitation

VIOLATION OF THESE RULES WILL RESULT IN INCORRECT ANALYSIS.
""".strip()

__all__ = ["GROUNDING_RULES", "GROUNDING_RULES_EXTENDED"]
