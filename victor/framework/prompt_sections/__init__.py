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

"""Structured prompt sections for Victor Framework.

This package provides composable, type-safe prompt section classes that
can be used with PromptBuilder for constructing system prompts.

Section Types:
- GroundingSection: Contextual grounding with variable substitution
- RuleSection: Ordered list of behavioral rules
- ChecklistSection: Checklist items for verification
- VerticalPromptSection: Vertical-specific content
"""

from victor.framework.prompt_sections.grounding import GroundingSection
from victor.framework.prompt_sections.rules import RuleSection
from victor.framework.prompt_sections.checklist import ChecklistSection
from victor.framework.prompt_sections.vertical import VerticalPromptSection

__all__ = [
    "GroundingSection",
    "RuleSection",
    "ChecklistSection",
    "VerticalPromptSection",
]
