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

"""ML models for team optimization and prediction.

This package contains machine learning models for:
- Team member selection
- Formation prediction
- Performance prediction
- Expertise matching
"""

from victor.teams.ml.team_member_selector import TeamMemberSelector
from victor.teams.ml.formation_predictor import FormationPredictor
from victor.teams.ml.performance_predictor import PerformancePredictor

__all__ = [
    "TeamMemberSelector",
    "FormationPredictor",
    "PerformancePredictor",
]
