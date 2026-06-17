"""Prompt optimization strategies.

Available strategies:
- GEPAStrategy: Reflective prompt evolution (rules + guidance)
- MIPROv2Strategy: Few-shot demonstration mining from traces
- CoTDistillationStrategy: Chain-of-thought distillation from strong models
"""

from victor.framework.rl.learners.strategies.miprov2_strategy import (
    MIPROv2Strategy,
)
from victor.framework.rl.learners.strategies.cot_distillation_strategy import (
    CoTDistillationStrategy,
)

__all__ = ["MIPROv2Strategy", "CoTDistillationStrategy"]
