from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class ThinkingConfig:
    effort: Optional[Literal["low", "medium", "high", "xhigh", "max"]] = None
    token_budget: Optional[int] = None
    include_thoughts: bool = False

    def __post_init__(self):
        if (self.effort is None) == (self.token_budget is None):
            raise ValueError("Must specify exactly one of: effort or token_budget")

        if self.token_budget is not None and self.token_budget < 0:
            raise ValueError("token_budget must be non-negative (use 0 to disable)")


EFFORT_TO_BUDGET = {"low": 1500, "medium": 4000, "high": 12000}


def effort_for_budget(token_budget: int) -> str:
    for effort, budget in EFFORT_TO_BUDGET.items():
        if token_budget <= budget:
            return effort
    return "high"
