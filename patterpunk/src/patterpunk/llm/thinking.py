from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class ThinkingConfig:
    effort: Optional[Literal["low", "medium", "high"]] = None
    token_budget: Optional[int] = None
    include_thoughts: bool = False

    def __post_init__(self):
        if (self.effort is None) == (self.token_budget is None):
            raise ValueError("Must specify exactly one of: effort or token_budget")

        if self.token_budget is not None and self.token_budget < 0:
            raise ValueError("token_budget must be non-negative (use 0 to disable)")
