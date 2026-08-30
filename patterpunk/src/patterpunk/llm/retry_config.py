from dataclasses import dataclass


@dataclass(frozen=True)
class RetryConfig:
    delays_s: tuple[float, ...]
    jitter: tuple[float, float] = (0.5, 1.0)

    def __post_init__(self):
        if any(delay < 0 for delay in self.delays_s):
            raise ValueError("delays_s values must be non-negative")
        low, high = self.jitter
        if not 0 <= low <= high:
            raise ValueError("jitter must satisfy 0 <= low <= high")
