"""Unit tests for the RetryConfig dataclass (patterpunk.llm.retry_config)."""

import dataclasses

import pytest

from patterpunk.llm.retry_config import RetryConfig


def test_default_jitter_is_half_to_one():
    config = RetryConfig(delays_s=(1.0,))
    assert config.jitter == (0.5, 1.0)


def test_frozen_rejects_mutation():
    config = RetryConfig(delays_s=(1.0,))
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.delays_s = (2.0,)


def test_negative_delay_rejected():
    with pytest.raises(ValueError):
        RetryConfig(delays_s=(30.0, -1.0))


def test_inverted_jitter_rejected():
    with pytest.raises(ValueError):
        RetryConfig(delays_s=(30.0,), jitter=(1.0, 0.5))


def test_negative_jitter_rejected():
    with pytest.raises(ValueError):
        RetryConfig(delays_s=(30.0,), jitter=(-0.1, 1.0))


def test_empty_delays_allowed():
    # Degenerates to a single attempt with no retries.
    config = RetryConfig(delays_s=())
    assert config.delays_s == ()


def test_deterministic_jitter_allowed():
    config = RetryConfig(delays_s=(30.0,), jitter=(1.0, 1.0))
    assert config.jitter == (1.0, 1.0)


def test_outage_schedule_construction():
    # The schedule shape this feature was built for: one short delay for
    # transient blips, then outage-sized delays summing to ~3h.
    config = RetryConfig(delays_s=(30, 15 * 60, 45 * 60, 120 * 60))
    assert len(config.delays_s) == 4
