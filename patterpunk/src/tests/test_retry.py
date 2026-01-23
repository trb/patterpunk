"""
Tests for the retry backoff utility module.

These tests verify:
1. Exponential backoff calculation
2. Jitter application
3. Min/max delay clamping
4. Retry-After header extraction
5. Retryable error detection

Run with:
    docker compose -p patterpunk run --rm patterpunk -c '/app/bin/test.dev /app/tests/test_retry.py'
"""

import pytest
from unittest.mock import Mock

from patterpunk.lib.retry import (
    calculate_backoff_delay,
    extract_retry_after,
    is_retryable_error,
)


class TestCalculateBackoffDelay:
    """Tests for the calculate_backoff_delay function."""

    def test_exponential_growth_without_jitter(self):
        """Verify exponential growth with jitter disabled."""
        # With jitter_factor=0, delay should be exactly base_delay * 2^attempt
        delays = [
            calculate_backoff_delay(
                attempt=i,
                base_delay=60.0,
                max_delay=1000.0,
                min_delay=0.0,
                jitter_factor=0.0,
            )
            for i in range(4)
        ]

        assert delays[0] == 60.0  # 60 * 2^0 = 60
        assert delays[1] == 120.0  # 60 * 2^1 = 120
        assert delays[2] == 240.0  # 60 * 2^2 = 240
        assert delays[3] == 480.0  # 60 * 2^3 = 480

    def test_min_delay_clamping(self):
        """Verify that delays are clamped to min_delay."""
        # With small base_delay and high jitter, result should still be >= min_delay
        for _ in range(20):  # Run multiple times to account for jitter randomness
            delay = calculate_backoff_delay(
                attempt=0,
                base_delay=10.0,
                max_delay=1000.0,
                min_delay=45.0,
                jitter_factor=0.5,
            )
            assert delay >= 45.0, f"Delay {delay} should be >= min_delay 45.0"

    def test_max_delay_clamping(self):
        """Verify that delays are clamped to max_delay."""
        # With high attempt number, result should still be <= max_delay
        for _ in range(20):  # Run multiple times to account for jitter randomness
            delay = calculate_backoff_delay(
                attempt=10,  # Would be 60 * 2^10 = 61440 without clamping
                base_delay=60.0,
                max_delay=300.0,
                min_delay=0.0,
                jitter_factor=0.5,
            )
            assert delay <= 300.0, f"Delay {delay} should be <= max_delay 300.0"

    def test_jitter_within_expected_range(self):
        """Verify jitter stays within the specified factor range."""
        base_delay = 100.0
        jitter_factor = 0.5
        min_expected = base_delay * (1 - jitter_factor)  # 50
        max_expected = base_delay * (1 + jitter_factor)  # 150

        delays = set()
        for _ in range(100):  # Run many times to test randomness
            delay = calculate_backoff_delay(
                attempt=0,
                base_delay=base_delay,
                max_delay=1000.0,
                min_delay=0.0,
                jitter_factor=jitter_factor,
            )
            delays.add(delay)
            assert (
                min_expected <= delay <= max_expected
            ), f"Delay {delay} outside expected range [{min_expected}, {max_expected}]"

        # Verify we got different values (jitter is actually random)
        assert len(delays) > 1, "Jitter should produce varying delays"

    def test_retry_after_honored(self):
        """Verify that retry_after header value is honored."""
        # retry_after should override calculated delay when larger
        delay = calculate_backoff_delay(
            attempt=0,
            base_delay=60.0,
            max_delay=300.0,
            min_delay=45.0,
            jitter_factor=0.0,
            retry_after=120.0,  # Server says wait 120s
        )
        assert delay == 120.0, "Should honor retry_after when larger than calculated"

    def test_retry_after_ignored_when_smaller(self):
        """Verify that retry_after is ignored when smaller than calculated delay."""
        delay = calculate_backoff_delay(
            attempt=0,
            base_delay=60.0,
            max_delay=300.0,
            min_delay=45.0,
            jitter_factor=0.0,
            retry_after=30.0,  # Server says 30s but our calculated delay is 60s
        )
        assert delay == 60.0, "Should use calculated delay when larger than retry_after"

    def test_default_values_produce_expected_ranges(self):
        """Verify that default values produce delays appropriate for per-minute rate limits."""
        # Test with defaults from the implementation
        for _ in range(50):
            delay_attempt_0 = calculate_backoff_delay(attempt=0)
            delay_attempt_1 = calculate_backoff_delay(attempt=1)
            delay_attempt_2 = calculate_backoff_delay(attempt=2)

            # Attempt 0: 60s base with ±50% jitter = 30-90s, clamped to min 45s = 45-90s
            assert (
                45.0 <= delay_attempt_0 <= 90.0
            ), f"Attempt 0 delay {delay_attempt_0} outside expected 45-90s range"

            # Attempt 1: 120s base with ±50% jitter = 60-180s (within bounds)
            assert (
                60.0 <= delay_attempt_1 <= 180.0
            ), f"Attempt 1 delay {delay_attempt_1} outside expected 60-180s range"

            # Attempt 2: 240s base with ±50% jitter = 120-360s, clamped to max 300s = 120-300s
            assert (
                120.0 <= delay_attempt_2 <= 300.0
            ), f"Attempt 2 delay {delay_attempt_2} outside expected 120-300s range"


class TestExtractRetryAfter:
    """Tests for the extract_retry_after function."""

    def test_extract_from_response_headers(self):
        """Extract retry-after from response headers."""
        mock_error = Mock()
        mock_error.response = Mock()
        mock_error.response.headers = {"retry-after": "45"}

        result = extract_retry_after(mock_error)
        assert result == 45.0

    def test_extract_from_response_headers_uppercase(self):
        """Extract Retry-After from headers (case variation)."""
        mock_error = Mock()
        mock_error.response = Mock()
        mock_error.response.headers = {"Retry-After": "60"}

        result = extract_retry_after(mock_error)
        assert result == 60.0

    def test_extract_from_body(self):
        """Extract retry_after from error body."""
        mock_error = Mock()
        mock_error.response = None
        mock_error.body = {"retry_after": 90}

        result = extract_retry_after(mock_error)
        assert result == 90.0

    def test_returns_none_when_not_found(self):
        """Return None when retry-after is not found."""
        mock_error = Mock()
        mock_error.response = None
        mock_error.body = None

        result = extract_retry_after(mock_error)
        assert result is None

    def test_returns_none_for_invalid_value(self):
        """Return None when retry-after value is invalid."""
        mock_error = Mock()
        mock_error.response = Mock()
        mock_error.response.headers = {"retry-after": "invalid"}

        result = extract_retry_after(mock_error)
        assert result is None


class TestIsRetryableError:
    """Tests for the is_retryable_error function."""

    def test_429_is_retryable(self):
        """429 Too Many Requests should be retryable."""
        mock_error = Mock()
        mock_error.status_code = 429

        assert is_retryable_error(mock_error) is True

    def test_500_is_retryable(self):
        """500 Internal Server Error should be retryable."""
        mock_error = Mock()
        mock_error.status_code = 500

        assert is_retryable_error(mock_error) is True

    def test_502_is_retryable(self):
        """502 Bad Gateway should be retryable."""
        mock_error = Mock()
        mock_error.status_code = 502

        assert is_retryable_error(mock_error) is True

    def test_503_is_retryable(self):
        """503 Service Unavailable should be retryable."""
        mock_error = Mock()
        mock_error.status_code = 503

        assert is_retryable_error(mock_error) is True

    def test_504_is_retryable(self):
        """504 Gateway Timeout should be retryable."""
        mock_error = Mock()
        mock_error.status_code = 504

        assert is_retryable_error(mock_error) is True

    def test_400_is_not_retryable(self):
        """400 Bad Request should not be retryable."""
        mock_error = Mock()
        mock_error.status_code = 400

        assert is_retryable_error(mock_error) is False

    def test_401_is_not_retryable(self):
        """401 Unauthorized should not be retryable."""
        mock_error = Mock()
        mock_error.status_code = 401

        assert is_retryable_error(mock_error) is False

    def test_rate_limit_in_message_is_retryable(self):
        """Error message containing 'rate limit' should be retryable."""

        class MockError(Exception):
            pass

        error = MockError("Rate limit exceeded")
        assert is_retryable_error(error) is True

    def test_too_many_requests_in_message_is_retryable(self):
        """Error message containing 'too many requests' should be retryable."""

        class MockError(Exception):
            pass

        error = MockError("Too many requests")
        assert is_retryable_error(error) is True

    def test_connection_error_is_retryable(self):
        """Error message containing 'connection' should be retryable."""

        class MockError(Exception):
            pass

        error = MockError("Connection refused")
        assert is_retryable_error(error) is True

    def test_timeout_error_is_retryable(self):
        """Error message containing 'timeout' should be retryable."""

        class MockError(Exception):
            pass

        error = MockError("Request timeout")
        assert is_retryable_error(error) is True


class TestBackoffIntegration:
    """Integration tests for the retry backoff system."""

    def test_thundering_herd_prevention(self):
        """Verify jitter prevents synchronized retries."""
        # Simulate 10 clients hitting rate limit at the same time
        delays = [
            calculate_backoff_delay(
                attempt=0,
                base_delay=60.0,
                max_delay=300.0,
                min_delay=45.0,
                jitter_factor=0.5,
            )
            for _ in range(10)
        ]

        # All delays should be different (with very high probability)
        unique_delays = set(delays)
        assert (
            len(unique_delays) > 5
        ), "Jitter should produce varied delays across clients"

        # Delays should be spread across the expected range
        min_delay = min(delays)
        max_delay = max(delays)
        spread = max_delay - min_delay
        assert (
            spread > 10.0
        ), f"Delays should be spread out, but spread was only {spread}s"

    def test_progressive_backoff_increases_delay(self):
        """Verify that subsequent attempts use longer delays."""
        # With jitter disabled, we can verify exact exponential growth
        delay_0 = calculate_backoff_delay(
            attempt=0,
            base_delay=60.0,
            max_delay=300.0,
            min_delay=0.0,
            jitter_factor=0.0,
        )
        delay_1 = calculate_backoff_delay(
            attempt=1,
            base_delay=60.0,
            max_delay=300.0,
            min_delay=0.0,
            jitter_factor=0.0,
        )
        delay_2 = calculate_backoff_delay(
            attempt=2,
            base_delay=60.0,
            max_delay=300.0,
            min_delay=0.0,
            jitter_factor=0.0,
        )

        assert delay_1 > delay_0, "Attempt 1 should have longer delay than attempt 0"
        assert delay_2 > delay_1, "Attempt 2 should have longer delay than attempt 1"
        assert delay_1 == 2 * delay_0, "Delay should double with each attempt"
        assert delay_2 == 2 * delay_1, "Delay should double with each attempt"
