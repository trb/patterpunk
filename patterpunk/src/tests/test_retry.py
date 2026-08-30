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

import ssl

import httpx
import httpx2
import pytest
from unittest.mock import Mock, patch

from patterpunk.lib.retry import (
    calculate_backoff_delay,
    extract_retry_after,
    is_retryable_error,
    run_with_retry_config,
    stream_with_retry_config,
)
from patterpunk.llm.retry_config import RetryConfig


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

    def test_408_status_code_is_retryable(self):
        """408 Request Timeout should be retryable via the status_code attribute."""

        class FakeStatusError(Exception):
            status_code = 408

        assert is_retryable_error(FakeStatusError("timed out")) is True

    def test_408_code_attribute_is_retryable(self):
        """408 should also be retryable via the Google-style code attribute."""

        class FakeCodeError(Exception):
            code = 408

        assert is_retryable_error(FakeCodeError("timed out")) is True

    def test_builtin_transport_errors_are_retryable(self):
        """Bare transport exceptions (empty str()) hit the isinstance check,
        not the message fallback."""
        assert is_retryable_error(ConnectionResetError()) is True
        assert is_retryable_error(BrokenPipeError()) is True
        assert is_retryable_error(TimeoutError()) is True
        assert is_retryable_error(ssl.SSLError()) is True

    def test_httpx_transport_errors_are_retryable(self):
        """httpx transport failures (connect, read, TLS, timeout) are retryable.
        httpx.ReadError('') stringifies empty, so only isinstance catches it."""
        assert is_retryable_error(httpx.ReadError("")) is True
        assert is_retryable_error(httpx.ConnectError("connection refused")) is True
        assert is_retryable_error(httpx.ConnectTimeout("timed out")) is True

    def test_httpx2_transport_errors_are_retryable(self):
        """anthropic>=1.0 and openai>=3.0 raise httpx2 errors, not httpx ones."""
        assert is_retryable_error(httpx2.ReadError("")) is True
        assert is_retryable_error(httpx2.ConnectError("connection refused")) is True
        assert is_retryable_error(httpx2.ConnectTimeout("timed out")) is True

    def test_plain_application_errors_are_not_retryable(self):
        """Application-level errors must fail fast."""
        assert is_retryable_error(ValueError("invalid literal")) is False
        assert is_retryable_error(TypeError("bad type")) is False
        assert is_retryable_error(Exception()) is False

    def test_provider_processing_error_is_not_retryable(self):
        """Errors raised by response processing (not the API) must fail fast."""

        class FakeAppError(Exception):
            pass

        error = FakeAppError("No content found in response")
        assert is_retryable_error(error) is False


class _FlakyOperation:
    """Callable that raises the given errors in order, then returns a value."""

    def __init__(self, errors, result="success"):
        self.errors = list(errors)
        self.result = result
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.result


def _retryable_error():
    return ConnectionResetError("connection reset by peer")


class TestRunWithRetryConfig:
    """Tests for the schedule-driven retry executor."""

    def test_success_on_first_attempt_makes_no_sleep(self):
        operation = _FlakyOperation(errors=[])
        with patch("patterpunk.lib.retry.time.sleep") as sleep:
            result = run_with_retry_config(
                RetryConfig(delays_s=(10.0,)), operation, "Test"
            )
        assert result == "success"
        assert operation.calls == 1
        sleep.assert_not_called()

    def test_schedule_honored_with_jitter_bounds(self):
        """Each sleep must be delays_s[n] * uniform(0.5, 1.0)."""
        operation = _FlakyOperation(errors=[_retryable_error(), _retryable_error()])
        with patch("patterpunk.lib.retry.time.sleep") as sleep:
            result = run_with_retry_config(
                RetryConfig(delays_s=(10.0, 20.0, 40.0)), operation, "Test"
            )
        assert result == "success"
        assert operation.calls == 3
        sleep_durations = [call.args[0] for call in sleep.call_args_list]
        assert len(sleep_durations) == 2
        assert 5.0 <= sleep_durations[0] <= 10.0
        assert 10.0 <= sleep_durations[1] <= 20.0

    def test_deterministic_jitter_gives_exact_sleeps(self):
        operation = _FlakyOperation(
            errors=[_retryable_error(), _retryable_error(), _retryable_error()]
        )
        with patch("patterpunk.lib.retry.time.sleep") as sleep:
            run_with_retry_config(
                RetryConfig(delays_s=(10.0, 20.0, 40.0), jitter=(1.0, 1.0)),
                operation,
                "Test",
            )
        sleep_durations = [call.args[0] for call in sleep.call_args_list]
        assert sleep_durations == [10.0, 20.0, 40.0]

    def test_non_retryable_error_fails_fast(self):
        original = ValueError("bad request")
        operation = _FlakyOperation(errors=[original])
        with patch("patterpunk.lib.retry.time.sleep") as sleep:
            with pytest.raises(ValueError) as exc_info:
                run_with_retry_config(
                    RetryConfig(delays_s=(10.0, 20.0)), operation, "Test"
                )
        assert exc_info.value is original
        assert operation.calls == 1
        sleep.assert_not_called()

    def test_exhaustion_reraises_last_native_error(self):
        errors = [_retryable_error(), _retryable_error(), _retryable_error()]
        last_error = errors[-1]
        operation = _FlakyOperation(errors=errors)
        with patch("patterpunk.lib.retry.time.sleep") as sleep:
            with pytest.raises(ConnectionResetError) as exc_info:
                run_with_retry_config(
                    RetryConfig(delays_s=(10.0, 20.0)), operation, "Test"
                )
        assert exc_info.value is last_error
        assert operation.calls == 3
        assert len(sleep.call_args_list) == 2

    def test_empty_delays_makes_single_native_attempt(self):
        original = _retryable_error()
        operation = _FlakyOperation(errors=[original])
        with patch("patterpunk.lib.retry.time.sleep") as sleep:
            with pytest.raises(ConnectionResetError) as exc_info:
                run_with_retry_config(RetryConfig(delays_s=()), operation, "Test")
        assert exc_info.value is original
        assert operation.calls == 1
        sleep.assert_not_called()


async def _stream_of(items):
    for item in items:
        yield item


class _FlakyStreamFactory:
    """acquire_stream fake that raises the given errors in order, then streams."""

    def __init__(self, errors, items=("a", "b")):
        self.errors = list(errors)
        self.items = items
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return _stream_of(self.items)


class TestStreamWithRetryConfig:
    """Tests for the schedule-driven streaming retry executor."""

    async def test_failure_then_stream_yields_all_items(self):
        factory = _FlakyStreamFactory(errors=[_retryable_error()])
        with patch("patterpunk.lib.retry.asyncio.sleep") as sleep:
            items = [
                item
                async for item in stream_with_retry_config(
                    RetryConfig(delays_s=(10.0,)), factory, "Test"
                )
            ]
        assert items == ["a", "b"]
        assert factory.calls == 2
        assert len(sleep.call_args_list) == 1
        assert 5.0 <= sleep.call_args_list[0].args[0] <= 10.0

    async def test_exhaustion_reraises_last_native_error(self):
        errors = [_retryable_error(), _retryable_error()]
        last_error = errors[-1]
        factory = _FlakyStreamFactory(errors=errors)
        with patch("patterpunk.lib.retry.asyncio.sleep"):
            with pytest.raises(ConnectionResetError) as exc_info:
                async for _ in stream_with_retry_config(
                    RetryConfig(delays_s=(10.0,)), factory, "Test"
                ):
                    pass
        assert exc_info.value is last_error
        assert factory.calls == 2

    async def test_non_retryable_error_fails_fast(self):
        original = ValueError("bad request")
        factory = _FlakyStreamFactory(errors=[original])
        with patch("patterpunk.lib.retry.asyncio.sleep") as sleep:
            with pytest.raises(ValueError):
                async for _ in stream_with_retry_config(
                    RetryConfig(delays_s=(10.0,)), factory, "Test"
                ):
                    pass
        assert factory.calls == 1
        sleep.assert_not_called()

    async def test_mid_stream_failure_reyields_from_start(self):
        """Whole-stream retry: a failure after partial yield re-acquires the
        stream and re-yields from the start (accepted duplicate-yield wart)."""

        class MidStreamFailingFactory:
            def __init__(self):
                self.calls = 0

            async def __call__(self):
                self.calls += 1
                if self.calls == 1:
                    return self._failing_stream()
                return _stream_of(["a", "b"])

            async def _failing_stream(self):
                yield "a"
                raise _retryable_error()

        factory = MidStreamFailingFactory()
        with patch("patterpunk.lib.retry.asyncio.sleep"):
            items = [
                item
                async for item in stream_with_retry_config(
                    RetryConfig(delays_s=(10.0,)), factory, "Test"
                )
            ]
        assert items == ["a", "a", "b"]
        assert factory.calls == 2


class TestBackoffIntegration:
    """Integration tests for the retry backoff system."""

    def test_thundering_herd_prevention(self):
        """Verify jitter prevents synchronized retries.

        min_delay is 0 here: with the default 45s floor, jittered values below
        the floor all clamp to exactly 45.0, which collapses the very variation
        this test asserts on (and made it flaky).
        """
        # Simulate 10 clients hitting rate limit at the same time
        delays = [
            calculate_backoff_delay(
                attempt=0,
                base_delay=60.0,
                max_delay=300.0,
                min_delay=0.0,
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
