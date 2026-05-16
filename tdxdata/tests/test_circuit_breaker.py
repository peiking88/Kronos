import time
import pytest
from tdxdata.errors.circuit_breaker import CircuitBreaker


class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitBreaker.CLOSED

    def test_allow_request_when_closed(self):
        cb = CircuitBreaker()
        assert cb.allow_request() is True

    def test_transitions_to_open_after_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN

    def test_rejects_requests_when_open(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN
        assert cb.allow_request() is False

    def test_transitions_to_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN

        time.sleep(0.02)
        assert cb.state == CircuitBreaker.HALF_OPEN

    def test_half_open_allows_limited_requests(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01, half_open_max_calls=1)
        cb.record_failure()
        time.sleep(0.02)

        assert cb.state == CircuitBreaker.HALF_OPEN
        assert cb.allow_request() is True
        assert cb.allow_request() is False

    def test_half_open_to_closed_on_success(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)

        cb.state
        cb.record_success()
        assert cb.state == CircuitBreaker.CLOSED

    def test_half_open_to_open_on_failure(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)

        cb.state
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN

    def test_reset(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        cb.reset()
        assert cb.state == CircuitBreaker.CLOSED
        assert cb.allow_request() is True
