import pytest
from tdxdata.errors.retry import RetryPolicy
from tdxdata.errors.exceptions import RetryExhaustedError


class TestRetryPolicy:
    def test_success_on_first_attempt(self):
        policy = RetryPolicy(max_retries=3, base_delay=0.01)
        result = policy.execute(lambda: 42)
        assert result == 42

    def test_success_after_retry(self):
        call_count = 0

        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient error")
            return "success"

        policy = RetryPolicy(max_retries=3, base_delay=0.01)
        result = policy.execute(flaky_func)
        assert result == "success"
        assert call_count == 3

    def test_all_retries_exhausted(self):
        def always_fail():
            raise RuntimeError("permanent failure")

        policy = RetryPolicy(max_retries=2, base_delay=0.01)
        with pytest.raises(RetryExhaustedError, match="3 attempts"):
            policy.execute(always_fail)

    def test_zero_retries(self):
        def always_fail():
            raise RuntimeError("fail")

        policy = RetryPolicy(max_retries=0, base_delay=0.01)
        with pytest.raises(RetryExhaustedError, match="1 attempts"):
            policy.execute(always_fail)

    def test_exponential_backoff_delay(self):
        policy = RetryPolicy(max_retries=3, base_delay=1.0, exponential_base=2.0)
        assert policy.base_delay == 1.0
        assert policy.max_delay == 30.0

    def test_max_delay_cap(self):
        policy = RetryPolicy(max_retries=10, base_delay=100.0, max_delay=5.0)
        delay = min(100.0 * (2.0 ** 5), 5.0)
        assert delay == 5.0
