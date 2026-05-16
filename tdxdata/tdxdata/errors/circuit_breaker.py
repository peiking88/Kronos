import time
import logging
import threading
from typing import Optional

from tdxdata.errors.exceptions import CircuitBreakerOpenError

logger = logging.getLogger(__name__)


class CircuitBreaker:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.RLock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN:
                if (
                    self._last_failure_time
                    and time.time() - self._last_failure_time >= self.recovery_timeout
                ):
                    self._state = self.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("Circuit breaker: OPEN -> HALF_OPEN")
            return self._state

    def allow_request(self) -> bool:
        with self._lock:
            state = self.state
            if state == self.CLOSED:
                return True
            if state == self.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            return False

    def record_success(self) -> None:
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._state = self.CLOSED
                self._failure_count = 0
                logger.info("Circuit breaker: HALF_OPEN -> CLOSED")
            self._failure_count = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._state == self.HALF_OPEN:
                self._state = self.OPEN
                logger.info("Circuit breaker: HALF_OPEN -> OPEN")
            elif self._failure_count >= self.failure_threshold:
                self._state = self.OPEN
                logger.info(
                    f"Circuit breaker: CLOSED -> OPEN (failures: {self._failure_count})"
                )

    def reset(self) -> None:
        with self._lock:
            self._state = self.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0
