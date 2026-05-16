import time
import logging
from typing import Callable, TypeVar, Optional

from tdxdata.errors.exceptions import RetryExhaustedError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryPolicy:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay,
                    )
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {self.max_retries + 1} attempts exhausted. Last error: {e}"
                    )
        raise RetryExhaustedError(
            f"Operation failed after {self.max_retries + 1} attempts"
        ) from last_exception
