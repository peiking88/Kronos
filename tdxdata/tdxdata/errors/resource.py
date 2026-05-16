import logging
from typing import Optional

from tdxdata.core.connection import TdxConnection

logger = logging.getLogger(__name__)


class ResourceManager:
    def __init__(self, connection: TdxConnection):
        self._connection = connection

    def __enter__(self):
        return self._connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._connection.close()
        if exc_type is not None:
            logger.error(f"Connection closed due to error: {exc_val}")
        return False
