import logging
from typing import Optional

from mootdx.quotes import Quotes

from tdxdata.errors.exceptions import ConnectionError

logger = logging.getLogger(__name__)


class TdxConnection:
    def __init__(self):
        self._client = None
        self._initialized = False

    @property
    def client(self):
        if self._client is None:
            raise ConnectionError("mootdx not connected. Call initialize() first.")
        return self._client

    @property
    def is_connected(self) -> bool:
        return self._initialized

    def initialize(self, server: Optional[tuple] = None, timeout: int = 15) -> None:
        if self._initialized:
            logger.warning("Connection already initialized, skipping")
            return

        try:
            self._client = Quotes.factory(
                market="std",
                server=server,
                timeout=timeout,
                quiet=True,
            )
            self._initialized = True
            logger.info("TDX connection initialized via mootdx")
        except Exception as e:
            raise ConnectionError(f"Failed to initialize mootdx connection: {e}") from e

    def close(self) -> None:
        if self._initialized and self._client is not None:
            try:
                self._client.close()
                logger.info("TDX connection closed")
            except Exception as e:
                logger.error(f"Error closing TDX connection: {e}")
            finally:
                self._client = None
                self._initialized = False

    def reconnect(self) -> None:
        if not self._initialized:
            raise ConnectionError("Not connected. Call initialize() first.")
        self._client.reconnect()
        logger.info("TDX connection reconnected")
