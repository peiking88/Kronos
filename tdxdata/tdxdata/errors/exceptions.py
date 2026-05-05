class TdxDataError(Exception):
    pass


class ConnectionError(TdxDataError):
    pass


class DataFetchError(TdxDataError):
    pass


class StorageError(TdxDataError):
    pass


class PluginNotFoundError(TdxDataError):
    pass


class RetryExhaustedError(TdxDataError):
    pass


class CircuitBreakerOpenError(TdxDataError):
    pass
