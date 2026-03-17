from typing import Optional


class ModelRetryLimitError(RuntimeError):
    """
    Raised when recoverable errors persist until retry budget is exhausted.
    """

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.operation = operation
        self.original_exception = original_exception


class ModelUnrecoverableError(RuntimeError):
    """
    Raised when an exception is classified as non-retryable.
    """

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.operation = operation
        self.original_exception = original_exception
