class ASLError(Exception):
    """
    Base error class for ASL.
    """
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class PythonCompilationError(ASLError):
    """
    Python code compilation error.
    """
    def __init__(self, message: str, source_code: str = None, line_number: int = None):
        self.source_code = source_code
        self.line_number = line_number
        super().__init__(message, "COMPILE_ERROR")


class ASLCompilationError(ASLError):
    """
    ASL code compilation error.
    """
    def __init__(self, message: str):
        super().__init__(message, "ASL_COMPILE_ERROR")

    
class ASLWorkerNameNotFoundError(ASLError):
    """
    Worker name not found error.
    """
    def __init__(self, worker_name: str):
        self.worker_name = worker_name
        super().__init__(f"Worker name {worker_name} not found, please check the worker name in the ASL code and wroker name in the python definition")


class ASLUnmatchedBraceError(ASLError):
    """
    Unmatched brace error.
    """
    def __init__(self, brace_type: str, brace_position: int):
        self.brace_type = brace_type
        self.brace_position = brace_position
        super().__init__(f"Unmatched {brace_type} brace at position {brace_position}")
