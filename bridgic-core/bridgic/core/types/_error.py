###########################################################
# Worker Errors
###########################################################

class WorkerSignatureError(Exception):
    """
    Raised when the signature of a worker is not valid.
    """
    pass

class WorkerArgsMappingError(Exception):
    """
    Raised when the parameter declaration format does not meet the requirements of the
    setting of the arguments mapping rule.
    """
    pass

class WorkerArgsInjectionError(Exception):
    """
    Raised when the arguments injection mechanism encountered an error during operation.
    """
    pass

class WorkerRuntimeError(RuntimeError):
    """
    Raised when the worker encounters an unexpected error during runtime.
    """
    pass

###########################################################
# Automa Errors
###########################################################

class AutomaDeclarationError(Exception):
    """
    Raised when the orchestration of workers within an Automa is not valid.
    """
    pass

class AutomaCompilationError(Exception):
    """
    Raised when the compilation of an Automa does not pass.
    """
    pass

class AutomaRuntimeError(RuntimeError):
    """
    Raised when the execution of an Automa encounters an unexpected error.
    """
    pass

###########################################################
# Prompt Errors
###########################################################

class PromptSyntaxError(Exception):
    pass

class PromptRenderError(RuntimeError):
    pass